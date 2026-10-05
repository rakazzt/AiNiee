"""声明式接口选项：解析、校验与安全写入请求体。

设计边界（来自技术评审团的结论）：

* 数据只负责「把哪个值写进哪个受控槽位」，不表达逻辑、不命名任意路径。
  因此路径必须落在本模块代码侧的允许表里，写入只做逐段字典走查，
  绝不 eval / setattr —— preset.json 是被鼓励做 drop-in 替换的数据文件，
  它一旦被改坏不能变成代码执行或静默改写 model / messages 的入口。
* 条件显示是封闭谓词（option 真值、成员、模型名前缀），不是表达式求值。
* 选项值以扁平标量存在 config["platforms"][key] 顶层，绝不再持久化
  「合并后的请求体」；否则回退到旧版本时会二次合并。

本模块不导入 PyQt5 / openai / 任何第三方库，这样纯 unittest 就能覆盖它，
CI 也能在不启动 GUI 的前提下守住这些规则。
"""

import copy
import json
import re

# 允许写入的请求体顶层键：数据只能在这些槽位里选，不能自己发明路径。
ALLOWED_BODY_ROOTS = frozenset(
    {
        "reasoning",
        "reasoning_effort",
        "provider",
        "models",
        "thinking",
        "enable_thinking",
        "thinking_budget",
        "temperature",
        "top_p",
        "max_tokens",
        "min_p",
        "seed",
    }
)

# 传输层与身份相关键：任何情况下都不允许由选项数据决定。
RESERVED_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "timeout",
        "api_key",
        "api_url",
        "base_url",
        "headers",
        "extra_headers",
        "client_args",
        "http_client",
    }
)

VALID_VEHICLES = frozenset({"param", "extra_body"})
VALID_TYPES = frozenset(
    {"bool", "enum", "int", "float", "string", "string-list", "json", "number-or-object"}
)
VALID_MERGES = frozenset({"leaf", "replace"})

# 必须以字母开头：直接挡掉 dunder 与私有名（__class__ / _private），
# 即便写入只做字典操作、并不做属性访问，也不给这类键留位置。
_SEGMENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
MAX_PATH_SEGMENTS = 3


class OptionSchemaError(ValueError):
    """描述符本身不合法。加载期出现时应当忽略该选项并记日志，而不是让界面打不开。"""


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def resolve_preset_key(platform_config, preset_platforms) -> str | None:
    """把用户配置里的平台条目对应回 preset 里的 tag。

    平台实例的 key 是生成的（PlatformPage 写成 tag_123456），而**老版本建的**
    实例里没有 options/docs 快照，所以 schema 必须按 tag 回到 Resource 里查，
    不能只按实例 key 查表 —— 否则存量用户拿不到任何新选项。
    """
    platform_config = _as_dict(platform_config)
    preset_platforms = _as_dict(preset_platforms)

    explicit = platform_config.get("preset_key")
    if isinstance(explicit, str) and explicit in preset_platforms:
        return explicit

    for candidate in (platform_config.get("tag"), platform_config.get("key")):
        if isinstance(candidate, str) and candidate in preset_platforms:
            return candidate

    # tag_123456 -> tag（取最长匹配，避免 openrouter 与 openrouterx 串味）
    for candidate in (platform_config.get("tag"), platform_config.get("key")):
        if not isinstance(candidate, str) or "_" not in candidate:
            continue
        head = candidate.rsplit("_", 1)[0]
        if head in preset_platforms:
            return head

    return None


def resolve_options(platform_config, preset_platforms) -> list:
    """返回该平台声明的选项列表；没有声明就返回空列表。"""
    preset_key = resolve_preset_key(platform_config, preset_platforms)
    if preset_key is None:
        return []
    options = _as_dict(preset_platforms).get(preset_key, {}).get("options")
    return options if isinstance(options, list) else []


def validate_descriptor(descriptor) -> None:
    """校验单个描述符，不合法就抛 OptionSchemaError（调用方决定忽略还是中止）。"""
    if not isinstance(descriptor, dict):
        raise OptionSchemaError("descriptor must be an object")

    key = descriptor.get("key")
    if not isinstance(key, str) or not key.strip():
        raise OptionSchemaError("descriptor.key must be a non-empty string")

    option_type = descriptor.get("type")
    if option_type not in VALID_TYPES:
        raise OptionSchemaError(f"{key}: unknown type {option_type!r}")

    vehicle = descriptor.get("vehicle", "extra_body")
    if vehicle not in VALID_VEHICLES:
        raise OptionSchemaError(f"{key}: unknown vehicle {vehicle!r}")

    merge = descriptor.get("merge", "leaf")
    if merge not in VALID_MERGES:
        raise OptionSchemaError(f"{key}: unknown merge {merge!r}")

    path = descriptor.get("path")
    if not isinstance(path, str) or not path.strip():
        raise OptionSchemaError(f"{key}: path is required")
    _validate_path(key, path)

    if option_type == "enum":
        values = descriptor.get("values")
        if not isinstance(values, list) or not values:
            raise OptionSchemaError(f"{key}: enum requires a non-empty values list")
        if "default" in descriptor and descriptor["default"] not in values:
            raise OptionSchemaError(f"{key}: default {descriptor['default']!r} not in values")

    when = descriptor.get("when", [])
    if not isinstance(when, list):
        raise OptionSchemaError(f"{key}: when must be a list")

    mapping = descriptor.get("map", {})
    if not isinstance(mapping, dict):
        raise OptionSchemaError(f"{key}: map must be an object")


def validate_schema(options) -> list:
    """校验整份 schema，返回错误信息列表（空列表表示通过）。

    这是热补丁路径唯一的守门人：preset.json 可以绕过 CI 被替换，
    所以 schema 必须在运行期自检，而不是只在开发期靠人眼。
    """
    errors = []
    if not isinstance(options, list):
        return ["options must be a list"]

    seen_keys = {}
    seen_paths = {}
    for index, descriptor in enumerate(options):
        try:
            validate_descriptor(descriptor)
        except OptionSchemaError as error:
            errors.append(f"[{index}] {error}")
            continue

        key = descriptor["key"]
        if key in seen_keys:
            errors.append(f"{key}: duplicate option key")
        seen_keys[key] = descriptor

        vehicle = descriptor.get("vehicle", "extra_body")
        path = f"{vehicle}:{descriptor['path']}"
        if path in seen_paths:
            errors.append(f"{key}: path {path} already written by {seen_paths[path]}")
        seen_paths[path] = key

    for key, descriptor in seen_keys.items():
        for predicate in descriptor.get("when", []):
            referenced = _as_dict(predicate).get("option")
            if isinstance(referenced, str) and referenced not in seen_keys:
                errors.append(f"{key}: when references unknown option {referenced!r}")

    return errors


def _validate_path(key, path) -> None:
    segments = path.split(".")
    if len(segments) > MAX_PATH_SEGMENTS:
        raise OptionSchemaError(f"{key}: path too deep ({len(segments)} segments)")
    for segment in segments:
        if not _SEGMENT_RE.match(segment):
            raise OptionSchemaError(f"{key}: illegal path segment {segment!r}")
        if segment in RESERVED_BODY_KEYS:
            raise OptionSchemaError(f"{key}: path may not touch reserved key {segment!r}")
    if segments[0] not in ALLOWED_BODY_ROOTS:
        raise OptionSchemaError(
            f"{key}: path root {segments[0]!r} is not in the allowed slot list"
        )


def write_path(target: dict, path: str, value, merge: str = "leaf") -> None:
    """把 value 写进 target 的 path，逐段走查字典，不做任何属性访问。

    中间节点必须已经是 dict（或不存在）；遇到标量就拒绝，避免把已配置好的
    结构整块覆盖掉。
    """
    segments = path.split(".")
    node = target
    for segment in segments[:-1]:
        existing = node.get(segment)
        if existing is None:
            node[segment] = {}
        elif not isinstance(existing, dict):
            raise OptionSchemaError(
                f"cannot descend into {segment!r}: existing value is {type(existing).__name__}"
            )
        node = node[segment]

    leaf = segments[-1]
    if merge == "replace" or leaf not in node:
        node[leaf] = value
        return
    # leaf 合并：保留同层其它键（例如 reasoning.summary），只覆盖这一个叶子
    existing = node[leaf]
    if isinstance(existing, dict) and isinstance(value, dict):
        merged = dict(existing)
        merged.update(value)
        node[leaf] = merged
    else:
        node[leaf] = value


def matches_when(descriptor, values) -> bool:
    """封闭谓词判定：只做键名真值、相等、成员与模型前缀匹配，绝不求值。"""
    for predicate in descriptor.get("when", []) or []:
        predicate = _as_dict(predicate)
        if "option" in predicate:
            actual = _as_dict(values).get(predicate["option"])
            if "eq" in predicate and actual != predicate["eq"]:
                return False
            if "in" in predicate:
                allowed = predicate["in"]
                if not isinstance(allowed, list) or actual not in allowed:
                    return False
            if "truthy" in predicate and bool(actual) != bool(predicate["truthy"]):
                return False
            continue
        if "model_matches" in predicate:
            model = str(_as_dict(values).get("model") or "")
            pattern = str(predicate["model_matches"])
            if not model.lower().startswith(pattern.lower()):
                return False
            continue
        return False
    return True


def build_option_body(platform_config, preset_platforms) -> dict:
    """把已声明且已填写的选项折算成要合并进请求的参数。

    返回 {"param": {...}, "extra_body": {...}}，由调用方分别合并；
    未填写（None / 缺键）的选项不会出现在结果里 —— 不能发 null 或空串。
    """
    platform_config = _as_dict(platform_config)
    result = {"param": {}, "extra_body": {}}

    options = resolve_options(platform_config, preset_platforms)
    if not options:
        return result

    values = dict(platform_config)
    for descriptor in options:
        try:
            validate_descriptor(descriptor)
        except OptionSchemaError:
            continue  # 坏描述符只跳过，不影响其它选项与界面构造

        key = descriptor["key"]
        if key not in platform_config:
            continue
        value = platform_config.get(key)
        if value is None or value == "" or value == []:
            continue
        if not matches_when(descriptor, values):
            continue

        mapping = descriptor.get("map") or {}
        # 只有可哈希的标量才谈得上查表；list/dict 值直接跳过映射
        if isinstance(mapping, dict) and isinstance(value, (str, int, float, bool)):
            if value in mapping:
                value = mapping[value]

        vehicle = descriptor.get("vehicle", "extra_body")
        target = result[vehicle]
        try:
            write_path(target, descriptor["path"], value, descriptor.get("merge", "leaf"))
        except OptionSchemaError:
            continue

    return result


def localized_text(value, language: str = "简中", fallback: str = "") -> str:
    """取多语言文本。

    选项标签与将来的 provider 文档都自带语言映射（{"简中": ..., "English": ...}），
    而不是走 tra() —— tra() 的 key 是中文原文，每改一次文档都要同步改 10 个
    本地化文件，"Resource 单文件热补丁"这个卖点就没了。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get(language)
        if isinstance(text, str) and text:
            return text
        for candidate in value.values():
            if isinstance(candidate, str) and candidate:
                return candidate
    return fallback


# 描述符 type -> 界面控件种类。界面层只做这一层映射，判断逻辑留在这里可测。
_WIDGET_BY_TYPE = {
    "bool": "switch",
    "enum": "combo",
    "string": "line",
    "int": "line",
    "float": "line",
    "string-list": "list",
    "json": "json",
    "number-or-object": "json",
}

# 组合框里代表「不设置」的哨兵：不设置意味着交给服务端默认值，
# 没有这个选项用户就无法把值改回默认。
UNSET_CHOICE = ""


def option_widget_plan(descriptor, language: str = "简中") -> dict:
    """把描述符折算成界面所需的信息（Qt-free，因此可以被单测覆盖）。

    返回 widget 种类、标题、说明与候选项；界面层据此挑控件，不再自己判断类型。
    """
    option_type = descriptor.get("type")
    widget = _WIDGET_BY_TYPE.get(option_type, "line")
    key = descriptor.get("key", "")

    choices = []
    if widget == "combo":
        # 第一项固定是「不设置」
        choices = [UNSET_CHOICE] + [
            str(v) for v in descriptor.get("values", []) if str(v) != UNSET_CHOICE
        ]

    return {
        "key": key,
        "widget": widget,
        "label": localized_text(descriptor.get("label"), language, key),
        "desc": localized_text(descriptor.get("desc"), language, ""),
        "choices": choices,
        "numeric": option_type in ("int", "float"),
    }


def encode_option_value(descriptor, raw_text: str):
    """把界面上的原始文本转成要存进 config 的值。

    返回 (ok, value)；ok 为 False 时调用方必须提示并且**不要写盘**，
    不能让半成品输入覆盖已经存好的值。空文本代表「不设置」，返回 (True, None)。
    """
    option_type = descriptor.get("type")
    text = (raw_text or "").strip()
    if not text:
        return True, None

    try:
        if option_type == "string-list":
            items = [part.strip() for part in text.replace("\n", ",").split(",")]
            return True, [item for item in items if item]
        if option_type == "int":
            return True, int(text)
        if option_type == "float":
            value = float(text)
            # nan/inf 会让 rapidjson 写出的 config 读不回来，
            # 而 Config.load_config 遇到解析失败会把 config 改名 .corrupt 并清空 ——
            # 用户的全部平台与密钥会在应用内消失，所以这里必须挡住。
            if value != value or value in (float("inf"), float("-inf")):
                return False, None
            return True, value
        if option_type in ("json", "number-or-object"):
            return True, json.loads(text)
    except (ValueError, TypeError):
        return False, None

    return True, text


def decode_option_value(descriptor, value) -> str:
    """把 config 里的值转回界面文本。"""
    if value is None:
        return ""
    if descriptor.get("type") == "string-list" and isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if descriptor.get("type") in ("json", "number-or-object") and not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def option_defaults(options) -> dict:
    """描述符里声明的唯一权威默认值，供投影层补齐缺失键。"""
    defaults = {}
    for descriptor in options or []:
        if not isinstance(descriptor, dict):
            continue
        key = descriptor.get("key")
        if isinstance(key, str) and "default" in descriptor:
            defaults[key] = descriptor["default"]
    return defaults


def merge_into(target: dict, overlay: dict) -> dict:
    """把 overlay 递归合并进 target（overlay 胜出），保留 target 的同层其它键。"""
    for key, value in _as_dict(overlay).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_into(target[key], value)
        else:
            target[key] = value
    return target


def apply_options_to_params(params: dict, platform_config, preset_platforms) -> dict:
    """把声明式选项折算进请求参数，就地返回 params。

    这是「UI 存了值、请求里却没有」这条断链的修复点：TaskConfig 每轮请求都会
    调用它。没有声明 options 的平台拿到空 overlay，params 一个字节都不变，
    因此存量 13 个 provider 的行为不受影响。

    优先级沿用现有语义：extra_body 打底，选项值按叶子覆盖。
    params["extra_body"] 是平台配置里的对象引用，先深拷贝再合并，否则会把
    合并结果写回 config。
    """
    option_body = build_option_body(platform_config, preset_platforms)

    # 必须深拷贝：dict() 只是浅拷贝，嵌套的 provider/... 对象仍与 config 共享，
    # merge_into 递归写下去就会把合并结果写回用户存的那份 extra_body，
    # 下一次 save_config 就把它持久化了 —— 回退到旧版本时会被二次合并。
    raw_extra_body = params.get("extra_body")
    extra_body = copy.deepcopy(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
    params["extra_body"] = merge_into(extra_body, option_body["extra_body"])

    merge_into(params, option_body["param"])
    return params
