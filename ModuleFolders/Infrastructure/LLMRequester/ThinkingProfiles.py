"""各 OpenAI 兼容平台的思考参数方言，做成可注册的 profile。

为什么要有这一层（技术评审团的结论）：

* 原来判定"这是哪个平台"靠的是散落各处的字符串嗅探 —— tag 前缀加 URL 子串，
  而且分布在请求器、界面、模型浏览器三处，各写各的。同一件事有五种写法，
  `sakura` 那种等值比较对 `sakura_123456` 恒为假就是踩过的坑。
* 现在改成：**数据里写一个 profile 名字，profile 怎么算留在本模块的注册表里**。
  嗅探只剩 resolve_profile 一处，且只作为没有显式声明时的兜底。
* 本模块不导入 PyQt5 / openai，所以方言逻辑可以被纯 unittest 覆盖，
  也才能用表征基线逐字节证明迁移没有改变行为。
"""

import copy

# 各平台使用的 profile 名字（preset.json 的 "profile" 字段写这些值）
PROFILE_VOLCENGINE = "volcengine"
PROFILE_OPENROUTER = "openrouter"
PROFILE_XAI = "xai"
PROFILE_OPENAI = "openai"
PROFILE_DEEPSEEK = "deepseek"
PROFILE_ZHIPU = "zhipu"
PROFILE_DASHSCOPE = "dashscope"
PROFILE_DEFAULT = "default"

VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class ThinkingContext:
    """一次思考参数装配所需的全部输入，避免各 profile 反复解包配置。"""

    __slots__ = ("params", "extra_body", "platform_config", "think_switch", "think_depth", "model_lower")

    def __init__(self, base_params: dict, platform_config: dict):
        platform_config = platform_config if isinstance(platform_config, dict) else {}
        self.platform_config = platform_config
        self.params = copy.deepcopy(base_params)

        # extra_body 中可能已有用户自定义参数，复制后再合并平台专用字段
        raw_extra_body = self.params.get("extra_body", {})
        self.extra_body = copy.deepcopy(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
        self.params["extra_body"] = self.extra_body

        self.think_switch = bool(platform_config.get("think_switch"))
        self.think_depth = platform_config.get("think_depth") or "medium"
        model_name = str(platform_config.get("model_name") or self.params.get("model") or "")
        self.model_lower = model_name.lower()

    def nested(self, key: str) -> dict:
        """取 extra_body 里的子对象，非 dict 时重置为空 dict。"""
        value = self.extra_body.get(key, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# 各平台方言。行为与迁移前逐字节一致，由表征基线 tests/fixtures 守住。
# --------------------------------------------------------------------------

def _profile_volcengine(ctx: ThinkingContext) -> dict:
    effort = ctx.think_depth if ctx.think_depth in VALID_EFFORTS else "medium"

    thinking = ctx.nested("thinking")
    thinking["type"] = "enabled" if ctx.think_switch else "disabled"
    ctx.extra_body["thinking"] = thinking

    # 清理自定义请求体或调用方遗留的同名字段，避免与界面设置冲突。
    ctx.params.pop("reasoning_effort", None)
    ctx.extra_body.pop("reasoning_effort", None)
    if ctx.think_switch:
        ctx.params["reasoning_effort"] = effort
    return ctx.params


def _profile_openrouter(ctx: ThinkingContext) -> dict:
    # OpenRouter 的思考强度走原生 reasoning.effort；它与顶层 reasoning_effort 是等价
    # 写法且不允许同时出现，所以只发 reasoning。关闭思考用 effort=none。
    valid_efforts = {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
    effort = ctx.think_depth if ctx.think_depth in valid_efforts else "medium"
    if not ctx.think_switch:
        effort = "none"

    # 保留用户自己在 extra_body 里写的 reasoning 附加项（如 summary）。
    reasoning = ctx.nested("reasoning")
    reasoning["effort"] = effort
    ctx.extra_body["reasoning"] = reasoning

    ctx.params.pop("reasoning_effort", None)
    ctx.extra_body.pop("reasoning_effort", None)
    return ctx.params


def _profile_xai(ctx: ThinkingContext) -> dict:
    # Grok 4.6 的推理无法关闭，始终按界面强度发送 reasoning_effort。
    ctx.params["reasoning_effort"] = (
        ctx.think_depth if ctx.think_depth in {"low", "medium", "high", "xhigh"} else "xhigh"
    )
    return ctx.params


def _profile_openai(ctx: ThinkingContext) -> dict:
    if not ctx.think_switch:
        if ctx.model_lower.startswith("gpt-5.6"):
            ctx.params["reasoning_effort"] = "none"
        return ctx.params

    # 推理模型使用顶层 reasoning_effort，普通模型不传该字段
    if ctx.model_lower.startswith(("o1", "o3", "o4", "gpt-5")):
        ctx.params["reasoning_effort"] = (
            ctx.think_depth if ctx.think_depth in VALID_EFFORTS else "medium"
        )
        # 新一代推理模型开启 reasoning 时不发送采样温度。
        ctx.params.pop("temperature", None)
    return ctx.params


def _profile_deepseek(ctx: ThinkingContext) -> dict:
    if not ctx.think_switch:
        ctx.extra_body["thinking"] = {"type": "disabled"}
        return ctx.params

    # low/medium/high 统一映射为 high，xhigh/max 映射为 max
    ctx.params["reasoning_effort"] = "max" if ctx.think_depth in {"xhigh", "max"} else "high"
    ctx.extra_body["thinking"] = {"type": "enabled"}
    return ctx.params


def _profile_zhipu(ctx: ThinkingContext) -> dict:
    is_glm_new = ctx.model_lower.startswith(("glm-4.5", "glm-4.6", "glm-4.7", "glm-5"))

    if not ctx.think_switch:
        if is_glm_new:
            ctx.extra_body["thinking"] = {"type": "disabled"}
        return ctx.params

    # GLM 新系列支持 extra_body.thinking，旧模型保持默认参数
    if is_glm_new:
        ctx.extra_body["thinking"] = {"type": "enabled"}
    if ctx.model_lower.startswith("glm-5.2"):
        ctx.params["reasoning_effort"] = (
            ctx.think_depth if ctx.think_depth in VALID_EFFORTS else "high"
        )
    return ctx.params


def _profile_dashscope(ctx: ThinkingContext) -> dict:
    if not ctx.think_switch:
        if ctx.model_lower.startswith("qwen3"):
            ctx.extra_body["enable_thinking"] = False
        return ctx.params

    # 兼容模式使用 enable_thinking，并可选传入 thinking_budget
    ctx.extra_body["enable_thinking"] = True
    try:
        thinking_budget = int(ctx.platform_config.get("thinking_budget"))
    except (TypeError, ValueError):
        thinking_budget = None
    if thinking_budget is not None and thinking_budget >= 0:
        ctx.extra_body["thinking_budget"] = thinking_budget
    return ctx.params


def _profile_default(ctx: ThinkingContext) -> dict:
    if not ctx.think_switch:
        return ctx.params
    ctx.params["reasoning_effort"] = ctx.think_depth
    return ctx.params


PROFILES = {
    PROFILE_VOLCENGINE: _profile_volcengine,
    PROFILE_OPENROUTER: _profile_openrouter,
    PROFILE_XAI: _profile_xai,
    PROFILE_OPENAI: _profile_openai,
    PROFILE_DEEPSEEK: _profile_deepseek,
    PROFILE_ZHIPU: _profile_zhipu,
    PROFILE_DASHSCOPE: _profile_dashscope,
    PROFILE_DEFAULT: _profile_default,
}

# 嗅探顺序必须与迁移前一致：volcengine 最先，之后 openrouter/xai，
# 再是 openai/deepseek/zhipu/dashscope，最后落到 default。
_LEGACY_DETECTORS = (
    (PROFILE_VOLCENGINE, ("volcengine",), ("volces.com", "volcengine")),
    (PROFILE_OPENROUTER, ("openrouter",), ("openrouter.ai",)),
    (PROFILE_XAI, ("xai",), ("api.x.ai",)),
    (PROFILE_OPENAI, ("openai",), ("api.openai.com",)),
    (PROFILE_DEEPSEEK, ("deepseek",), ("api.deepseek.com",)),
    (PROFILE_ZHIPU, ("zhipu",), ("bigmodel.cn",)),
    (
        PROFILE_DASHSCOPE,
        ("dashscope",),
        ("dashscope.aliyuncs.com", "bailian"),
    ),
)


def _legacy_profile_name(target_platform: str, api_url: str) -> str:
    """存量配置没有 profile 字段时的兜底嗅探。

    只在这一处保留字符串嗅探，新配置应当写 profile。
    """
    for name, tag_prefixes, url_fragments in _LEGACY_DETECTORS:
        if target_platform.startswith(tag_prefixes):
            return name
        if any(fragment in api_url for fragment in url_fragments):
            return name
    if "aliyuncs.com" in api_url and "compatible-mode" in api_url:
        return PROFILE_DASHSCOPE
    return PROFILE_DEFAULT


def resolve_profile_name(platform_config) -> str:
    """决定用哪个 profile：显式声明 > preset 名 > 兜底嗅探。"""
    platform_config = platform_config if isinstance(platform_config, dict) else {}

    declared = platform_config.get("profile")
    if isinstance(declared, str) and declared in PROFILES:
        return declared

    preset_key = platform_config.get("preset_key")
    if isinstance(preset_key, str) and preset_key in PROFILES:
        return preset_key

    target_platform = str(
        platform_config.get("target_platform") or platform_config.get("tag") or ""
    ).lower()
    api_url = str(platform_config.get("api_url") or "").lower()
    return _legacy_profile_name(target_platform, api_url)


def build_thinking_params(base_params: dict, platform_config: dict) -> dict:
    """按解析出的 profile 装配思考参数。"""
    profile = PROFILES.get(resolve_profile_name(platform_config), _profile_default)
    return profile(ThinkingContext(base_params, platform_config))
