"""提示词前缀缓存（prompt caching）：把「不变的前缀」变成可复用的缓存。

为什么需要这一层：AiNiee 每个批次都要重发同一份系统提示词，而这份提示词里
混着两类内容 —— 整份任务逐字节不变的部分（基础提示词 / 世界书 / 写作风格 /
翻译示例），和按当前批次筛选出来的部分（术语表 / 禁翻表 / 项目角色术语表）。
前缀缓存只做严格前缀匹配，所以不变的部分必须连成一段排在最前面；按官方
文档，一旦变化的内容插在中间：

- Anthropic 会退化成「每轮都在写缓存、永远读不到」——写入按 1.25 倍计费，
  比不开缓存还贵；
- OpenAI / DeepSeek / Gemini 的自动缓存只能命中那段基础提示词，而它通常
  远低于 1024 token 的最小可缓存长度，等于完全没有缓存。

这里只放 Qt-free 的纯逻辑（可被单测直接覆盖），请求器只负责把结果塞进
各自的报文格式。

官方口径（2026-04 核对）：
- Anthropic：显式 ``cache_control: {"type": "ephemeral"}``；写入 1.25 倍（5 分钟）
  / 2 倍（1 小时）、读取 0.1 倍；低于最小可缓存长度（按模型 512~4096 token）
  时静默不缓存、不报错；``usage.input_tokens`` **不含**缓存读写。
- OpenAI：自动前缀缓存，最小 1024 token（GPT-5.6+），读取 0.1 倍 / 写入 1.25 倍，
  命中量在 ``usage.prompt_tokens_details.cached_tokens``。
- DeepSeek：默认自动缓存，命中量在 ``usage.prompt_cache_hit_tokens``。
- OpenRouter：把 ``cache_control`` 透传给 Anthropic / 阿里云百炼等需要显式
  断点的 provider，并对同一模型的后续请求做 provider 粘性路由。
"""

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# 各家 usage 的累计口径，只用于把「缓存到底有没有生效」显示出来
_usage_totals = {"read": 0, "write": 0, "uncached": 0, "requests": 0, "cacheable_requests": 0}


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _int_of(source, *names) -> int:
    """从 dict 或 SDK 对象里取第一个能转成 int 的字段，取不到按 0。"""
    for name in names:
        raw = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


def supports_explicit_cache(platform_config) -> bool:
    """只有「必须显式声明缓存断点」的 provider 才注入 cache_control。

    OpenAI / DeepSeek / Gemini 2.5 是自动前缀缓存：多送一个 cache_control
    既不会提高命中率，还可能被严格校验字段的中转站直接 400，所以一律不送。
    """
    config = _as_dict(platform_config)
    if config.get("api_format") == "Anthropic":
        return True

    # OpenRouter 会做字段归一化，并把断点透传给需要它的上游 provider
    endpoint = f"{config.get('target_platform') or ''} {config.get('api_url') or ''}".lower()
    if "openrouter" not in endpoint:
        return False

    # 但 Gemini 例外：它的缓存把 systemInstruction 视为不可变，同一个 system 消息里
    # 「稳定块 + 每批筛选的术语表」会被整体固化，术语表后续变化不再生效（OpenRouter
    # 文档明确写了这一点，并要求把变化内容挪到后面的 user 消息）。这里退回单字符串，
    # 让 Gemini 自己的隐式缓存去处理稳定前缀，宁可少拿一点折扣，也不能让术语表失效。
    return "gemini" not in str(config.get("model_name") or "").lower()


def build_system_blocks(system_stable: str, system_full: str) -> list[dict]:
    """把系统提示词切成 [稳定块(带断点), 变化块] 两个文本块。

    ``system_stable`` 为空（或不是 ``system_full`` 的前缀）时返回空列表，
    调用方应退回原来的单字符串形式 —— 宁可不缓存，也不能切错位置。
    """
    stable = system_stable or ""
    if not stable.strip() or not (system_full or "").startswith(stable):
        return []

    blocks = [{"type": "text", "text": stable, "cache_control": dict(CACHE_CONTROL_EPHEMERAL)}]

    tail = system_full[len(stable):]
    if tail:
        blocks.append({"type": "text", "text": tail})

    return blocks


def anthropic_usage_totals(usage) -> tuple[int, int, int]:
    """返回 ``(cache_read, cache_write, total_input)``。

    Anthropic 的 ``input_tokens`` 只统计断点之后、既没读也没写的 token；
    直接拿它当 prompt_tokens 会把缓存命中和写入全部漏掉 —— 开了缓存之后
    token 统计反而变小，看着像省钱，其实是少算了。
    """
    read = _int_of(usage, "cache_read_input_tokens")
    write = _int_of(usage, "cache_creation_input_tokens")
    return read, write, read + write + _int_of(usage, "input_tokens")


def openai_cache_hit_tokens(usage) -> int:
    """自动前缀缓存下「命中缓存」的输入 token 数。

    OpenAI（``prompt_tokens_details.cached_tokens``）、DeepSeek
    （``prompt_cache_hit_tokens``）、OpenRouter（同一套字段）都覆盖；
    取不到就是 0，不影响 prompt_tokens 本身的统计（它已经包含命中部分）。
    """
    hit = _int_of(usage, "prompt_cache_hit_tokens")
    if hit:
        return hit

    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    return _int_of(details, "cached_tokens")


_announced = False


def announce_explicit_cache(system_stable: str, platform_config, log) -> None:
    """每次运行只提示一次：断点落在哪里、稳定前缀有多长。

    没有这行日志，用户无法判断「缓存到底生效了没有」—— 前缀缓存不命中时不会
    报错，只会安静地按原价计费。
    """
    global _announced
    if _announced:
        return
    _announced = True

    name = _as_dict(platform_config).get("target_platform") or "当前接口"
    log.info(
        f"已启用提示词前缀缓存（{name}）：稳定前缀 {len(system_stable or '')} 字符，"
        "缓存断点注入在其末尾。该前缀低于 provider 的最小可缓存长度时（Anthropic 按模型 "
        "512~4096 Tokens、OpenAI 1024 Tokens）不会生效，属正常现象。"
    )


def record_usage(cache_read: int = 0, cache_write: int = 0, uncached: int = 0) -> None:
    """累计一次请求的缓存口径，用于在日志里给出整体命中率。"""
    _usage_totals["read"] += max(0, int(cache_read or 0))
    _usage_totals["write"] += max(0, int(cache_write or 0))
    _usage_totals["uncached"] += max(0, int(uncached or 0))
    _usage_totals["requests"] += 1
    if cache_read or cache_write:
        _usage_totals["cacheable_requests"] += 1


def usage_snapshot() -> dict:
    return dict(_usage_totals)


def reset_usage() -> None:
    for key in _usage_totals:
        _usage_totals[key] = 0


def format_usage() -> str:
    """一行可读的缓存统计；命中率按「命中 / (命中 + 未命中)」算，不含写入。"""
    read = _usage_totals["read"]
    write = _usage_totals["write"]
    uncached = _usage_totals["uncached"]
    billed = read + uncached

    if not (read or write):
        return "缓存命中 0 Tokens（未命中缓存：前缀低于该 provider 的最小可缓存长度）"

    rate = (read / billed * 100) if billed else 0.0
    return f"缓存命中 {read} Tokens / 写入 {write} Tokens / 未命中 {uncached} Tokens（命中率 {rate:.1f}%）"
