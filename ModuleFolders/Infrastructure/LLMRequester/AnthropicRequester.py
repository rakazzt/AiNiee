from ModuleFolders.Base.Base import Base
from ModuleFolders.Log.Log import LogMixin
from ModuleFolders.Infrastructure.LLMRequester.LLMClientFactory import LLMClientFactory
from ModuleFolders.Infrastructure.LLMRequester.ModelConfigHelper import ModelConfigHelper
from ModuleFolders.Infrastructure.LLMRequester import PromptCache


# 接口请求器
class AnthropicRequester(LogMixin, Base):
    def __init__(self) -> None:
        pass

    def request_anthropic(self, messages, system_prompt, platform_config, system_prompt_stable: str = "") -> tuple[bool, str, str, int, int]:
        try:
            model_name = platform_config.get("model_name")
            request_timeout = platform_config.get("request_timeout", 60)
            think_switch = platform_config.get("think_switch")
            think_depth = platform_config.get("think_depth") or "medium"

            max_tokens = ModelConfigHelper.get_claude_max_output_tokens(model_name)

            # Claude 5 不接受 assistant 末尾预填充。
            if (
                messages
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "assistant"
            ):
                messages = messages[:-1]

            # 显式缓存断点：Anthropic 不声明 cache_control 就完全不缓存，等于每个批次都按
            # 原价重发整份系统提示词。断点落在「整份任务不变」的稳定前缀末尾，其后按批次
            # 变化的内容照常按原价计费（它们本来也缓存不了）。
            system_param = system_prompt
            if PromptCache.supports_explicit_cache(platform_config):
                system_blocks = PromptCache.build_system_blocks(system_prompt_stable, system_prompt or "")
                if system_blocks:
                    system_param = system_blocks
                    PromptCache.announce_explicit_cache(system_prompt_stable, platform_config, self)

            # 参数基础配置
            base_params = {
                "model": model_name,
                "system": system_param,
                "messages": messages,
                "timeout": request_timeout,
                "max_tokens": max_tokens,
            }

            # Claude 5 统一使用 adaptive thinking + output_config.effort，且不发送采样温度。
            if think_switch:
                base_params["thinking"] = {
                    "type": "adaptive",
                    "display": "summarized",
                }
                base_params["output_config"] = {
                    "effort": think_depth
                    if think_depth in {"low", "medium", "high", "xhigh", "max"}
                    else "medium"
                }
            elif not ModelConfigHelper.is_claude_always_thinking_model(model_name):
                base_params["thinking"] = {"type": "disabled"}

            # 从工厂获取客户端
            client = LLMClientFactory().get_anthropic_client(platform_config)
            # 发送请求
            response = client.messages.create(**base_params)

            # 提取回复的文本内容和思考内容
            thinking_parts = []
            content_parts = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "thinking":
                        thinking_text = getattr(block, "thinking", "")
                        if thinking_text:
                            thinking_parts.append(thinking_text)
                    elif block.type == "text":
                        content_parts.append(block.text)

            response_think = "".join(thinking_parts)
            response_content = "".join(content_parts)

            # stop_reason=max_tokens意味着回复被截断，半截译文写进输出比请求失败更糟
            # （与AmazonbedrockRequester的stop_reason检查对称）
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise RuntimeError(
                    f"Response truncated (stop_reason=max_tokens), model {model_name}"
                )

        except Exception as e:
            if Base.work_status == Base.STATUS.STOPING:
                return True, None, None, None, None
            self.error_repeat(f"请求任务错误 ... {e}", e)
            return True, None, None, None, None

        # 获取指令消耗（Anthropic 使用 input_tokens）
        # 注意：input_tokens 只统计断点之后、既没读也没写的 token，不含缓存命中与写入。
        # 直接拿它当 prompt_tokens，开了缓存之后统计反而变小 —— 看着像省钱，其实是少算。
        try:
            cache_read, cache_write, prompt_tokens = PromptCache.anthropic_usage_totals(response.usage)
        except Exception:
            cache_read, cache_write, prompt_tokens = 0, 0, 0

        # 获取回复消耗（Anthropic 使用 output_tokens）
        try:
            completion_tokens = int(response.usage.output_tokens)
        except Exception:
            completion_tokens = 0

        PromptCache.record_usage(cache_read, cache_write, prompt_tokens - cache_read - cache_write)
        self.debug(
            f"提示词缓存: 本次命中 {cache_read} / 写入 {cache_write} Tokens；{PromptCache.format_usage()}"
        )

        return False, response_think, response_content, prompt_tokens, completion_tokens
