from ModuleFolders.Base.Base import Base
from ModuleFolders.Log.Log import LogMixin
from ModuleFolders.Infrastructure.LLMRequester.LLMClientFactory import LLMClientFactory
from ModuleFolders.Infrastructure.LLMRequester.ThinkingProfiles import build_thinking_params

import json
from openai.types.chat import ChatCompletion


# 接口请求器
class OpenaiRequester(LogMixin, Base):
    def __init__(self) -> None:
        pass

    # 思考参数装配已迁到 ThinkingProfiles 的 profile 注册表（Qt-free，可被单测覆盖）。
    # 这里保留同名入口，调用方与表征基线都不受影响；原先散落的 is_openai / is_deepseek
    # / is_xai / is_zhipu / is_dashscope / is_openrouter 字符串嗅探已收敛到
    # resolve_profile_name 一处，且只在存量配置没有声明 profile 时兜底。
    def apply_platform_thinking_params(self, base_params: dict, platform_config: dict) -> dict:
        return build_thinking_params(base_params, platform_config)

    # 手动解析SSE流式响应，合并为完整的ChatCompletion结果
    def _parse_sse_response(self, raw_text: str) -> tuple[str, str, int, int]:
        """
        解析SSE格式的流式响应，将多个chunk合并为完整结果。
        返回: (response_think, response_content, prompt_tokens, completion_tokens)
        """
        response_content = ""
        response_think = ""
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = None

        for line in raw_text.splitlines():
            # 跳过空行和非data行（如 event:、id:、retry:）
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()

            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # 提取 delta 中的内容
            choices = chunk.get("choices")
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    response_content += delta["content"]
                # OpenRouter 流式增量放在 reasoning，DeepSeek 用 reasoning_content
                delta_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if delta_reasoning:
                    response_think += delta_reasoning
                # 记录最后一个带 finish_reason 的 chunk（截断检测用，与 _extract_from_completion 对称）
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

            # 提取 usage 信息（通常在最后一个 chunk）
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        # 流式回复被max_tokens截断：抛错让上层按失败重试，不能把半截译文当成功返回
        if finish_reason == "length":
            raise RuntimeError(
                "Response truncated by max_tokens (finish_reason=length, SSE)；"
                "可在该平台设置里调大「最大生成长度」，或改用上限更高的模型"
            )

        return response_think, response_content, prompt_tokens, completion_tokens

    # 兼容各家推理字段：OpenRouter 放在 reasoning，DeepSeek/部分中转放在 reasoning_content
    @staticmethod
    def _extract_reasoning_text(message) -> str:
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value:
                return value

        # OpenRouter 的结构化推理明细（reasoning.text / reasoning.summary）
        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            parts = []
            for item in details:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("summary")
                else:
                    text = getattr(item, "text", None) or getattr(item, "summary", None)
                if isinstance(text, str) and text:
                    parts.append(text)
            if parts:
                return "".join(parts)

        return ""

    # 从响应中提取内容和token消耗
    def _extract_from_completion(self, response: ChatCompletion) -> tuple[str, str, int, int]:
        """
        从ChatCompletion对象中提取内容。
        返回: (response_think, response_content, prompt_tokens, completion_tokens)
        """
        message = response.choices[0].message

        # finish_reason=length意味着回复被max_tokens截断：截断的译文直接写入输出会造成
        # 半句/残留占位符，必须抛错让外层except按失败处理、上层重试，不能当成功返回
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length":
            raise RuntimeError(
                f"Response truncated by max_tokens (finish_reason=length), model {getattr(response, 'model', '?')}；"
                "可在该平台设置里调大「最大生成长度」，或改用上限更高的模型"
            )

        # 纯推理模型可能只返回 reasoning 而没有 content，这里统一成空串避免后续判断报错
        content = message.content or ""

        # 自适应提取推理过程
        if "</think>" in content:
            splited = content.split("</think>")
            response_think = splited[0].removeprefix("<think>").replace("\n\n", "\n")
            response_content = splited[-1]
        else:
            response_think = self._extract_reasoning_text(message)
            response_content = content

        # 获取token消耗
        try:
            prompt_tokens = int(response.usage.prompt_tokens)
        except Exception:
            prompt_tokens = 0
        try:
            completion_tokens = int(response.usage.completion_tokens)
        except Exception:
            completion_tokens = 0

        return response_think, response_content, prompt_tokens, completion_tokens

    # 发起请求
    def request_openai(self, messages, system_prompt, platform_config) -> tuple[bool, str, str, int, int]:
        try:
            # 获取具体配置
            model_name = platform_config.get("model_name")
            request_timeout = platform_config.get("request_timeout", 60)
            temperature = platform_config.get("temperature", 1.0)
            extra_body = platform_config.get("extra_body", {})

            # 插入系统消息
            if system_prompt:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt
                    })

            # 从工厂获取客户端
            client = LLMClientFactory().get_openai_client(platform_config)

            # 针对ds模型的特殊处理，因为该模型不支持模型预输入回复
            if 'deepseek' in model_name.lower():
                # 检查一下最后的消息是否用户消息，以免误删。(用户使用了推理模型卻不切换为推理模型提示词的情况)
                if messages and isinstance(messages[-1], dict) and messages[-1].get('role') != 'user':
                    messages = messages[:-1]  # 移除最后一个元素

            # 参数基础配置
            base_params = {
                "extra_body": extra_body,
                "model": model_name,
                "messages": messages,
                "timeout": request_timeout,
                "stream": False
            }

            # 按需添加参数
            if temperature != 1:
                base_params["temperature"] = temperature

            # 根据平台规则注入思考参数
            base_params = self.apply_platform_thinking_params(base_params, platform_config)

            # 使用with_raw_response获取原始响应，以便处理中转站强制返回流式响应的情况
            raw_response = client.chat.completions.with_raw_response.create(**base_params)

            # 尝试解析响应，部分中转站可能无视stream=False强制返回流式响应
            try:
                response = raw_response.parse()
            except Exception as parse_error:
                # parse报错（如text/json头但内容是SSE导致JSON解析失败），降级为SSE处理
                self.debug(f"响应解析失败: {parse_error}，尝试作为SSE处理")
                response = None

            # 根据响应类型选择处理方式
            if isinstance(response, ChatCompletion):
                # 标准非流式响应处理
                response_think, response_content, prompt_tokens, completion_tokens = self._extract_from_completion(
                    response)
            else:
                # 非标准响应，尝试作为SSE流式响应解析
                self.debug(f"收到非标准响应，尝试作为SSE处理")

                raw_text = raw_response.text
                response_think, response_content, prompt_tokens, completion_tokens = self._parse_sse_response(raw_text)

                # 自适应提取推理过程（针对某些模型将推理内容嵌入content的情况）
                if response_content and "</think>" in response_content:
                    splited = response_content.split("</think>")
                    response_think = splited[0].removeprefix("<think>").replace("\n\n", "\n")
                    response_content = splited[-1]

                # SSE解析后仍无内容，视为失败
                if not response_content:
                    raise ValueError(f"无法解析响应内容，原始响应: {raw_text[:500]}")

        except Exception as e:
            if Base.work_status == Base.STATUS.STOPING:
                return True, None, None, None, None
            self.error(f"请求任务错误 ... {e}", e)
            return True, None, None, None, None

        return False, response_think, response_content, prompt_tokens, completion_tokens
