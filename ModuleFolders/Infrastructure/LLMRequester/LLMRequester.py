from ModuleFolders.Infrastructure.LLMRequester.SakuraRequester import SakuraRequester
from ModuleFolders.Infrastructure.LLMRequester.LocalLLMRequester import LocalLLMRequester
from ModuleFolders.Infrastructure.LLMRequester.GoogleRequester import GoogleRequester
from ModuleFolders.Infrastructure.LLMRequester.AnthropicRequester import AnthropicRequester
from ModuleFolders.Infrastructure.LLMRequester.AmazonbedrockRequester import AmazonbedrockRequester
from ModuleFolders.Infrastructure.LLMRequester.OpenaiRequester import OpenaiRequester

from ModuleFolders.Infrastructure.DecisionEngine.DecisionEngine import is_decision_format

def is_decision_platform(platform_config: dict) -> bool:
    """该接口是否为决策模型（System One / JEV）。

    决策模型只回答类型化问题、不生成文本，所以它不能当聊天/翻译接口用。判定放在这里而不是
    sent_request 里，是为了让「哪些平台会被交给聊天请求器」这件事只有一个定义——测试
    （test_option_mapping_baseline）直接复用本函数，两边不会各写一份而慢慢走偏。
    """
    if not isinstance(platform_config, dict):
        return False
    # The format check is the fallback for callers that hand over a stripped config with no
    # group (the interface test used to). It has to know every decision format: knowing only
    # "SystemOne" sent a "decisions" platform into the chat branch, which is a 404 for a
    # model that never generates text.
    return platform_config.get("group") == "decision" or is_decision_format(
        platform_config.get("api_format")
    )


# 接口请求器
class LLMRequester():
    def __init__(self) -> None:
        pass

    # 分发请求
    # system_prompt_stable：system_prompt 里「整份任务逐字节不变」的前缀，仅供需要显式
    # 缓存断点的 provider（Anthropic 系 / OpenRouter 透传）切分缓存断点使用；其余请求器
    # 忽略它，行为与此前一致。
    def sent_request(self, messages: list[dict], system_prompt: str, platform_config: dict, system_prompt_stable: str = "") -> tuple[bool, str, str, int, int]:
        # 决策模型（System One / JEV）只回答类型化问题，不生成文本，因此不能当翻译接口用。
        # 与其让它落进 OpenAI 分支后报一个看不懂的 400，不如在这里直接说清楚。
        if is_decision_platform(platform_config):
            name = platform_config.get("name") or platform_config.get("tag") or "该接口"
            return True, "", "「{}」是决策模型接口（只做判断，不生成文本），不能用作翻译/润色接口；请在接口管理中改选一个翻译接口。".format(name), 0, 0

        # 获取平台参数
        target_platform = platform_config.get("target_platform")
        api_format = platform_config.get("api_format")

        # 发起请求（sakura/LocalLLM 用 startswith，兼容新增平台 key 如 sakura_123456、LocalLLM_456789）
        if target_platform.startswith("sakura"):
            sakura_requester = SakuraRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = sakura_requester.request_sakura(
                messages,
                system_prompt,
                platform_config,
            )
        elif target_platform.startswith("LocalLLM"):
            local_llm_requester = LocalLLMRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = local_llm_requester.request_LocalLLM(
                messages,
                system_prompt,
                platform_config,
            )
        elif target_platform.startswith("google") or (
            (target_platform.startswith("custom_platform_") or target_platform.startswith("custom_"))
            and api_format == "Google"
        ):
            google_requester = GoogleRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = google_requester.request_google(
                messages,
                system_prompt,
                platform_config,
            )
        elif target_platform.startswith("anthropic") or (
            (target_platform.startswith("custom_platform_") or target_platform.startswith("custom_"))
            and api_format == "Anthropic"
        ):
            anthropic_requester = AnthropicRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = anthropic_requester.request_anthropic(
                messages,
                system_prompt,
                platform_config,
                system_prompt_stable,
            )
        elif target_platform.startswith("amazonbedrock"):
            amazonbedrock_requester = AmazonbedrockRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = amazonbedrock_requester.request_amazonbedrock(
                messages,
                system_prompt,
                platform_config,
            )
        else:
            openai_requester = OpenaiRequester()
            skip, response_think, response_content, prompt_tokens, completion_tokens = openai_requester.request_openai(
                messages,
                system_prompt,
                platform_config,
                system_prompt_stable,
            )

        return skip, response_think, response_content, prompt_tokens, completion_tokens
