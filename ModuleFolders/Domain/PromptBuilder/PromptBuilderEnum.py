from ModuleFolders.Base.Base import Base

class PromptBuilderEnum(Base):

    COMMON = 100
    COT = 200
    THINK = 300
    LOCAL = 400
    CUSTOM = 1000

    POLISH_COMMON = 10001

    FORMAT_COMMON = 20001

    EXTRACT_COMMON = 30001
    EXTRACT_JUDGMENT = 30002

    # 面向决策层（System One / JEV）的提取提示词变体，与上面两套通用提示词并列可选。
    EXTRACT_COMMON_DECISION = 30003
    EXTRACT_JUDGMENT_DECISION = 30004
