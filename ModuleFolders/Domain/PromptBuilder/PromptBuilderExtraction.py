from functools import lru_cache

from ModuleFolders.Config.FilePathConfig import prompt_path
from ModuleFolders.Domain.PromptBuilder.PromptBuilderEnum import PromptBuilderEnum


class PromptBuilderExtraction:
    BASIC = "basic"
    JUDGMENT = "judgment"

    # 与翻译提示词一致：目标语言为中文时使用中文提示，其余目标语言使用英文提示。
    CHINESE_TARGET_LANGUAGES = ("chinese_simplified", "chinese_traditional")

    STAGES = {
        BASIC: {
            "selection_key": "extract_prompt_selection",
            "user_data_key": "extract_user_prompt_data",
            "preset_id": PromptBuilderEnum.EXTRACT_COMMON,
            "file_names": {
                "zh": "basic_system_zh.txt",
                "en": "basic_system_en.txt",
            },
        },
        JUDGMENT: {
            "selection_key": "extract_judgment_prompt_selection",
            "user_data_key": "extract_judgment_user_prompt_data",
            "preset_id": PromptBuilderEnum.EXTRACT_JUDGMENT,
            "file_names": {
                "zh": "judgment_system_zh.txt",
                "en": "judgment_system_en.txt",
            },
        },
    }

    @staticmethod
    def _config_value(config, key: str, default=None):
        # 页面使用配置字典，任务使用启动时载入的 TaskConfig 快照。
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def is_chinese_target(target_language) -> bool:
        """目标语言是否为中文（决定提取提示词语言，与翻译提示词的 zh/en 选择保持一致）"""
        return str(target_language or "") in PromptBuilderExtraction.CHINESE_TARGET_LANGUAGES

    @classmethod
    def resolve_prompt_language(cls, config) -> str:
        """根据目标语言选择提示词语言：中文目标 -> zh；其他目标 -> en；未配置目标时回退 zh（兼容旧行为）"""
        target_language = cls._config_value(config, "target_language", "")
        if not target_language:
            return "zh"
        return "zh" if cls.is_chinese_target(target_language) else "en"

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_system_file(stage: str, language: str) -> str:
        file_name = PromptBuilderExtraction.STAGES[stage]["file_names"][language]
        return prompt_path("Extract", file_name).read_text(encoding="utf-8").strip()

    @classmethod
    def get_system_default(cls, config, stage: str) -> str:
        """获取内置系统提示词，语言由配置的目标语言决定（与翻译提示词行为一致）"""
        language = cls.resolve_prompt_language(config)
        return cls._read_system_file(stage, language)

    @classmethod
    def get_user_prompts(cls, config, stage: str) -> list[dict]:
        prompts = cls._config_value(config, cls.STAGES[stage]["user_data_key"], [])
        if not isinstance(prompts, list):
            return []
        return [
            dict(prompt)
            for prompt in prompts
            if isinstance(prompt, dict)
            and prompt.get("type") == "user"
            and all(
                isinstance(prompt.get(key), str) and prompt[key].strip()
                for key in ("id", "name", "content")
            )
        ]

    @classmethod
    def get_selected_user_prompt(cls, config, stage: str) -> dict | None:
        selection = cls._config_value(config, cls.STAGES[stage]["selection_key"], {})
        if not isinstance(selection, dict):
            return None
        content = selection.get("prompt_content")
        if not isinstance(content, str) or not content.strip():
            return None
        return next(
            (
                prompt for prompt in cls.get_user_prompts(config, stage)
                if prompt["id"] == selection.get("last_selected_id")
            ),
            None,
        )

    @classmethod
    def build_system(cls, config, stage: str) -> str:
        # 已删除或无效的卡片不能仅凭残留的 prompt_content 继续生效。
        if cls.get_selected_user_prompt(config, stage) is not None:
            selection = cls._config_value(config, cls.STAGES[stage]["selection_key"])
            return selection["prompt_content"]
        # 选中的若是内置变体（如决策增强版），也必须生效，否则卡片只是个摆设。
        builtin = cls.get_selected_system_prompt(config, stage)
        if builtin is not None:
            return builtin["content"]
        return cls.get_system_default(config, stage)
    # ========================================================================
    # 内置提示词目录：通用版 + 面向决策层（JEV）的变体
    # ========================================================================
    # 通用版仍由 STAGES[stage] 的 file_names 提供；这里只登记新增的决策增强版文件，
    # 不动原有结构，避免与其它改动互相纠缠。
    COMMON_PRESET_KEY = "common"
    DECISION_PRESET_KEY = "decision"

    PRESET_NAMES = {
        COMMON_PRESET_KEY: "通用",
        DECISION_PRESET_KEY: "决策增强（JEV）",
    }

    PRESET_DESCRIPTIONS = {
        COMMON_PRESET_KEY: "原有提示词：只做提取与合并裁决。",
        DECISION_PRESET_KEY: "配合决策层：拆开派生名、保留独立短形式、拒绝不安全键。",
    }

    DECISION_FILE_NAMES = {
        BASIC: {
            "zh": "basic_system_decision_zh.txt",
            "en": "basic_system_decision_en.txt",
        },
        JUDGMENT: {
            "zh": "judgment_system_decision_zh.txt",
            "en": "judgment_system_decision_en.txt",
        },
    }

    DECISION_PRESET_IDS = {
        BASIC: PromptBuilderEnum.EXTRACT_COMMON_DECISION,
        JUDGMENT: PromptBuilderEnum.EXTRACT_JUDGMENT_DECISION,
    }

    @classmethod
    def _read_decision_file(cls, stage: str, language: str) -> str:
        file_name = cls.DECISION_FILE_NAMES[stage][language]
        return prompt_path("Extract", file_name).read_text(encoding="utf-8").strip()

    @classmethod
    def get_system_presets(cls, config, stage: str) -> list[dict]:
        """该阶段全部内置提示词，按卡片顺序返回，每项都是可直接渲染的提示词字典。

        语言与通用版一致，由目标语言决定；两个变体必须成对出现，缺一个就说明资源文件丢了。
        """
        language = cls.resolve_prompt_language(config)
        presets = [{
            "id": cls.STAGES[stage]["preset_id"],
            "key": cls.COMMON_PRESET_KEY,
            "name": cls.PRESET_NAMES[cls.COMMON_PRESET_KEY],
            "description": cls.PRESET_DESCRIPTIONS[cls.COMMON_PRESET_KEY],
            "content": cls.get_system_default(config, stage),
            "type": "system",
        }]
        if stage in cls.DECISION_FILE_NAMES:
            presets.append({
                "id": cls.DECISION_PRESET_IDS[stage],
                "key": cls.DECISION_PRESET_KEY,
                "name": cls.PRESET_NAMES[cls.DECISION_PRESET_KEY],
                "description": cls.PRESET_DESCRIPTIONS[cls.DECISION_PRESET_KEY],
                "content": cls._read_decision_file(stage, language),
                "type": "system",
            })
        return presets

    @classmethod
    def get_selected_system_prompt(cls, config, stage: str) -> dict | None:
        """当前选中的内置变体；选的是用户自建提示词或未选择时返回 None。"""
        selection = cls._config_value(config, cls.STAGES[stage]["selection_key"], {})
        if not isinstance(selection, dict):
            return None
        selected_id = selection.get("last_selected_id")
        return next(
            (preset for preset in cls.get_system_presets(config, stage) if preset["id"] == selected_id),
            None,
        )
