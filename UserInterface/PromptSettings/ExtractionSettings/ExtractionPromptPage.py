from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    FluentIcon,
    FluentWindow,
    HorizontalSeparator,
    IconWidget,
    PrimaryPushButton,
    ScrollArea,
    StrongBodyLabel,
    TextEdit,
)

from ModuleFolders.Base.Base import Base
from ModuleFolders.Config.Config import ConfigMixin
from ModuleFolders.Domain.PromptBuilder.PromptBuilderExtraction import PromptBuilderExtraction
from UserInterface.PromptSettings.TranslationSettings.SystemPromptPage import AddEditPromptDialog, PromptCard


class ExtractionPromptPage(QFrame, ConfigMixin, Base):
    DESCRIPTIONS = {
        PromptBuilderExtraction.BASIC: "用于第一阶段：从原文提取角色、术语和禁翻项。",
        PromptBuilderExtraction.JUDGMENT: "用于第二阶段：合并候选结果，裁定角色和术语分类。",
    }

    def __init__(self, text: str, window: FluentWindow, stage: str) -> None:
        super().__init__(window)
        self.setObjectName(text.replace(" ", "-"))
        self.stage = stage
        self.settings = PromptBuilderExtraction.STAGES[stage]

        config = self.load_config()
        # 内置变体（通用 / 决策增强）与用户自建提示词并列展示；内置卡片不可编辑或删除。
        self.system_prompts = PromptBuilderExtraction.get_system_presets(config, stage)
        for preset in self.system_prompts:
            preset["name"] = self.tra(preset["name"])
            preset["description"] = self.tra(preset["description"])
        self.preset_prompt = self.system_prompts[0]
        self.user_prompts = PromptBuilderExtraction.get_user_prompts(config, stage)
        selected = (
            PromptBuilderExtraction.get_selected_user_prompt(config, stage)
            or PromptBuilderExtraction.get_selected_system_prompt(config, stage)
            or self.preset_prompt
        )
        self.selected_prompt_card = None
        self.selected_prompt_id = selected["id"]

        self.init_ui()
        self.update_prompt_cards(self.selected_prompt_id)

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 0)
        main_layout.setSpacing(15)

        self.top_display_card = CardWidget(self)
        top_layout = QVBoxLayout(self.top_display_card)
        top_layout.setContentsMargins(20, 15, 20, 15)
        top_layout.setSpacing(12)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        pin_icon = IconWidget(FluentIcon.PIN, self.top_display_card)
        pin_icon.setFixedSize(18, 18)
        header_layout.addWidget(pin_icon)
        header_layout.addWidget(StrongBodyLabel(self.tra("当前提示词"), self.top_display_card))
        header_layout.addStretch(1)
        top_layout.addLayout(header_layout)
        top_layout.addWidget(HorizontalSeparator(self.top_display_card))

        description = self.tra(self.DESCRIPTIONS[self.stage]) + "\n" + self.tra(
            "仅配置系统提示词，请保留预设中的 JSON 输出字段。示例、输入数据和目标语言要求由程序生成；修改将在下次提取任务生效。"
        )
        self.description_label = CaptionLabel(description, self.top_display_card)
        self.description_label.setWordWrap(True)
        top_layout.addWidget(self.description_label)

        name_layout = QHBoxLayout()
        name_layout.addWidget(StrongBodyLabel(self.tra("名称："), self.top_display_card))
        self.selected_prompt_name_label = StrongBodyLabel("", self.top_display_card)
        self.selected_prompt_name_label.setWordWrap(True)
        name_layout.addWidget(self.selected_prompt_name_label, 1)
        top_layout.addLayout(name_layout)

        self.selected_prompt_content_text = TextEdit(self.top_display_card)
        self.selected_prompt_content_text.setReadOnly(True)
        self.selected_prompt_content_text.setMinimumHeight(200)
        top_layout.addWidget(self.selected_prompt_content_text)
        main_layout.addWidget(self.top_display_card, 1)

        self.bottom_grid_card = CardWidget(self)
        bottom_layout = QVBoxLayout(self.bottom_grid_card)
        bottom_layout.setContentsMargins(20, 15, 20, 15)
        bottom_layout.setSpacing(12)

        grid_header = QHBoxLayout()
        grid_header.addWidget(StrongBodyLabel(self.tra("提示词广场"), self.bottom_grid_card))
        grid_header.addStretch(1)
        self.add_new_prompt_button = PrimaryPushButton(FluentIcon.ADD, self.tra("创建新提示词"), self.bottom_grid_card)
        self.add_new_prompt_button.clicked.connect(self.open_add_prompt_dialog)
        grid_header.addWidget(self.add_new_prompt_button)
        bottom_layout.addLayout(grid_header)

        self.scroll_area = ScrollArea(self.bottom_grid_card)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")
        self.card_container_widget = QWidget()
        self.card_container_widget.setStyleSheet("background-color: transparent;")
        self.card_grid_layout = QGridLayout(self.card_container_widget)
        self.card_grid_layout.setSpacing(15)
        self.card_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.card_container_widget)
        bottom_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.bottom_grid_card, 3)

    def update_prompt_cards(self, selected_id) -> None:
        # 清空旧引用，避免重建后高亮已被 Qt 删除的卡片。
        self.selected_prompt_card = None
        while self.card_grid_layout.count():
            item = self.card_grid_layout.takeAt(0)
            item.widget().deleteLater()

        self.all_prompts = self.system_prompts + self.user_prompts
        for index, prompt in enumerate(self.all_prompts):
            card = PromptCard(prompt, self.card_container_widget)
            card.prompt_selected.connect(self.display_prompt_details)
            if prompt["type"] == "user":
                card.edit_requested.connect(self.open_add_prompt_dialog)
                card.delete_requested.connect(self.delete_user_prompt)
            self.card_grid_layout.addWidget(card, index // 3, index % 3)

        selected = next((prompt for prompt in self.all_prompts if prompt["id"] == selected_id), self.preset_prompt)
        self.display_prompt_details(selected)

    def find_card_widget(self, prompt_id):
        for index in range(self.card_grid_layout.count()):
            card = self.card_grid_layout.itemAt(index).widget()
            if card.prompt_data["id"] == prompt_id:
                return card
        return None

    def display_prompt_details(self, prompt_data: dict) -> None:
        card = self.find_card_widget(prompt_data["id"])
        if card is None:
            return
        if self.selected_prompt_card is not None and self.selected_prompt_card is not card:
            self.selected_prompt_card.set_default_style()
        card.set_selected_style()
        self.selected_prompt_card = card
        self.selected_prompt_id = prompt_data["id"]
        # 内置变体带一句说明，让它显示出来，而不是当作死字段带着。
        description = prompt_data.get("description", "")
        self.selected_prompt_name_label.setText(
            "{0} — {1}".format(prompt_data["name"], description) if description else prompt_data["name"]
        )
        self.selected_prompt_content_text.setPlainText(prompt_data["content"])

        # 同一次写入保存卡片列表与选择，且只更新本阶段的两个配置项。
        self.save_config({
            self.settings["selection_key"]: {
                "last_selected_id": prompt_data["id"],
                "prompt_content": prompt_data["content"],
            },
            self.settings["user_data_key"]: self.user_prompts,
        })

    def open_add_prompt_dialog(self, prompt_to_edit=None) -> None:
        if isinstance(prompt_to_edit, bool):
            prompt_to_edit = None
        if prompt_to_edit is not None and prompt_to_edit.get("type") != "user":
            return
        dialog = AddEditPromptDialog(prompt_to_edit, self)
        if not dialog.exec_():
            return
        data = dialog.get_data()
        if not data:
            return

        for index, prompt in enumerate(self.user_prompts):
            if prompt["id"] == data["id"]:
                self.user_prompts[index] = data
                break
        else:
            self.user_prompts.append(data)
        self.update_prompt_cards(data["id"])

    def delete_user_prompt(self, prompt_id: str) -> None:
        if not any(prompt["id"] == prompt_id for prompt in self.user_prompts):
            return
        self.user_prompts = [prompt for prompt in self.user_prompts if prompt["id"] != prompt_id]
        self.update_prompt_cards(self.selected_prompt_id)
