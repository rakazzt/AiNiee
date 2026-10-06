import uuid

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    DoubleSpinBox,
    FluentIcon,
    FluentWindow,
    HorizontalSeparator,
    IconWidget,
    LineEdit,
    MessageBoxBase,
    PrimaryPushButton,
    ScrollArea,
    SpinBox,
    StrongBodyLabel,
    SwitchButton,
    TextEdit,
)

from ModuleFolders.Base.Base import Base
from ModuleFolders.Config.Config import ConfigMixin
from ModuleFolders.Infrastructure.DecisionEngine import DecisionSettings
from UserInterface.PromptSettings.TranslationSettings.SystemPromptPage import PromptCard
from UserInterface.Widget.Toast import ToastMixin


class DecisionSettingsDialog(MessageBoxBase, ConfigMixin, ToastMixin, Base):
    """创建 / 编辑一套决策层设置。字段与 DecisionSettings 的键一一对应。"""

    def __init__(self, settings_data: dict = None, parent=None):
        super().__init__(parent)
        self.settings_data = settings_data or {}
        current = DecisionSettings.normalize(self.settings_data.get("settings"))

        self.view = QWidget(self)
        self.view_layout = QVBoxLayout(self.view)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        self.view_layout.setSpacing(14)
        self.view.setMinimumWidth(560)

        self.name_label = StrongBodyLabel(self.tra("名称"), self)
        self.name_edit = LineEdit(self)
        self.name_edit.setPlaceholderText(self.tra("例如：只报告不剔除"))
        self.name_edit.setText(str(self.settings_data.get("name", "")))
        self.view_layout.addWidget(self.name_label)
        self.view_layout.addWidget(self.name_edit)
        self.view_layout.addWidget(HorizontalSeparator(self))

        self.switches = {}
        for key in DecisionSettings.BOOL_KEYS:
            row = QHBoxLayout()
            row.addWidget(StrongBodyLabel(self.tra(DecisionSettings.FIELD_LABELS[key]), self), 1)
            switch = SwitchButton(self)
            switch.setChecked(bool(current[key]))
            self.switches[key] = switch
            row.addWidget(switch, 0, Qt.AlignRight)
            self.view_layout.addLayout(row)

        self.threshold_label = StrongBodyLabel(self.tra(DecisionSettings.FIELD_LABELS["threshold"]), self)
        self.threshold_spin = DoubleSpinBox(self)
        self.threshold_spin.setRange(*DecisionSettings.THRESHOLD_RANGE)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(float(current["threshold"]))
        self.view_layout.addWidget(self.threshold_label)
        self.view_layout.addWidget(self.threshold_spin)

        self.max_pairs_label = StrongBodyLabel(self.tra(DecisionSettings.FIELD_LABELS["max_pairs"]), self)
        self.max_pairs_spin = SpinBox(self)
        self.max_pairs_spin.setRange(*DecisionSettings.MAX_PAIRS_RANGE)
        self.max_pairs_spin.setValue(int(current["max_pairs"]))
        self.view_layout.addWidget(self.max_pairs_label)
        self.view_layout.addWidget(self.max_pairs_spin)

        note = CaptionLabel(
            self.tra("这些设置决定巡检会提出哪些问题、以及会剔除什么；均由程序执行，不改写任何译文。"),
            self,
        )
        note.setWordWrap(True)
        self.view_layout.addWidget(note)

        self.viewLayout.addWidget(self.view)
        self.yesButton.setText(self.tra("保存"))
        self.cancelButton.setText(self.tra("取消"))

    def accept(self) -> None:
        """Refuse to close on an empty name, so the typed settings are not silently lost."""
        if self.get_data() is None:
            self.warning_toast("", self.tra("请输入名称"))
            return
        super().accept()

    def get_data(self) -> dict | None:
        name = self.name_edit.text().strip()
        if not name:
            return None
        settings = {key: switch.isChecked() for key, switch in self.switches.items()}
        settings["threshold"] = self.threshold_spin.value()
        settings["max_pairs"] = int(self.max_pairs_spin.value())
        return {
            "id": self.settings_data.get("id") or uuid.uuid4().hex[:12],
            "name": name,
            "settings": DecisionSettings.normalize(settings),
        }


class DecisionSettingsPage(QFrame, ConfigMixin, Base):
    """决策层（System One / JEV）设置：一套内置默认 + 任意多套自定义。"""

    def __init__(self, text: str, window: FluentWindow) -> None:
        super().__init__(window)
        self.setObjectName(text.replace(" ", "-"))

        config = self.load_config()
        self.default_preset = self._default_preset()
        self.user_presets = self._user_presets(config)
        self.selected_preset = self.default_preset
        self.selected_card = None

        self.init_ui()
        self.update_cards(self._selected_id(config))

    # --- data ---------------------------------------------------------------

    def _default_preset(self) -> dict:
        settings = DecisionSettings.default_settings()
        return {
            "id": DecisionSettings.DEFAULT_ID,
            "name": self.tra("默认（推荐）"),
            "content": DecisionSettings.summarize(settings, self.tra),
            "type": "system",
            "settings": settings,
        }

    def _user_presets(self, config) -> list:
        presets = []
        for entry in DecisionSettings.get_user_settings(config):
            presets.append({
                "id": entry["id"],
                "name": entry["name"],
                "content": DecisionSettings.summarize(entry["settings"], self.tra),
                "type": "user",
                "settings": entry["settings"],
            })
        return presets

    def _selected_id(self, config) -> str:
        selection = config.get(DecisionSettings.SELECTION_KEY) if isinstance(config, dict) else None
        if isinstance(selection, dict):
            recorded = selection.get("last_selected_id")
            if isinstance(recorded, str) and any(
                preset["id"] == recorded for preset in [self.default_preset] + self.user_presets
            ):
                return recorded
        return self.default_preset["id"]

    # --- ui -----------------------------------------------------------------

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
        header_layout.addWidget(IconWidget(FluentIcon.PIN, self.top_display_card))
        header_layout.addWidget(StrongBodyLabel(self.tra("当前决策设置"), self.top_display_card))
        header_layout.addStretch(1)
        top_layout.addLayout(header_layout)
        top_layout.addWidget(HorizontalSeparator(self.top_display_card))

        description = self.tra(
            "决策层（System One / JEV）在抽取完成后判定条目关系、检查派生词译法一致性，"
            "并剔除不安全的词条。这里只决定它能做什么，判断本身由决策模型给出；"
            "若未在「接口管理 → 决策模型」中配置接口，巡检会整体跳过。"
        )
        self.description_label = CaptionLabel(description, self.top_display_card)
        self.description_label.setWordWrap(True)
        top_layout.addWidget(self.description_label)

        name_layout = QHBoxLayout()
        name_layout.addWidget(StrongBodyLabel(self.tra("名称："), self.top_display_card))
        self.selected_name_label = StrongBodyLabel("", self.top_display_card)
        self.selected_name_label.setWordWrap(True)
        name_layout.addWidget(self.selected_name_label, 1)
        top_layout.addLayout(name_layout)

        self.selected_detail_text = TextEdit(self.top_display_card)
        self.selected_detail_text.setReadOnly(True)
        self.selected_detail_text.setMinimumHeight(170)
        top_layout.addWidget(self.selected_detail_text)
        main_layout.addWidget(self.top_display_card, 1)

        self.bottom_card = CardWidget(self)
        bottom_layout = QVBoxLayout(self.bottom_card)
        bottom_layout.setContentsMargins(20, 15, 20, 15)
        bottom_layout.setSpacing(12)

        grid_header = QHBoxLayout()
        grid_header.addWidget(StrongBodyLabel(self.tra("决策设置"), self.bottom_card))
        grid_header.addStretch(1)
        self.add_button = PrimaryPushButton(FluentIcon.ADD, self.tra("创建新设置"), self.bottom_card)
        self.add_button.clicked.connect(self.open_add_dialog)
        grid_header.addWidget(self.add_button)
        bottom_layout.addLayout(grid_header)

        self.scroll_area = ScrollArea(self.bottom_card)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")
        self.container_widget = QWidget()
        self.container_widget.setStyleSheet("background-color: transparent;")
        self.card_grid = QGridLayout(self.container_widget)
        self.card_grid.setSpacing(15)
        self.card_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.container_widget)
        bottom_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.bottom_card, 3)

    def update_cards(self, selected_id: str) -> None:
        self.selected_card = None
        while self.card_grid.count():
            item = self.card_grid.takeAt(0)
            item.widget().deleteLater()

        self.all_presets = [self.default_preset] + self.user_presets
        for index, preset in enumerate(self.all_presets):
            card = PromptCard(preset, self.container_widget)
            card.prompt_selected.connect(self.display_details)
            if preset["type"] == "user":
                card.edit_requested.connect(self.open_add_dialog)
                card.delete_requested.connect(self.delete_preset)
            self.card_grid.addWidget(card, index // 3, index % 3)

        selected = next(
            (preset for preset in self.all_presets if preset["id"] == selected_id),
            self.default_preset,
        )
        self.display_details(selected)

    def find_card(self, preset_id):
        for index in range(self.card_grid.count()):
            widget = self.card_grid.itemAt(index).widget()
            if widget is not None and widget.prompt_data["id"] == preset_id:
                return widget
        return None

    def display_details(self, preset: dict) -> None:
        card = self.find_card(preset["id"])
        if card is None:
            return
        if self.selected_card is not None and self.selected_card is not card:
            self.selected_card.set_default_style()
        card.set_selected_style()
        self.selected_card = card
        self.selected_preset = preset

        self.selected_name_label.setText(preset["name"])
        settings = DecisionSettings.normalize(preset.get("settings"))
        lines = []
        for key in DecisionSettings.BOOL_KEYS:
            lines.append("{0}：{1}".format(
                self.tra(DecisionSettings.FIELD_LABELS[key]),
                self.tra("开") if settings[key] else self.tra("关"),
            ))
        lines.append("{0}：{1:.2f}".format(
            self.tra(DecisionSettings.FIELD_LABELS["threshold"]), settings["threshold"]))
        lines.append("{0}：{1}".format(
            self.tra(DecisionSettings.FIELD_LABELS["max_pairs"]), settings["max_pairs"]))
        self.selected_detail_text.setPlainText(chr(10).join(lines))

        # 选择与卡片列表一次写入；删除卡片时也会走到这里，所以两处必须同时更新。
        self.save_config({
            DecisionSettings.SELECTION_KEY: {
                "last_selected_id": preset["id"],
                "settings": settings,
            },
            DecisionSettings.USER_KEY: [
                {"id": entry["id"], "name": entry["name"], "settings": entry["settings"]}
                for entry in self.user_presets
            ],
        })

    def open_add_dialog(self, preset_to_edit=None) -> None:
        # 按钮的 clicked 会带一个 bool；卡片信号带的是 prompt_data 字典（不是 id）。
        if isinstance(preset_to_edit, bool):
            preset_to_edit = None
        if preset_to_edit is not None and preset_to_edit.get("type") != "user":
            return  # 内置默认不可编辑
        dialog = DecisionSettingsDialog(preset_to_edit, self.window())
        if not dialog.exec_():
            return
        data = dialog.get_data()
        if data is None:
            return
        self.user_presets = DecisionSettings.upsert_user_entry(self.user_presets, {
            "id": data["id"],
            "name": data["name"],
            "content": DecisionSettings.summarize(data["settings"], self.tra),
            "type": "user",
            "settings": data["settings"],
        })
        self.update_cards(data["id"])

    def delete_preset(self, preset_id: str) -> None:
        if not any(preset["id"] == preset_id for preset in self.user_presets):
            return
        # 删掉的恰好是当前选中项时退回内置默认；删别的条目不能顺手改掉当前选择。
        was_selected = self.selected_preset.get("id") == preset_id
        self.user_presets = [p for p in self.user_presets if p["id"] != preset_id]
        self.update_cards(self.default_preset["id"] if was_selected else self.selected_preset["id"])