import os
import json

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout

from qfluentwidgets import PlainTextEdit
from qfluentwidgets import MessageBoxBase
from qfluentwidgets import SingleDirectionScrollArea
from qfluentwidgets import SmoothMode

from ModuleFolders.Base.Base import Base
from ModuleFolders.Config.Config import ConfigMixin
from ModuleFolders.Config.FilePathConfig import platform_preset_path
from ModuleFolders.Log.Log import LogMixin
from UserInterface.Widget.SliderCard import SliderCard
from UserInterface.Widget.GroupCard import GroupCard
from UserInterface.Widget.SwitchButtonCard import SwitchButtonCard
from UserInterface.Widget.ComboBoxCard import ComboBoxCard
from UserInterface.Widget.SpinCard import SpinCard
from UserInterface.Widget.Toast import ToastMixin
from UserInterface.Widget.LineEditCard import LineEditCard
from UserInterface.Widget.PlainTextEditCard import PlainTextEditCard
from ModuleFolders.Infrastructure.LLMRequester.OptionSchema import (
    OptionSchemaError,
    decode_option_value,
    encode_option_value,
    option_widget_plan,
    resolve_options,
    resolve_preset_key,
    validate_descriptor,
    validate_schema,
)

class ArgsEditPage(MessageBoxBase, ConfigMixin, LogMixin, ToastMixin, Base):

    @staticmethod
    def is_volcengine_platform(key: str, platform: dict) -> bool:
        platform = platform if isinstance(platform, dict) else {}
        key = str(key or "").lower()
        tag = str(platform.get("tag") or "").lower()
        api_url = str(platform.get("api_url") or "").lower()
        return (
            key.startswith("volcengine")
            or tag.startswith("volcengine")
            or "volces.com" in api_url
            or "volcengine" in api_url
        )

    def __init__(self, window, key):
        super().__init__(window)

        # 初始化
        self.key = key

        # 设置框体
        self.widget.setFixedSize(960, 720)
        self.yesButton.setText(self.tra("关闭"))
        self.cancelButton.hide()

        # 载入配置文件
        config = self.load_config()
        preset = self.load_file(platform_preset_path())
        platform = config.get("platforms").get(self.key)
        settings = platform.get("key_in_settings") or []
        is_volcengine = self.is_volcengine_platform(self.key, platform)

        # 设置主布局
        self.viewLayout.setContentsMargins(0, 0, 0, 0)

        # 设置滚动器
        self.scroller = SingleDirectionScrollArea(self, orient = Qt.Vertical)
        # 弹窗内设置项较多，关闭平滑滚动可减少滚动动画带来的连续重绘。
        self.scroller.setSmoothMode(SmoothMode.NO_SMOOTH)
        self.scroller.setWidgetResizable(True)
        self.scroller.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.viewLayout.addWidget(self.scroller)

        # 设置滚动控件
        self.vbox_parent = QWidget(self)
        self.vbox_parent.setObjectName("argsEditScrollWidget")
        self.vbox_parent.setStyleSheet("#argsEditScrollWidget { background: transparent; }")
        self.vbox = QVBoxLayout(self.vbox_parent)
        self.vbox.setSpacing(8)
        self.vbox.setContentsMargins(24, 24, 24, 24) # 左、上、右、下
        self.scroller.setWidget(self.vbox_parent)


        # extra_body
        if "extra_body" in settings:
            self.add_widget_extra_body(self.vbox, config)

        # rpm_limit
        if "rpm_limit" in settings:
            self.add_widget_rpm(self.vbox, config)

        # temperature
        if "temperature" in settings:
            self.add_widget_temperature(self.vbox, config, preset)

        # think_switch。火山方舟旧实例的 key_in_settings 可能尚未包含该字段，
        # 因此按平台族补充显示，但仍保存到通用 think_switch。
        if "think_switch" in settings or is_volcengine:
            self.add_widget_think_switch(self.vbox, config)

        # 获取接口格式以进行条件渲染
        api_format = config.get("platforms").get(self.key).get("api_format")

        # think_depth - 仅在格式为 OpenAI 或 Anthropic 时显示
        if (
            ("think_depth" in settings or is_volcengine)
            and api_format in ["OpenAI", "Anthropic"]
        ):
            self.add_widget_think_depth(self.vbox, config)

        # Google 格式的思考参数配置 - 根据模型版本互斥显示
        if api_format == "Google":
            from ModuleFolders.Infrastructure.LLMRequester.ModelConfigHelper import ModelConfigHelper

            model_name = config.get("platforms").get(self.key).get("model", "")

            if ModelConfigHelper.is_gemini_3_or_newer(model_name):
                # Gemini 3.x 使用 thinking_level
                self.add_widget_thinking_level(self.vbox, config)
            elif "thinking_budget" in settings:
                # Gemini 2.5.x 及更早版本使用 thinking_budget
                self.add_widget_thinking_budget(self.vbox, config, preset)
        # tls_switch
        api_format = config.get("platforms").get(self.key).get("api_format")
        if "tls_switch" in settings or api_format == "OpenAI":
            self.add_widget_tls_switch(self.vbox, config)

        # 声明式选项（preset 里声明了 options 的平台）
        self.add_declared_options(self.vbox, config)

        # 填充
        self.vbox.addStretch(1)

    # 从文件加载
    def load_file(self, path: str) -> dict:
        result = {}

        if os.path.exists(path):
            with open(path, "r", encoding = "utf-8") as reader:
                result = json.load(reader)
        else:
            self.error(f"未找到 {path} 文件 ...")

        return result


    # TLS指纹模拟开关
    def add_widget_tls_switch(self, parent, config):
        def init(widget):
            widget.set_checked(config.get("platforms").get(self.key).get("tls_switch", False))

        def checked_changed(widget, checked: bool):
            config = self.load_config()
            config["platforms"][self.key]["tls_switch"] = checked
            self.save_config(config)

        parent.addWidget(
            SwitchButtonCard(
                self.tra("TLS 指纹模拟"),
                self.tra("开启后使用浏览器 TLS 指纹请求 OpenAI 兼容接口，并自动应用系统代理"),
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 思考开关
    def add_widget_think_switch(self, parent, config):
        def init(widget):
            widget.set_checked(config.get("platforms").get(self.key).get("think_switch", True))

        def checked_changed(widget, checked: bool):
            config = self.load_config()
            config["platforms"][self.key]["think_switch"] = checked
            self.save_config(config)

        parent.addWidget(
            SwitchButtonCard(
                self.tra("思考模式"),
                self.tra("开启后按当前平台规则附加模型思考参数，不支持的模型会保持默认请求"),
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 思考深度
    def add_widget_think_depth(self, parent, config):
        def init(widget):
            platform = config.get("platforms").get(self.key)

            platform_tag = str(platform.get("tag") or self.key or "").lower()
            api_url = str(platform.get("api_url") or "").lower()
            if platform_tag.startswith("xai") or "api.x.ai" in api_url:
                items = ["low", "medium", "high", "xhigh"]
            else:
                items = ["low", "medium", "high", "xhigh", "max"]
            widget.set_items(items)
            current = platform.get("think_depth", "medium")
            if current not in items:
                current = "medium"
            widget.set_current_index(widget.find_text(current))

        def current_text_changed(widget, text: str):
            config = self.load_config()
            config["platforms"][self.key]["think_depth"] = text.strip()
            self.save_config(config)

        parent.addWidget(
            ComboBoxCard(
                self.tra("思考强度"),
                self.tra("调整模型的思考强度等级"),
                [],
                init = init,
                current_text_changed = current_text_changed,
            )
        )

    # 思维预算
    def add_widget_thinking_budget(self, parent, config, preset):
        def init(widget):
            widget.set_range(-1, 32768)
            value = config.get("platforms").get(self.key).get("thinking_budget", -1)
            widget.set_text(str(value))
            widget.set_value(value)

        def value_changed(widget, value):
            widget.set_text(str(value))
            config = self.load_config()
            config["platforms"][self.key]["thinking_budget"] = value
            self.save_config(config)

        if self.key in preset.get("platforms"):
            default_value = preset.get("platforms").get(self.key).get("thinking_budget")
        else:
            default_value = -1

        info_cont = self.tra("Gemini 2.5 等模型的思考 Token 预算，-1 表示自动；默认值为") + f" {default_value}"
        parent.addWidget(
            SliderCard(
                self.tra("思考预算"),
                info_cont,
                init = init,
                value_changed = value_changed,
            )
        )

    # 思考深度级别 (Gemini 3 专用)
    def add_widget_thinking_level(self, parent, config):
        from ModuleFolders.Infrastructure.LLMRequester.ModelConfigHelper import ModelConfigHelper

        def init(widget):
            platform = config.get("platforms").get(self.key)
            model_name = platform.get("model", "")

            # 根据模型类型设置可用选项
            items = ModelConfigHelper.get_thinking_level_options(model_name)
            widget.set_items(items)

            current = platform.get("thinking_level", "high")
            idx = widget.find_text(current)
            widget.set_current_index(max(0, idx if idx >= 0 else len(items) - 1))

        def current_text_changed(widget, text: str):
            config = self.load_config()
            config["platforms"][self.key]["thinking_level"] = text.strip()
            self.save_config(config)

        parent.addWidget(
            ComboBoxCard(
                self.tra("思考级别"),
                self.tra("Gemini 3 专用，可用级别会随当前模型自动调整"),
                [],
                init=init,
                current_text_changed=current_text_changed,
            )
        )

    # 自定义Body
    def add_widget_extra_body(self, parent, config):

        def text_changed(widget):
            # 不再做 text.replace("'", '"')：那会把合法 JSON 里的撇号改坏，
            # 例如 {"a": "it's ok"} 会直接解析失败。
            try:
                extra_body_str = widget.toPlainText().strip()
                if not extra_body_str:
                    extra_body_dict = {}
                else:
                    extra_body_dict = json.loads(extra_body_str)
                    if not isinstance(extra_body_dict, dict):
                        raise ValueError("extra_body 顶层必须是 JSON 对象")
            except Exception as e:
                # --windowed 下 print 不可见，必须给可见反馈，并且**不写盘**：
                # 逐键保存意味着半成品输入会覆盖已经存好的值。
                if not getattr(widget, "_ainiee_extra_body_invalid", False):
                    self.warning_toast(
                        self.tra("自定义请求体"),
                        self.tra("JSON 格式不正确，已保留上一个有效值"),
                    )
                    widget._ainiee_extra_body_invalid = True
                self.error(f"接口保存 extra_body 参数失败: {e}")
                return

            widget._ainiee_extra_body_invalid = False
            config = self.load_config()
            config["platforms"][self.key]["extra_body"] = extra_body_dict
            self.save_config(config)

        def init(widget):
            plain_text_edit = PlainTextEdit(self)

            extra_body = config.get("platforms").get(self.key).get("extra_body")
            
            # 只有当 extra_body 是非空字典时才显示内容
            if isinstance(extra_body, dict) and extra_body:
                plain_text_edit.setPlainText(json.dumps(extra_body, ensure_ascii=False, indent=2))
            else:
                plain_text_edit.setPlainText("")

            info_cont = self.tra("请输入extra_body额外请求参数 JSON")
            plain_text_edit.setPlaceholderText(info_cont)
            plain_text_edit.textChanged.connect(lambda: text_changed(plain_text_edit))
            widget.addWidget(plain_text_edit)

        parent.addWidget(
            GroupCard(
                self.tra("自定义请求体"),
                self.tra("填写会合并到请求体的额外 JSON 参数，适合高级接口选项"),
                init = init,
            )
        )

    # 每分钟请求数
    def add_widget_rpm(self, parent, config):
        def init(widget):
            widget.set_range(0, 9999999)
            widget.set_value(config.get("platforms").get(self.key).get("rpm_limit", 4096))

        def value_changed(widget, value: str):
            config = self.load_config()
            config["platforms"][self.key]["rpm_limit"] = value
            self.save_config(config)

        parent.addWidget(
            SpinCard(
                self.tra("每分钟请求数"),
                self.tra("限制该接口每分钟最多发送的请求数量（RPM）"),
                init = init,
                value_changed = value_changed,
            )
        )

    # temperature
    def add_widget_temperature(self, parent, config, preset):
        def init(widget):
            widget.set_range(0, 200)
            # 手工改过的 config 可能缺 temperature 或存的不是数字，这里统一兜底，
            # 否则格式化字符串会直接抛异常，整个设置页打不开。
            raw = config.get("platforms").get(self.key).get("temperature")
            temperature = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 1.0
            widget.set_text(f"{temperature:.2f}")
            widget.set_value(int(temperature * 100))

        def value_changed(widget, value):
            widget.set_text(f"{(value / 100):.2f}")

            config = self.load_config()
            config["platforms"][self.key]["temperature"] = value / 100
            self.save_config(config)

        # 平台实例 key 带随机后缀，必须回到 preset 的 tag 上取默认值
        preset_platforms = preset.get("platforms", {})
        preset_key = resolve_preset_key(config.get("platforms").get(self.key) or {}, preset_platforms)
        default_platform = preset_platforms.get(preset_key) or preset_platforms.get("openai") or {}
        default_value = default_platform.get("temperature")

        info_cont = self.tra("控制回复随机性，值越高输出越发散；默认值为") + f" {default_value}"
        parent.addWidget(
            SliderCard(
                self.tra("温度"),
                info_cont,
                init = init,
                value_changed = value_changed,
            )
        )

    # ---- 声明式选项渲染 ----
    # preset 里声明了 options 的平台由通用渲染器出控件。选项键与 legacy 的
    # key_in_settings 不重叠（有测试守住），所以两套可以并存；等所有 provider
    # 迁移完成后，上面的手写分支和 is_volcengine_platform 就可以整段删除。
    def add_declared_options(self, parent, config):
        platform = config.get("platforms").get(self.key) or {}
        preset_platforms = (self.load_file(platform_preset_path()) or {}).get("platforms", {})
        options = resolve_options(platform, preset_platforms)
        if not options:
            return

        # schema 来自可热补丁的 Resource 文件：坏描述符只跳过并提示，
        # 绝不能让设置页构造失败（那等于用户打不开接口设置）。
        problems = validate_schema(options)
        if problems:
            self.warning_toast(self.tra("接口选项声明有误"), "; ".join(problems[:3]))
            self.error(f"接口选项声明有误: {problems}")

        for descriptor in options:
            try:
                validate_descriptor(descriptor)
            except OptionSchemaError as error:
                self.error(f"跳过无效选项声明: {error}")
                continue
            self._add_declared_option(parent, config, descriptor)

    def _save_option(self, descriptor, value):
        config = self.load_config()
        key = descriptor["key"]
        if value is None:
            # 不设置 = 交给服务端默认值，直接删键而不是写 null
            config["platforms"][self.key].pop(key, None)
        else:
            config["platforms"][self.key][key] = value
        self.save_config(config)

    def _add_declared_option(self, parent, config, descriptor):
        plan = option_widget_plan(descriptor, ConfigMixin.current_interface_language)
        current = config.get("platforms").get(self.key).get(descriptor["key"])
        title, desc = plan["label"], plan["desc"]
        unset_label = self.tra("（不设置）")

        if plan["widget"] == "switch":
            def init(widget):
                widget.set_checked(bool(current))

            def checked_changed(widget, checked: bool):
                self._save_option(descriptor, bool(checked))

            parent.addWidget(
                SwitchButtonCard(title, desc, init=init, checked_changed=checked_changed)
            )
            return

        if plan["widget"] == "combo":
            choices = plan["choices"]

            def init(widget):
                widget.set_items([choice if choice else unset_label for choice in choices])
                text = "" if current is None else str(current)
                widget.set_current_index(max(0, widget.find_text(text or unset_label)))

            def current_text_changed(widget, text: str):
                value = "" if text == unset_label else text
                self._save_option(descriptor, value or None)

            parent.addWidget(
                ComboBoxCard(
                    title, desc, [],
                    init=init,
                    current_text_changed=current_text_changed,
                )
            )
            return

        card_class = PlainTextEditCard if plan["widget"] in ("list", "json") else LineEditCard

        def init(widget):
            widget.set_text(decode_option_value(descriptor, current))
            if plan["widget"] == "list":
                widget.set_placeholder_text(self.tra("每行一个，留空表示不设置"))
            elif plan["widget"] == "json":
                widget.set_placeholder_text(self.tra("JSON，留空表示不设置"))

        def text_changed(widget, text: str):
            ok, value = encode_option_value(descriptor, text)
            if not ok:
                # 不写盘：半成品输入不能覆盖已经存好的值；只在进入非法状态时提示一次，
                # 避免逐键保存时每敲一个字符弹一次
                if not getattr(widget, "_ainiee_option_invalid", False):
                    self.warning_toast(title, self.tra("格式不正确，已忽略本次输入"))
                    widget._ainiee_option_invalid = True
                return
            widget._ainiee_option_invalid = False
            self._save_option(descriptor, value)

        parent.addWidget(card_class(title, desc, init=init, text_changed=text_changed))
