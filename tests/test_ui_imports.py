import importlib
import unittest

try:  # 本机可能没装 PyQt5；CI 装了，所以这条在 CI 里才真正生效
    from PyQt5.QtWidgets import QWidget

    # 必须是真正的类：测试替身（把 PyQt5 换成假模块）会让所有基类塌缩成 object，
    # 那样导入界面类只会得到假的 MRO 报错，不如直接跳过。
    HAVE_QT = isinstance(QWidget, type)
except Exception:  # pragma: no cover - 取决于运行环境
    HAVE_QT = False


UI_MODULES = (
    "UserInterface.Platform.ArgsEditPage",
    "UserInterface.Platform.PlatformPage",
    "UserInterface.Platform.APIEditPage",
    "UserInterface.Platform.APIItemCard",
    "UserInterface.Platform.AddAPIDialog",
    "UserInterface.Platform.ModelBrowserDialog",
    "UserInterface.Platform.APIBindingDialog",
    "UserInterface.Widget.LineEditCard",
    "UserInterface.Widget.PlainTextEditCard",
)


@unittest.skipUnless(HAVE_QT, "PyQt5 is not installed; this check runs in CI")
class UiImportSmokeTests(unittest.TestCase):
    """界面模块至少必须能 import。

    接口设置的改造动的是 ArgsEditPage，而它只能靠编译与导入检查把关：
    既没有 GUI 测试环境，也没有显示器。语法错误、写错的导入、类基类顺序
    冲突都会在这里变红，而不是等用户点开设置页才发现。
    """

    def test_platform_ui_modules_import(self):
        for name in UI_MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
