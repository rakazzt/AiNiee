import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Tools.build_guard import assert_no_user_config


class BuildGuardTests(unittest.TestCase):
    """打包前必须挡住 Resource/config.json，否则开发机密钥会进发布包。"""

    def _project_root(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "Resource").mkdir(parents=True, exist_ok=True)
        return root

    def test_clean_project_passes(self):
        with TemporaryDirectory() as tmp:
            root = self._project_root(tmp)
            (root / "Resource" / "platforms").mkdir()
            (root / "Resource" / "platforms" / "preset.json").write_text("{}", encoding="utf-8")
            assert_no_user_config(root)

    def test_project_without_resource_dir_passes(self):
        with TemporaryDirectory() as tmp:
            assert_no_user_config(Path(tmp))

    def test_user_config_aborts_the_build(self):
        with TemporaryDirectory() as tmp:
            root = self._project_root(tmp)
            (root / "Resource" / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                assert_no_user_config(root)
            # 报错必须指出具体路径，否则维护者不知道删哪个文件
            self.assertIn("config.json", str(caught.exception))

    def test_derived_config_files_also_abort(self):
        # Config.py 解析失败会产生 .corrupt，写盘中间态会产生 .tmp，两者同样含密钥
        for name in ("config.json.corrupt", "config.json.tmp"):
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                root = self._project_root(tmp)
                (root / "Resource" / name).write_text("{}", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    assert_no_user_config(root)

    def test_similarly_named_file_does_not_abort(self):
        with TemporaryDirectory() as tmp:
            root = self._project_root(tmp)
            (root / "Resource" / "config.json.example").write_text("{}", encoding="utf-8")
            assert_no_user_config(root)

    def test_accepts_a_string_root_like_pyinstall_py_passes(self):
        """回归：pyinstall.py 的 ROOT 是 str，pyinstall_macos.py 的是 Path。

        两者都必须直接可用，调用方不允许自己拼路径 —— 之前正是 `ROOT / "Resource"`
        在 Windows 打包脚本里抛 TypeError，只在 CI 才暴露。
        """
        with TemporaryDirectory() as tmp:
            root = self._project_root(tmp)
            (root / "Resource" / "config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit):
                assert_no_user_config(str(root))  # pyinstall.py 的 str 形态
            with self.assertRaises(SystemExit):
                assert_no_user_config(root)  # pyinstall_macos.py 的 Path 形态

    def test_string_root_without_config_passes(self):
        with TemporaryDirectory() as tmp:
            assert_no_user_config(str(self._project_root(tmp)))


if __name__ == "__main__":
    unittest.main()
