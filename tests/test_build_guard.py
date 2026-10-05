import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Tools.build_guard import assert_no_user_config


class BuildGuardTests(unittest.TestCase):
    """打包前必须挡住 Resource/config.json，否则开发机密钥会进发布包。"""

    def test_clean_resource_directory_passes(self):
        with TemporaryDirectory() as tmp:
            resource = Path(tmp)
            (resource / "platforms").mkdir()
            (resource / "platforms" / "preset.json").write_text("{}", encoding="utf-8")
            assert_no_user_config(resource)  # 不应抛异常

    def test_empty_directory_passes(self):
        with TemporaryDirectory() as tmp:
            assert_no_user_config(Path(tmp))

    def test_user_config_aborts_the_build(self):
        with TemporaryDirectory() as tmp:
            resource = Path(tmp)
            (resource / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                assert_no_user_config(resource)
            # 报错必须指出具体路径，否则维护者不知道删哪个文件
            self.assertIn("config.json", str(caught.exception))

    def test_derived_config_files_also_abort(self):
        # Config.py 解析失败会产生 .corrupt，写盘中间态会产生 .tmp，两者同样含密钥
        for name in ("config.json.corrupt", "config.json.tmp"):
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                resource = Path(tmp)
                (resource / name).write_text("{}", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    assert_no_user_config(resource)

    def test_similarly_named_file_does_not_abort(self):
        with TemporaryDirectory() as tmp:
            resource = Path(tmp)
            (resource / "config.json.example").write_text("{}", encoding="utf-8")
            assert_no_user_config(resource)


if __name__ == "__main__":
    unittest.main()
