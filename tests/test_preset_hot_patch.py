import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ModuleFolders.Config.Config import ConfigMixin
from ModuleFolders.Infrastructure.LLMRequester.OptionSchema import resolve_options
from ModuleFolders.Infrastructure.LLMRequester.ProviderDocs import resolve_docs

PRESET_TEMPLATE = {
    "platforms": {
        "openrouter": {
            "tag": "openrouter",
            "options": [{"key": "provider_order", "type": "string-list",
                         "path": "provider.order", "label": "order"}],
            "docs": {"docs_url": "https://openrouter.ai/docs"},
        }
    }
}


class PresetHotPatchTests(unittest.TestCase):
    """Resource 热补丁是这套设计的核心卖点，必须真的成立。

    设计评审把它列为验收项：装好的构建只替换 Resource/platforms/preset.json，
    不重新打包就能更新选项与文档；同时缓存不能把补丁吃掉，读坏了也不能让
    请求或界面崩掉。
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.resource_dir = Path(self._tmp.name)
        (self.resource_dir / "platforms").mkdir(parents=True, exist_ok=True)
        self.preset_path = self.resource_dir / "platforms" / "preset.json"
        self._write_preset(PRESET_TEMPLATE)

        env = patch.dict(os.environ, {"AINIEE_RESOURCE_DIR": str(self.resource_dir)})
        env.start()
        self.addCleanup(env.stop)
        # 缓存是类级字典，用例之间必须隔离
        ConfigMixin._preset_cache.clear()
        self.addCleanup(ConfigMixin._preset_cache.clear)
        self.addCleanup(self._tmp.cleanup)

    def _write_preset(self, data, bump_time=True):
        self.preset_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if bump_time:
            # 保证 mtime 变化，避免同一时间片内写入被缓存判定为未变
            stat = self.preset_path.stat()
            os.utime(self.preset_path, ns=(stat.st_atime_ns + 1_000_000, stat.st_mtime_ns + 1_000_000))

    def test_preset_is_read_from_the_overridden_resource_dir(self):
        presets = ConfigMixin.load_platform_presets()
        self.assertIn("openrouter", presets["platforms"])

    def test_hot_patch_is_picked_up_without_restart(self):
        before = resolve_options({"tag": "openrouter_1"}, ConfigMixin.load_platform_presets()["platforms"])
        self.assertEqual([o["key"] for o in before], ["provider_order"])

        patched = json.loads(json.dumps(PRESET_TEMPLATE))
        patched["platforms"]["openrouter"]["options"].append(
            {"key": "provider_zdr", "type": "bool", "path": "provider.zdr", "label": "zdr"}
        )
        patched["platforms"]["openrouter"]["docs"]["docs_url"] = "https://openrouter.ai/docs/patched"
        self._write_preset(patched)

        after = ConfigMixin.load_platform_presets()
        keys = [o["key"] for o in resolve_options({"tag": "openrouter_1"}, after["platforms"])]
        self.assertEqual(keys, ["provider_order", "provider_zdr"], "cache swallowed the patch")
        self.assertEqual(
            resolve_docs({"tag": "openrouter_1"}, after["platforms"])["docs_url"],
            "https://openrouter.ai/docs/patched",
        )

    def test_docs_only_patch_is_picked_up(self):
        patched = json.loads(json.dumps(PRESET_TEMPLATE))
        patched["platforms"]["openrouter"]["docs"]["docs_url"] = "https://example.invalid/new"
        self._write_preset(patched)
        docs = resolve_docs({"tag": "openrouter_1"}, ConfigMixin.load_platform_presets()["platforms"])
        self.assertEqual(docs["docs_url"], "https://example.invalid/new")

    def test_broken_preset_degrades_instead_of_raising(self):
        """热补丁写坏是必然会发生的事：只能降级，不能把请求或界面带崩。"""
        self.preset_path.write_text("{ this is not json", encoding="utf-8")
        presets = ConfigMixin.load_platform_presets()
        self.assertEqual(presets, {"platforms": {}})
        self.assertEqual(resolve_options({"tag": "openrouter_1"}, presets["platforms"]), [])

    def test_preset_with_wrong_root_type_degrades(self):
        self.preset_path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(ConfigMixin.load_platform_presets(), {"platforms": {}})

    def test_missing_preset_file_degrades(self):
        self.preset_path.unlink()
        self.assertEqual(ConfigMixin.load_platform_presets(), {"platforms": {}})


if __name__ == "__main__":
    unittest.main()
