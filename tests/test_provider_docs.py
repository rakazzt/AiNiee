import json
import unittest
from pathlib import Path

from ModuleFolders.Infrastructure.LLMRequester.ProviderDocs import (
    doc_links,
    doc_notes,
    has_docs,
    resolve_docs,
    safe_external_url,
)

PRESET = {
    "openrouter": {
        "tag": "openrouter",
        "docs": {
            "api_key_url": "https://openrouter.ai/settings/keys",
            "docs_url": "https://openrouter.ai/docs",
            "models_url": "https://openrouter.ai/models",
            "notes": {"简中": "支持 300+ 模型。", "English": "300+ models."},
        },
    },
    "plain": {"tag": "plain"},
    "notes_only": {"tag": "notes_only", "docs": {"notes": "只有说明，没有链接"}},
}


class SafeExternalUrlTests(unittest.TestCase):
    """preset.json 是可热补丁的数据文件，链接是新的信任边界。"""

    def test_https_with_host_is_allowed(self):
        self.assertEqual(
            safe_external_url("https://openrouter.ai/docs"), "https://openrouter.ai/docs"
        )

    def test_non_https_schemes_are_rejected(self):
        for url in (
            "http://openrouter.ai/docs",
            "file:///C:/Windows/System32/calc.exe",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ftp://example.com/x",
            "custom-scheme://do-something",
        ):
            with self.subTest(url=url):
                self.assertEqual(safe_external_url(url), "")

    def test_missing_host_is_rejected(self):
        self.assertEqual(safe_external_url("https://"), "")
        self.assertEqual(safe_external_url("https:///nohost"), "")

    def test_non_string_and_blank_are_rejected(self):
        for value in (None, 123, {}, [], "   "):
            with self.subTest(value=value):
                self.assertEqual(safe_external_url(value), "")

    def test_surrounding_whitespace_is_trimmed(self):
        self.assertEqual(
            safe_external_url("  https://openrouter.ai/docs  "), "https://openrouter.ai/docs"
        )


class ResolveDocsTests(unittest.TestCase):
    def test_generated_instance_key_resolves_to_its_preset(self):
        self.assertEqual(
            resolve_docs({"tag": "openrouter_482913"}, PRESET)["docs_url"],
            "https://openrouter.ai/docs",
        )

    def test_platform_without_docs_returns_empty(self):
        self.assertEqual(resolve_docs({"tag": "plain"}, PRESET), {})

    def test_unknown_platform_returns_empty(self):
        self.assertEqual(resolve_docs({"tag": "ghost_1"}, PRESET), {})

    def test_malformed_docs_block_is_ignored(self):
        preset = {"x": {"docs": "not-a-dict"}}
        self.assertEqual(resolve_docs({"tag": "x"}, preset), {})


class DocLinksTests(unittest.TestCase):
    def test_links_carry_label_and_url_in_declared_order(self):
        links = doc_links(PRESET["openrouter"]["docs"], "English")
        self.assertEqual([f for f, _, _ in links], ["api_key_url", "docs_url", "models_url"])
        self.assertEqual(links[0][1], "Get an API key")

    def test_unsafe_link_is_dropped_from_the_list(self):
        docs = {"docs_url": "javascript:alert(1)", "api_key_url": "https://ok.example/key"}
        links = doc_links(docs)
        self.assertEqual([f for f, _, _ in links], ["api_key_url"])

    def test_notes_only_docs_yield_no_links(self):
        self.assertEqual(doc_links(PRESET["notes_only"]["docs"]), [])


class DocNotesTests(unittest.TestCase):
    def test_notes_are_localized(self):
        self.assertEqual(doc_notes(PRESET["openrouter"]["docs"], "English"), "300+ models.")
        self.assertEqual(doc_notes(PRESET["openrouter"]["docs"], "简中"), "支持 300+ 模型。")

    def test_missing_notes_are_empty(self):
        self.assertEqual(doc_notes({}), "")


class HasDocsTests(unittest.TestCase):
    def test_detects_links_or_notes(self):
        self.assertTrue(has_docs(PRESET["openrouter"]["docs"]))
        self.assertTrue(has_docs(PRESET["notes_only"]["docs"]))
        self.assertFalse(has_docs({}))
        self.assertFalse(has_docs({"docs_url": "javascript:alert(1)"}))


class RealPresetDocsTests(unittest.TestCase):
    """真实 preset.json 的文档块自检 —— 链接写错会让用户点到非预期页面。"""

    @classmethod
    def setUpClass(cls):
        preset_path = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"
        cls.preset = json.loads(preset_path.read_text(encoding="utf-8"))

    def test_every_declared_doc_link_is_https(self):
        problems = []
        for tag, platform in self.preset["platforms"].items():
            docs = platform.get("docs")
            if not isinstance(docs, dict):
                continue
            for field in ("api_key_url", "docs_url", "models_url"):
                raw = docs.get(field)
                if raw is None:
                    continue
                if not safe_external_url(raw):
                    problems.append(f"{tag}.{field} = {raw!r}")
        self.assertEqual(problems, [], "doc links must be https with a host")

    def test_every_online_platform_offers_a_key_page(self):
        missing = []
        for tag, platform in self.preset["platforms"].items():
            if platform.get("group") != "online":
                continue
            docs = platform.get("docs") or {}
            if not safe_external_url(docs.get("api_key_url")):
                missing.append(tag)
        self.assertEqual(missing, [], "online platforms should tell users where to get a key")

    def test_doc_text_is_localized(self):
        problems = []
        for tag, platform in self.preset["platforms"].items():
            docs = platform.get("docs")
            if not isinstance(docs, dict):
                continue
            for field in ("notes",):
                value = docs.get(field)
                if value is None:
                    continue
                if not isinstance(value, dict) or not value.get("简中") or not value.get("English"):
                    problems.append(f"{tag}.{field}")
        self.assertEqual(problems, [], "notes must carry 简中 and English text")


if __name__ == "__main__":
    unittest.main()
