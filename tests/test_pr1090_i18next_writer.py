"""Round-trip tests for I18nextWriter, the other writer that rewrites a file
derived from the user's source (PR #1090 changed it from rebuilding the JSON to
writing back into a copy of the source).

The point is that a translation run must not destroy the parts of the source JSON
that are not translatable strings.
"""
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ModuleFolders.Domain.FileOutputer.I18nextWriter import I18nextWriter
from ModuleFolders.Service.Cache.CacheFile import CacheFile
from ModuleFolders.Service.Cache.CacheItem import CacheItem, TranslationStatus

SOURCE = {
    "greet": "hello",
    "count": 42,
    "ratio": 1.5,
    "flag": True,
    "nothing": None,
    "list": [1, 2, 3],
    "empty": {},
    "nested": {"deep": "text"},
}


class I18nextWriterRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "en.json"
        self.output = self.tmp / "out.json"
        self.writer = I18nextWriter.__new__(I18nextWriter)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_source(self, payload=None, bom=False):
        text = json.dumps(SOURCE if payload is None else payload, ensure_ascii=False)
        self.source.write_text(("\ufeff" if bom else "") + text, encoding="utf-8")

    def _item(self, path, translated, status=TranslationStatus.TRANSLATED):
        item = CacheItem(source_text="src", translated_text=translated,
                         extra={"i18next_path": path})
        item.translation_status = status
        return item

    def _run(self, items, bom=False):
        self._write_source(bom=bom)
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.writer.on_write_translated(
            self.output, CacheFile(items=items), None, source_file_path=self.source
        )
        after = hashlib.sha256(self.source.read_bytes()).hexdigest()
        return before, after, json.loads(self.output.read_text(encoding="utf-8-sig"))

    def test_translated_strings_are_written(self):
        _, _, data = self._run([self._item(["greet"], "你好"),
                                self._item(["nested", "deep"], "深层")])
        self.assertEqual(data["greet"], "你好")
        self.assertEqual(data["nested"]["deep"], "深层")

    def test_non_string_leaves_survive(self):
        """These are the values the old rebuild-from-scratch writer deleted."""
        _, _, data = self._run([self._item(["greet"], "你好")])
        self.assertEqual(data["count"], 42)
        self.assertEqual(data["ratio"], 1.5)
        self.assertIs(data["flag"], True)
        self.assertIsNone(data["nothing"])
        self.assertEqual(data["list"], [1, 2, 3])
        self.assertEqual(data["empty"], {})

    def test_source_file_is_never_modified(self):
        before, after, _ = self._run([self._item(["greet"], "你好")])
        self.assertEqual(before, after)

    def test_untranslated_key_keeps_the_source_value(self):
        _, _, data = self._run([self._item(["greet"], "", TranslationStatus.UNTRANSLATED)])
        self.assertEqual(data["greet"], "hello")

    def test_source_with_bom_is_read(self):
        _, _, data = self._run([self._item(["greet"], "你好")], bom=True)
        self.assertEqual(data["greet"], "你好")
        self.assertEqual(data["count"], 42)

    def test_empty_translation_does_not_blank_the_key(self):
        _, _, data = self._run([self._item(["greet"], "   ")])
        self.assertEqual(data["greet"], "hello")


if __name__ == "__main__":
    unittest.main()
