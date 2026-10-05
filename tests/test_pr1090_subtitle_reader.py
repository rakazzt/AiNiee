"""Fixture regressions for the subtitle reader fixes merged from PR #1090.

The old SRT parser only started a block on a pure-digit line, so a file whose cue
identifiers were missing or non-numeric was discarded whole. These cases pin the
behaviour that replaced it.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from ModuleFolders.Domain.FileReader.SrtReader import SrtReader

META = SimpleNamespace(encoding="utf-8")


class SrtReaderFixtureTests(unittest.TestCase):
    def _read(self, content: str):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.srt"
            path.write_text(content, encoding="utf-8")
            reader = SrtReader.__new__(SrtReader)  # on_read_source needs no instance state
            return reader.on_read_source(path, META)

    def test_standard_srt_keeps_indices_and_text(self):
        cache = self._read(
            "1\n00:00:01,000 --> 00:00:02,000\n第一句\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n第二句\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第一句", "第二句"])
        self.assertEqual([i.extra.get("subtitle_number") for i in cache.items], ["1", "2"])

    def test_srt_without_index_lines_is_not_discarded(self):
        """没有 cue 序号行时，时间轴行应当直接开始一个块。"""
        cache = self._read(
            "00:00:01,000 --> 00:00:02,000\n第一句\n\n"
            "00:00:03,000 --> 00:00:04,000\n第二句\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第一句", "第二句"])
        # 没有序号就不该写入 subtitle_number，否则 writer 会写出 "None"
        for item in cache.items:
            self.assertNotIn("subtitle_number", item.extra)

    def test_non_numeric_cue_identifier_is_accepted(self):
        cache = self._read(
            "cue-1\n00:00:01,000 --> 00:00:02,000\n第一句\n\n"
            "cue-2\n00:00:03,000 --> 00:00:04,000\n第二句\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第一句", "第二句"])
        self.assertEqual([i.extra.get("subtitle_number") for i in cache.items], ["cue-1", "cue-2"])

    def test_bom_is_stripped(self):
        cache = self._read(
            "\ufeff1\n00:00:01,000 --> 00:00:02,000\n第一句\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第一句"])

    def test_multiline_cue_text_is_joined(self):
        cache = self._read(
            "1\n00:00:01,000 --> 00:00:02,000\n第一行\n第二行\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第一行\n第二行"])

    def test_broken_timecode_block_is_dropped(self):
        # 时间轴行格式错误时丢弃该块，但后续正常块仍要解析出来
        cache = self._read(
            "1\n这不是时间轴\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n第二句\n"
        )
        self.assertEqual([i.source_text for i in cache.items], ["第二句"])


if __name__ == "__main__":
    unittest.main()
