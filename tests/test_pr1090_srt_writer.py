"""Writer-side regression for the SRT fix merged from PR #1090.

_mapping a cue with no identifier used to emit str(None), writing the literal
word "None" into the subtitle file as a cue number. This is the counterpart to
the reader fix that made identifiers optional.
"""
import unittest

from ModuleFolders.Domain.FileOutputer.SrtWriter import SrtWriter
from ModuleFolders.Service.Cache.CacheItem import CacheItem

TIMECODE = "00:00:01,000 --> 00:00:02,000"


class SrtWriterCueNumberTests(unittest.TestCase):
    def setUp(self):
        self.writer = SrtWriter.__new__(SrtWriter)  # _map_to_translated_item needs no instance state

    def _item(self, **extra):
        return CacheItem(source_text="原文", translated_text="译文", extra=extra)

    def test_cue_with_identifier_keeps_its_number(self):
        block = self.writer._map_to_translated_item(
            self._item(subtitle_time=TIMECODE, subtitle_number="cue-1")
        )
        self.assertEqual(block[0], "cue-1")
        self.assertEqual(block[1], TIMECODE)

    def test_cue_without_identifier_does_not_write_none(self):
        block = self.writer._map_to_translated_item(self._item(subtitle_time=TIMECODE))
        joined = "\n".join(block)
        self.assertNotIn("None", joined)
        # 时间轴行成为首行，而不是被一个假的序号行顶掉
        self.assertEqual(block[0], TIMECODE)

    def test_translated_text_is_preserved(self):
        block = self.writer._map_to_translated_item(self._item(subtitle_time=TIMECODE))
        self.assertIn("译文", "\n".join(block))


if __name__ == "__main__":
    unittest.main()
