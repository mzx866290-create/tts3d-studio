from __future__ import annotations

import unittest

import numpy as np

from tts3d_app.text_chunking import concatenate_mono_chunks, split_text_for_tts, take_reference_sentence


class TextChunkingTests(unittest.TestCase):
    def test_short_text_stays_one_chunk(self) -> None:
        chunks = split_text_for_tts("你好，这是短文本。", max_chars=400)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "你好，这是短文本。")
        self.assertEqual(chunks[0].pause_after_ms, 0)

    def test_packs_sentences_under_max_chars(self) -> None:
        text = "第一句。" * 10
        chunks = split_text_for_tts(text, max_chars=20, sentence_pause_ms=300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 20 for chunk in chunks))
        self.assertEqual(chunks[-1].pause_after_ms, 0)
        self.assertTrue(all(chunk.pause_after_ms == 300 for chunk in chunks[:-1]))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_paragraph_break_uses_longer_pause(self) -> None:
        text = "第一段结束。\n\n第二段开始。"
        chunks = split_text_for_tts(
            text,
            max_chars=8,
            sentence_pause_ms=300,
            paragraph_pause_ms=700,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].pause_after_ms, 700)

    def test_concatenate_inserts_silence(self) -> None:
        left = np.ones(4, dtype=np.float32)
        right = np.ones(4, dtype=np.float32)
        combined = concatenate_mono_chunks([left, right], [500], sample_rate=1000)
        self.assertEqual(combined.shape[0], 4 + 500 + 4)
        np.testing.assert_array_equal(combined[4:504], np.zeros(500, dtype=np.float32))

    def test_take_reference_sentence_packs_opening_sentences(self) -> None:
        sentence = "这是一句用来触发分块的测试。"
        text = sentence * 10
        ref = take_reference_sentence(text, max_chars=60)
        self.assertTrue(ref)
        self.assertLessEqual(len(ref), 60)
        self.assertTrue(ref.startswith(sentence))
        self.assertGreaterEqual(len(ref), len(sentence))
        self.assertTrue(text.startswith(ref))

    def test_take_reference_sentence_splits_overlong_first_sentence(self) -> None:
        text = "这是一段没有句号但有逗号，用来切开超长首句，后面还有很多字。"
        ref = take_reference_sentence(text, max_chars=20)
        self.assertTrue(ref)
        self.assertLessEqual(len(ref), 20)
        self.assertTrue(text.startswith(ref.rstrip()))

    def test_take_reference_sentence_empty_or_disabled(self) -> None:
        self.assertEqual(take_reference_sentence("", max_chars=60), "")
        self.assertEqual(take_reference_sentence("你好。", max_chars=0), "")
        self.assertEqual(take_reference_sentence("短句。", max_chars=60), "短句。")


if __name__ == "__main__":
    unittest.main()
