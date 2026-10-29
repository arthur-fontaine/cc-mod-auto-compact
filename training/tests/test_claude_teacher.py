import unittest
from unittest import mock

import helpers  # noqa: F401

import claude_teacher


class LetterTest(unittest.TestCase):
    def test_one_option_letter_or_none(self):
        self.assertEqual(claude_teacher.letter("B"), "B")
        self.assertEqual(claude_teacher.letter(" **c**\n"), "C")
        self.assertEqual(claude_teacher.letter("A."), "A")
        self.assertIsNone(claude_teacher.letter("Based on the memory"))
        self.assertIsNone(claude_teacher.letter("F"))
        self.assertIsNone(claude_teacher.letter(""))


class AskTest(unittest.TestCase):
    def test_retries_then_caches(self):
        state = {"conversation": "0+1|user: hi", "latest_user_message": "next"}
        with mock.patch.object(claude_teacher, "CACHE", claude_teacher.Path(self.tmp)), \
                mock.patch.object(claude_teacher, "call", side_effect=["Sure, here", "D"]) as call:
            self.assertEqual(claude_teacher.ask("m", state)[0], "independent")
            self.assertEqual(claude_teacher.ask("m", state), ("independent", 0.0))
        self.assertEqual(call.call_count, 2)

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()


if __name__ == "__main__":
    unittest.main()
