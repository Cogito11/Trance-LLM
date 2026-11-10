import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_trance12 import render
from prepare_trance12_foundation import story_identity


class FoundationPreparationTests(unittest.TestCase):
    def test_full_story_and_continuation_have_same_identity(self):
        full = render([('user', 'Tell me a short story.'), ('assistant', 'Ada saw a bird. It flew away.')])
        continuation = render([('user', 'Continue this story:\nAda saw a bird.'), ('assistant', 'It flew away.')])
        self.assertEqual(story_identity(full), story_identity(continuation))

    def test_unidentified_instruction_cannot_become_story_data(self):
        with self.assertRaises(AssertionError):
            story_identity(render([('user', 'What is the sun?'), ('assistant', 'A star.')]))


if __name__ == '__main__':
    unittest.main()
