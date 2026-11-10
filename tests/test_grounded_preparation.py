import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_trance12_grounded import evaluation_rows, fact_pairs, family_key, synthetic
from prepare_trance12 import raw_token_bound, render


class GroundedTests(unittest.TestCase):
    def test_answers_follow_changed_facts(self):
        rows = fact_pairs(['Ada', 'Ben', 'Cora'], ['book', 'cup', 'hat'],
                          ['red', 'orange', 'blue'],
                          ['on the shelf', 'under the desk', 'by the bed'], [2, 0, 1], 0)
        # Same question, ownership changes; all four task types checked independently.
        self.assertEqual([t[-1][1] for t in rows[:8]],
                         ['book', 'cup', 'on the shelf', 'by the bed',
                          'red', 'blue', 'Ada', 'Cora'])
        self.assertIn('Ben owns an orange cup', rows[0][0][1])
        for original, changed in zip(rows[::2], rows[1::2]):
            self.assertEqual(original[0][1].split('Question: ')[1], changed[0][1].split('Question: ')[1])
            self.assertNotEqual(original[-1][1], changed[-1][1])

    def test_permutations_cannot_cross_splits(self):
        a = [['Ada', 'Ben', 'Cora'], ['book', 'cup', 'hat'],
             ['red', 'orange', 'blue'], ['shelf', 'desk', 'bed']]
        self.assertEqual(family_key(*a), family_key(*(x[::-1] for x in a)))
        rows, families = synthetic(20, 5, 91)
        self.assertFalse(set(families['train']) & set(families['validation']))
        for split, expected in [('train', 320), ('validation', 80)]:
            self.assertEqual(len(rows[split]), expected)
            for turns in rows[split]:
                self.assertLessEqual(raw_token_bound(turns), 1024)
                self.assertIn(turns[-1][1], turns[0][1])
        self.assertEqual((rows, families), synthetic(20, 5, 91))

    def test_evaluation_keeps_changed_answer_pairs(self):
        rows, _ = synthetic(2, 6, 93)
        questions = evaluation_rows([render(t) for t in rows['validation']])
        self.assertEqual(len(questions), 12)
        self.assertEqual(len({r['prompt'] for r in questions}), 12)
        for a, b in zip(questions[::2], questions[1::2]):
            self.assertEqual(a['pair_id'], b['pair_id'])
            self.assertNotEqual(a['expected_answer'], b['expected_answer'])


if __name__ == '__main__':
    unittest.main()
