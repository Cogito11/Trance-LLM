import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_conditioning_probe import FACT, simplify, presence_pair, QUESTION, novel_pairs, prompt_key
from prepare_trance12 import render
from prepare_trance12_grounded import fact_pairs


class ConditioningProbeTests(unittest.TestCase):
    def test_overlap_excludes_both_members_of_pair(self):
        rows = [render([('user', f'Question {i}'), ('assistant', str(i))]) for i in range(4)]
        clean, removed = novel_pairs(''.join(rows), {prompt_key(rows[0])})
        self.assertEqual(removed, 2)
        self.assertEqual(clean, ''.join(rows[2:]))

    def test_keeps_referenced_fact_for_every_question_and_style(self):
        for style in range(3):
            rows = fact_pairs(['Ada', 'Ben', 'Cora'], ['hat', 'cup', 'book'],
                              ['red', 'blue', 'green'], ['on the shelf', 'under the desk', 'by the bed'],
                              [2, 0, 1], style)
            for count in (1, 2):
                for turns in rows:
                    result = simplify(turns, count, random.Random(99))
                    self.assertEqual(result[-1], turns[-1])
                    self.assertEqual(len(FACT.findall(result[0][1])), count)
                    self.assertIn(result[-1][1], result[0][1])

    def test_one_fact_has_the_correct_ownership_mapping(self):
        rows = fact_pairs(['Ada', 'Ben', 'Cora'], ['hat', 'cup', 'book'],
                          ['red', 'blue', 'green'], ['on the shelf', 'under the desk', 'by the bed'],
                          [2, 0, 1], 0)
        original = simplify(rows[0], 1, random.Random(1))[0][1]
        changed = simplify(rows[1], 1, random.Random(1))[0][1]
        self.assertIn('Ada owns a red hat', original)
        self.assertIn('Ada owns a red cup', changed)
        self.assertNotIn('Ben owns', original)

    def test_missing_answer_pair_really_omits_the_queried_entity(self):
        rows = fact_pairs(['Ada', 'Ben', 'Cora'], ['hat', 'cup', 'book'],
                          ['red', 'blue', 'green'], ['on the shelf', 'under the desk', 'by the bed'],
                          [2, 0, 1], 0)
        for turns in rows:
            positive, negative = presence_pair(turns, random.Random(3))
            question = QUESTION.search(turns[0][1]).group()
            self.assertIn(question, positive[0][1])
            self.assertIn(question, negative[0][1])
            self.assertEqual(positive[-1], turns[-1])
            self.assertEqual(negative[-1][1], 'Not stated.')
            fact = FACT.search(negative[0][1])
            target = question.split()[2] if question.startswith('What does') else question.split()[-1][:-1]
            self.assertNotEqual(fact.group(1 if question.startswith('What does') else 3), target)


if __name__ == '__main__':
    unittest.main()
