"""Replay mix isolation and preservation checks; no native engine runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_trance12_replay import build,FOCUS


def record(prompt,answer):return f'<user>\n{prompt}\n\n<assistant>\n{answer}\n\n<eot>\n'


class ReplayPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        base=self.root/'data/trance12';base.mkdir(parents=True)
        focused=[]
        for i,name in enumerate(FOCUS):
            directory=base/name;directory.mkdir()
            text=record(f'Focused question {i}',f'Focused answer {i}')
            (directory/'train.txt').write_text(text);focused.append(text)
            (directory/'validation.txt').write_text(record('Held-out sentinel','Never train this answer'))
        self.original=base/'train.txt'
        self.original.write_text(''.join(focused)+''.join(record(f'Original question {i}',f'Original answer {i}') for i in range(9)))
        (base/'validation.txt').write_text(record('Held-out sentinel','Never train this answer'))
        self.output=self.root/'data/trance12_replay/train.txt'

    def tearDown(self):self.temp.cleanup()

    def test_balanced_mix_is_repeatable_and_excludes_validation_without_altering_inputs(self):
        original=self.original.read_bytes();first=build(self.root)
        self.assertEqual(first['focused_records'],3);self.assertEqual(first['replay_records'],9)
        data=self.output.read_bytes();self.assertNotIn(b'Held-out sentinel',data)
        self.assertEqual(data.count(b'<eot>'),12)
        self.assertEqual(build(self.root),first)
        self.assertEqual(self.output.read_bytes(),data);self.assertEqual(self.original.read_bytes(),original)

    def test_changed_existing_outputs_are_preserved(self):
        self.output.parent.mkdir(parents=True);self.output.write_text('Keep existing data')
        with self.assertRaisesRegex(ValueError,'preserved'):build(self.root)
        self.assertEqual(self.output.read_text(),'Keep existing data')
        self.assertFalse((self.output.parent/'manifest.json').exists())

    def test_malformed_unlabeled_records_fail_before_publishing(self):
        self.original.write_text(self.original.read_text()+'<user>\nNo assistant target\n\n<eot>\n')
        with self.assertRaisesRegex(ValueError,'Invalid'):build(self.root)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
