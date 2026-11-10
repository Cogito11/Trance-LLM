import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gui import storage
from gui.training import EventTail


class StorageTests(unittest.TestCase):
    def test_history_omits_interrupted_turns(self):
        messages=[{"role":"user","content":"First"},{"role":"assistant","content":"Answer","status":"complete"},
                  {"role":"user","content":"Second"},{"role":"assistant","content":"Partial","status":"cancelled"}]
        self.assertEqual(storage.completed_context(messages),[{"role":"user","content":"First"},{"role":"assistant","content":"Answer"}])

    def test_atomic_unicode_history_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"history.json"
            storage.atomic_json(path,{"text":"你好 🌙\nquoted \"text\""})
            self.assertEqual(storage.read_json(path)["text"],"你好 🌙\nquoted \"text\"")
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_real_model_metadata_and_preserved_output(self):
        info=storage.model_info(storage.ROOT/"models/trance1.1_21m.bin")
        self.assertEqual(info["parameters"],22475008)
        self.assertEqual(info["step"],140000)
        config=json.loads((storage.ROOT/"configs/trance1.2_30m.json").read_text())
        self.assertEqual(storage.validate_config(config),config)
        config["output_model"]="models/trance1.1_21m.bin"
        with self.assertRaisesRegex(ValueError,"already exists"): storage.validate_config(config)
        config["resume_model"]="models/trance1.1_21m.bin"
        with self.assertRaisesRegex(ValueError,"optimizer"): storage.validate_config(config)

    def test_invalid_training_settings_do_not_start_a_job(self):
        config=json.loads((storage.ROOT/"configs/trance1.2_30m.json").read_text())
        for key,value in (("attention_heads",9),("learning_rate",float("nan")),("training_steps",0),("dropout",.1)):
            invalid={**config,key:value}
            with self.assertRaises(ValueError): storage.validate_config(invalid)

    def test_log_tail_waits_for_complete_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"events.jsonl"; tail=EventTail(Path(tmp))
            path.write_bytes(b'{"event":"progress","step":25}')
            self.assertEqual(tail.read(),[])
            with path.open("ab") as f: f.write(b'\n{"event":"complete"}\n')
            self.assertEqual([e["event"] for e in tail.read()],["progress","complete"])
            self.assertEqual(tail.read(),[])


if __name__=="__main__": unittest.main()
