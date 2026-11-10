"""Protocol/report checks with a CPU mock; no models or CUDA are touched."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT=Path(__file__).resolve().parents[1]/"tools/compare_checkpoints.py"


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=self.root/"model.bin";self.model.touch();Path(str(self.model)+".tok").touch()
        self.prompts=self.root/"prompts.json"
        self.prompts.write_text(json.dumps([{"prompt":"Hello"},{"prompt":"Another question"}]))
        self.report=self.root/"report.json";self.engine=self.root/"mock-engine"
        self.engine.write_text('''#!/usr/bin/env python3
import json,sys
print(json.dumps({'event':'ready','protocol':1,'gpu':'CPU mock'}),flush=True)
for line in sys.stdin:
 r=json.loads(line);i=r['id'];cmd=r['command']
 def emit(kind,**fields):print(json.dumps({'event':kind,'id':i,**fields}),flush=True)
 if cmd=='load':emit('loaded',step=40000)
 elif cmd=='generate':
  assert len(r['messages'])==1
  assert r['seed']==4004 and r['top_k']==1
  emit('generation_started',prompt_tokens=5)
  raw='Hello 🌌'.encode()
  for part in (raw[:7],raw[7:]):emit('delta',bytes=part.hex())
  emit('done',reason='end_of_turn',generated_tokens=8,elapsed_seconds=.01)
 elif cmd=='shutdown':emit('bye');break
''');self.engine.chmod(0o755)

    def tearDown(self):self.temp.cleanup()

    def run_comparison(self,*extra):
        return subprocess.run([sys.executable,str(SCRIPT),str(self.model),"--engine",str(self.engine),
                               "--prompts",str(self.prompts),"--top-k","1","--output",str(self.report),*extra],
                              capture_output=True,text=True,timeout=5)

    def test_fresh_context_and_split_utf8_survive_in_a_reviewable_report(self):
        result=self.run_comparison()
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(self.report.read_text())
        self.assertTrue(report['fresh_context_per_prompt'])
        self.assertEqual(report['settings']['top_k'],1)
        self.assertEqual([r['content'] for r in report['results'][0]['responses']],["Hello 🌌"]*2)
        self.assertEqual(report['results'][0]['step'],40000)

    def test_existing_reports_are_preserved(self):
        self.report.write_text('keep this report')
        result=self.run_comparison()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.report.read_text(),'keep this report')

    def test_stalled_engine_is_terminated_without_a_false_success_report(self):
        self.engine.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(20)\n')
        result=self.run_comparison('--timeout','.1')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('timed out',result.stderr)
        self.assertFalse(self.report.exists())


if __name__=='__main__':unittest.main()
