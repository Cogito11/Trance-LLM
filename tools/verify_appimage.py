#!/usr/bin/env python3
"""CPU-only checks of the actual AppImage: startup, layout, independent worker.
No native CUDA commands are executed. The worker test uses an isolated project
and a mock engine that only emits JSON events.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui import APPIMAGE_NAME
IMAGE=ROOT/"dist"/APPIMAGE_NAME


def json_output(result):
    for line in reversed(result.stdout.splitlines()):
        try: return json.loads(line)
        except ValueError: pass
    raise ValueError(result.stdout+result.stderr)


def main():
    env={**os.environ,"APPIMAGE_EXTRACT_AND_RUN":"1","QT_QPA_PLATFORM":"offscreen"}
    diagnostics=subprocess.run([str(IMAGE),"--project",str(ROOT),"--diagnostics"],env=env,capture_output=True,text=True,timeout=90,check=True)
    metadata=json_output(diagnostics)
    assert metadata['frozen'] and metadata['engine_exists']
    assert metadata['project']==str(ROOT)
    assert metadata['documentation_articles']>=14
    screenshot=ROOT/"dist/TranceLLM-preview.png"
    subprocess.run([str(IMAGE),"--project",str(ROOT),"--screenshot",str(screenshot)],env=env,capture_output=True,text=True,timeout=90,check=True)
    assert screenshot.is_file() and screenshot.read_bytes().startswith(b'\x89PNG')
    for page in ("configs","docs"):
        capture=ROOT/"dist"/f"TranceLLM-{page}-preview.png"
        subprocess.run([str(IMAGE),"--project",str(ROOT),"--page",page,"--screenshot",str(capture)],env=env,capture_output=True,text=True,timeout=90,check=True)
        assert capture.is_file() and capture.read_bytes().startswith(b'\x89PNG')
    with tempfile.TemporaryDirectory(prefix="trance-appimage-proof-") as temp:
        project=Path(temp); (project/"configs").mkdir(); (project/"models").mkdir();(project/"data").mkdir()
        data=project/"data/train.txt"; data.write_text("CPU-only packaging fixture\n")
        mock=project/"mock-engine"
        mock.write_text('''#!/usr/bin/python3
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config');p.add_argument('--stop-file');p.add_argument('--json-events',action='store_true');a=p.parse_args()
c=json.loads(Path(a.config).read_text())
def emit(v):print(json.dumps(v),flush=True)
emit({'event':'stage','stage':'CPU-only packaging test'})
emit({'event':'training_started','step':0,'end_step':c['training_steps']})
for step in range(1,c['training_steps']+1):
 time.sleep(.05)
 emit({'event':'progress','step':step,'end_step':c['training_steps'],'train_loss':1.0,'valid_loss':1.0,'elapsed_seconds':step*.05})
 if Path(a.stop_file).exists():
  checkpoint=str(c['output_model'])+'.interrupted.bin';Path(checkpoint).write_text('mock checkpoint')
  emit({'event':'stopped','step':step,'path':checkpoint});break
else:
 Path(c['output_model']).write_text('mock model')
 emit({'event':'complete','step':c['training_steps'],'path':c['output_model']})
'''); mock.chmod(0o755)
        config={"vocab_size":259,"context_length":32,"embedding_dim":16,"layers":1,"attention_heads":2,
                "feed_forward_dim":64,"batch_size":1,"training_steps":200,"eval_interval":25,"checkpoint_interval":0,
                "learning_rate":.001,"gradient_clip":1,"seed":4,"train_data":str(data),"validation_data":"",
                "output_model":str(project/"models/mock.bin")}
        path=project/"configs/test.json"; path.write_text(json.dumps(config))
        probe_env={**env,"TRANCE_ENGINE_BINARY":str(mock)}
        result=subprocess.run([str(IMAGE),"--project",str(project),"--verify-detached-worker",str(path)],
                              env=probe_env,capture_output=True,text=True,timeout=90,check=True)
        directory=Path(json_output(result)['run_dir'])
        # Parent AppImage has exited and removed its extracted runtime now.
        # Its separately launched worker must still start and run successfully.
        status={};deadline=time.monotonic()+90
        try:
            while time.monotonic()<deadline:
                status=json.loads((directory/"status.json").read_text())
                if status.get('status')=='failed': raise AssertionError(status)
                if status.get('native_ready'):break
                time.sleep(.1)
            assert status.get('status')=='running' and status.get('native_ready'),status
            assert status['engine_pid']!=os.getpid()
            (directory/"stop.request").touch()
            while time.monotonic()<deadline:
                status=json.loads((directory/"status.json").read_text())
                if status.get('status') not in ('running','starting'):break
                time.sleep(.1)
            assert status['status']=='stopped',status
            assert status['end_step']==200 and status['step']<200
            assert Path(status['checkpoint']).is_file()
        finally:
            (directory/"stop.request").touch()
    manifest=json.loads(Path(str(IMAGE)+'.json').read_text())
    sha=hashlib.sha256()
    with IMAGE.open('rb') as f:
        while chunk:=f.read(1024*1024):sha.update(chunk)
    assert sha.hexdigest()==manifest['sha256']
    report={"appimage":IMAGE.name,"sha256":sha.hexdigest(),"startup":True,"layout_capture":True,
            "detached_worker_survives_parent_exit":True,"checkpointed_stop_protocol":True,
            "config_builder_startup":True,"offline_documentation_articles":metadata['documentation_articles'],
            "gpu_execution_performed":False,"training_job_untouched":True}
    (ROOT/"dist/AppImage-verification.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
