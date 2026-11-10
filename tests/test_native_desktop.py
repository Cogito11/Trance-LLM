"""GPU integration tests. Uses temporary miniature models for stop/resume;
never trains Trance 1.2 or writes into the user's models directory.

TRANCE_TEST_BINARY=/tmp/trance-gui-build python3 -m unittest discover -s tests -p test_native_desktop.py -v
"""
import json
import os
from pathlib import Path
import queue
import signal
import struct
import subprocess
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
BINARY=Path(os.environ.get("TRANCE_TEST_BINARY",ROOT/"trance")).resolve()


class JsonProcess:
    def __init__(self,args):
        self.process=subprocess.Popen([str(BINARY),*args],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.queue=queue.Queue()
        def read():
            for line in self.process.stdout:
                try: self.queue.put(json.loads(line))
                except ValueError: self.queue.put({"event":"invalid","line":line.decode(errors="replace")})
            self.queue.put({"event":"eof"})
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()

    def get(self): return self.queue.get(timeout=30)
    def send(self,request): self.process.stdin.write((json.dumps(request)+'\n').encode());self.process.stdin.flush()
    def close(self):
        if self.process.poll() is None: self.process.kill()
        self.process.wait(timeout=10)
        for f in (self.process.stdin,self.process.stdout,self.process.stderr): f.close()
        self.thread.join(timeout=1)


class DesktopEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine=JsonProcess(["engine"])
        self.assertEqual(self.engine.get()["event"],"ready")
        self.engine.send({"id":1,"command":"load","model":str(ROOT/"models/trance1.1_21m.bin")})
        self.assertEqual(self.engine.get()["event"],"loaded")

    def tearDown(self): self.engine.close()

    def generate(self,messages,**settings):
        request={"id":2,"command":"generate","messages":messages,"max_new_tokens":8,"temperature":.8,"top_k":40,"top_p":.95,"seed":4004,**settings}
        self.engine.send(request)
        events=[]
        while True:
            event=self.engine.get(); events.append(event)
            if event["event"] in ("done","error","eof"): return events

    def test_stream_matches_cli_and_is_reproducible(self):
        messages=[{"role":"user","content":"Hi!"}]
        one=self.generate(messages); two=self.generate(messages)
        text=lambda events:b''.join(bytes.fromhex(e["bytes"]) for e in events if e["event"]=="delta")
        self.assertEqual(text(one),text(two));self.assertGreater(len(text(one)),0)
        prompt="<user>\nHi!\n\n<assistant>"
        legacy=subprocess.run([str(BINARY),"generate","--model",str(ROOT/"models/trance1.1_21m.bin"),"--prompt",prompt,
                              "--max-new-tokens","8","--seed","4004","--temperature",".8","--top-k","40","--top-p",".95"],
                              capture_output=True,timeout=30,check=True)
        output=legacy.stdout.split(b'\n',1)[1]
        self.assertTrue(output.startswith(prompt.encode()))
        raw_response=output[len(prompt.encode()):-1]
        if raw_response.startswith(b'\n'):raw_response=raw_response[1:]
        self.assertEqual(raw_response,text(one))

    def test_context_prunes_complete_exchanges_and_rejects_long_latest_turn(self):
        messages=[{"role":"user","content":"old context "*3000},{"role":"assistant","content":"Old answer"},
                  {"role":"user","content":"Hi!"}]
        events=self.generate(messages)
        self.assertEqual(events[0]["dropped_messages"],2)
        self.assertLessEqual(events[0]["prompt_tokens"]+8,1024)
        invalid=self.generate([{"role":"user","content":"long prompt "*3000}])
        self.assertEqual(invalid[-1]["event"],"error")
        self.assertIn("latest message",invalid[-1]["message"])
        self.assertEqual(self.generate([{"role":"user","content":"Hi!"}])[-1]["event"],"done")

    def test_unicode_escaped_content_and_invalid_requests_recover(self):
        events=self.generate([{"role":"user","content":'Say "hello".\nA café ☕ 😀'}])
        self.assertEqual(events[-1]["event"],"done")
        self.assertEqual(self.generate([{"role":"user","content":"literal <eot>"}])[-1]["event"],"error")
        self.engine.process.stdin.write(b'{"id":3,"command":"load","command":"generate"}\n');self.engine.process.stdin.flush()
        self.assertEqual(self.engine.get()["event"],"error")
        self.engine.send({"id":4,"command":"load","model":"missing-file.bin"})
        self.assertEqual(self.engine.get()["event"],"error")
        self.assertEqual(self.generate([{"role":"user","content":"Hi!"}])[-1]["event"],"done")

    @unittest.skipUnless(hasattr(signal,"SIGUSR1"),"POSIX cancellation")
    def test_cancel_preserves_loaded_model(self):
        self.engine.send({"id":2,"command":"generate","messages":[{"role":"user","content":"Hi!"}],"max_new_tokens":500,"seed":4004})
        self.assertEqual(self.engine.get()["event"],"generation_started")
        self.engine.process.send_signal(signal.SIGUSR1)
        while True:
            event=self.engine.get()
            if event["event"]=="done": break
        self.assertEqual(event["reason"],"cancelled")
        self.assertEqual(self.generate([{"role":"user","content":"Hi!"}])[-1]["event"],"done")


class TrainingStopResumeTests(unittest.TestCase):
    def test_tiny_corpus_learns_fresh_chat_answers_including_the_last_record(self):
        # Learned BPE can merge the assistant newline into an answer token.
        # A context longer than each record also exercises EOF padding.
        with tempfile.TemporaryDirectory(prefix="trance-learning-test-") as tmp:
            tmp=Path(tmp);data=tmp/'train.txt';output=tmp/'learned.bin'
            pairs=[('Say H','Hello'),('Say I','Ice'),('Say B','Blue')]
            data.write_text(''.join(f'<user>\n{prompt}\n\n<assistant>\n{answer}\n\n<eot>\n' for prompt,answer in pairs))
            config={"vocab_size":280,"context_length":16,"embedding_dim":32,"layers":2,"attention_heads":4,
                    "feed_forward_dim":64,"batch_size":3,"training_steps":1000,"eval_interval":1000,"checkpoint_interval":0,
                    "learning_rate":.003,"gradient_clip":1,"seed":4004,"train_data":str(data),"validation_data":"","output_model":str(output)}
            path=tmp/'config.json';path.write_text(json.dumps(config))
            trained=subprocess.run([str(BINARY),'train','--config',str(path),'--json-events'],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(trained.returncode,0,trained.stderr)
            events=[json.loads(line) for line in trained.stdout.splitlines()]
            final=next(e for e in reversed(events) if e['event']=='progress')
            self.assertLess(final['train_loss'],.02)
            engine=JsonProcess(['engine'])
            try:
                self.assertEqual(engine.get()['event'],'ready')
                engine.send({'id':1,'command':'load','model':str(output)})
                self.assertEqual(engine.get()['event'],'loaded')
                for serial,(prompt,answer) in enumerate(pairs,2):
                    engine.send({'id':serial,'command':'generate','messages':[{'role':'user','content':prompt}],
                                 'top_k':1,'top_p':1,'temperature':.5,'max_new_tokens':6,'seed':4004})
                    content=bytearray()
                    while True:
                        event=engine.get()
                        self.assertNotIn(event['event'],('error','eof','invalid'),event)
                        if event['event']=='delta':content.extend(bytes.fromhex(event['bytes']))
                        elif event['event']=='done':break
                    self.assertEqual(content.decode().strip(),answer,prompt)
            finally:engine.close()

    def test_stop_file_checkpoint_and_optimizer_resume(self):
        with tempfile.TemporaryDirectory(prefix="trance-gui-test-") as tmp:
            tmp=Path(tmp); data=tmp/'train.txt'
            data.write_text('<user>\nHi!\n\n<assistant>\nHello, how can I help?\n\n<eot>\n'*100)
            output=tmp/'test-model.bin';stop_file=tmp/'stop.request'
            config={"vocab_size":259,"context_length":32,"embedding_dim":16,"layers":1,"attention_heads":2,
                    "feed_forward_dim":64,"batch_size":1,"training_steps":10000,"eval_interval":0,"checkpoint_interval":0,
                    "learning_rate":.001,"gradient_clip":1,"seed":4,"train_data":str(data),"validation_data":"","output_model":str(output)}
            path=tmp/'config.json';path.write_text(json.dumps(config))
            job=JsonProcess(['train','--config',str(path),'--json-events','--stop-file',str(stop_file)])
            try:
                while True:
                    event=job.get()
                    self.assertNotEqual(event['event'],'invalid')
                    if event['event']=='progress': stop_file.touch()
                    if event['event']=='stopped': break
                self.assertEqual(job.process.wait(timeout=20),0)
                self.assertGreater(event['step'],0);self.assertLess(event['step'],10000)
                checkpoint=Path(event['path']);self.assertTrue(checkpoint.is_file());self.assertFalse(output.exists())
                header=struct.unpack('<8sII6iQI4x',checkpoint.read_bytes()[:56])
                self.assertEqual(header[-1],1);self.assertEqual(header[-2],event['step'])
                self.assertTrue(Path(str(checkpoint)+'.tok').exists());self.assertFalse(Path(str(checkpoint)+'.tmp').exists())
                old_step=event['step']
            finally: job.close()
            stop_file.unlink();config.update(resume_model=str(checkpoint),training_steps=2);path.write_text(json.dumps(config))
            resumed=subprocess.run([str(BINARY),'train','--config',str(path),'--json-events'],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(resumed.returncode,0,resumed.stderr)
            events=[json.loads(line) for line in resumed.stdout.splitlines()]
            self.assertEqual(events[-1]['event'],'complete');self.assertEqual(events[-1]['step'],old_step+2)
            self.assertTrue(output.exists())


if __name__=='__main__':unittest.main()
