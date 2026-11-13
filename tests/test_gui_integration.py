"""Qt + real CUDA engine integration, with isolated conversation storage.
Run with QT_QPA_PLATFORM=offscreen and TRANCE_TEST_BINARY set as needed.
"""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from gui.app import MainWindow
from gui.storage import ROOT, read_json
from gui.widgets import STYLE
from gui import training


class GuiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion");cls.app.setStyleSheet(STYLE)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="trance-qt-test-")
        self.state=Path(self.temp.name)
        self.window=MainWindow(state_dir=self.state,binary=Path(os.environ.get("TRANCE_TEST_BINARY",ROOT/"trance")))
        self.window.max_tokens.setValue(8);self.window.show()

    def tearDown(self):
        self.window.close();self.app.processEvents();self.temp.cleanup()

    def wait_for(self,predicate,timeout=30):
        end=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>end:self.fail("Timed out; status: "+self.window.statusBar().currentMessage())
            QTest.qWait(15)

    def send(self,text):
        self.window.composer.setPlainText(text)
        QTest.keyClick(self.window.composer,Qt.Key.Key_Return)
        self.wait_for(lambda:self.window.operation is None)
        message=self.window.current_chat['messages'][-1]
        self.assertEqual(message['status'],'complete',message)
        return message

    def test_real_chat_context_and_saved_history(self):
        first=self.send('Hi!')
        self.assertTrue(first['content'])
        second=self.send('What did I just say?')
        self.assertGreater(second['prompt_tokens'],first['prompt_tokens'])
        self.assertEqual(len(self.window.current_chat['messages']),4)
        saved=read_json(self.state/'chats.json')
        self.assertEqual(saved[0]['messages'][-1]['content'],second['content'])
        self.assertIn('Context:',self.window.chat_hint.text())
        self.assertTrue(self.window.send_button.isEnabled())

    def test_sequential_comparison_with_identical_seed(self):
        self.window.compare_a.setCurrentIndex(1);self.window.compare_b.setCurrentIndex(1)
        self.window.compare_prompt.setPlainText('Hi!')
        self.window._start_compare()
        self.wait_for(lambda:self.window.operation is None)
        a,b=self.window.comparison['responses']
        self.assertEqual(a['status'],'complete',a);self.assertEqual(b['status'],'complete',b)
        self.assertEqual(a['content'],b['content'])
        self.assertEqual(len(list((self.state/'comparisons').glob('*.json'))),1)
        self.assertTrue(self.window.compare_run.isEnabled())

    def test_stop_button_and_next_request(self):
        self.window.max_tokens.setValue(500);self.window.composer.setPlainText('Hi!');self.window._send_chat()
        self.wait_for(lambda:self.window.engine.trance._phase=='generating')
        QTest.mouseClick(self.window.stop_button,Qt.MouseButton.LeftButton)
        self.wait_for(lambda:self.window.operation is None)
        self.assertEqual(self.window.current_chat['messages'][-1]['status'],'cancelled')
        self.window.max_tokens.setValue(8);self.send('Hello again!')

    def test_training_survives_window_close_and_reconnects_for_stop_resume(self):
        directory=Path(self.temp.name)
        data=directory/'tiny-train.txt'
        data.write_text('<user>\nHi!\n\n<assistant>\nHello!\n\n<eot>\n'*100)
        config={"vocab_size":259,"context_length":32,"embedding_dim":16,"layers":1,"attention_heads":2,
                "feed_forward_dim":64,"batch_size":1,"training_steps":20000,"eval_interval":25,"checkpoint_interval":0,
                "learning_rate":.001,"gradient_clip":1,"seed":4,"train_data":str(data),"validation_data":str(data),
                "output_model":str(directory/'tiny-model.bin')}
        reopened=None
        with patch.object(training,'STATE',directory/'training-state'):
            try:
                self.window.config_editor.setPlainText(__import__('json').dumps(config))
                self.window._start_training()
                self.assertIsNotNone(self.window.training_dir,self.window.train_message.text())
                run_dir=self.window.training_dir
                self.wait_for(lambda:read_json(run_dir/'status.json',{}).get('native_ready',False))
                self.wait_for(lambda:'"step": 25' in (run_dir/'events.jsonl').read_text())
                self.window.close()
                self.assertTrue(training.run_alive(read_json(run_dir/'status.json')))
                reopened=MainWindow(state_dir=self.state,binary=Path(os.environ.get('TRANCE_TEST_BINARY',ROOT/'trance')))
                reopened.show()
                self.assertTrue(reopened.training_active)
                self.assertEqual(reopened.training_dir,run_dir)
                reopened._stop_training()
                self.wait_for(lambda:read_json(run_dir/'status.json',{}).get('status')=='stopped')
                reopened._poll_training()
                self.assertFalse(reopened.training_active)
                self.assertTrue(reopened.resume_button.isEnabled())
                self.assertGreater(len(reopened.chart.train),0)
                self.assertGreater(len(reopened.chart.valid),0)
                reopened._resume_training()
                resumed=__import__('json').loads(reopened.config_editor.toPlainText())
                self.assertLess(resumed['training_steps'],20000)
                self.assertTrue(Path(resumed['resume_model']).exists())
                status=read_json(run_dir/'status.json')
                self.assertEqual(resumed['training_steps'],status['end_step']-status['step'])
                # Recreate a resumed-run snapshot whose input checkpoint was
                # replaced by the latest stop. Remaining budget must still
                # come from the recorded target, not the overwritten file header.
                snapshot={**config,'resume_model':resumed['resume_model'],'training_steps':19999}
                from gui.storage import atomic_json
                atomic_json(run_dir/'config.json',snapshot)
                reopened._resume_training()
                again=__import__('json').loads(reopened.config_editor.toPlainText())
                self.assertEqual(again['training_steps'],20000-status['step'])
            finally:
                if self.window.training_dir and training.run_alive(read_json(self.window.training_dir/'status.json',{})):
                    training.stop_training(self.window.training_dir)
                    self.wait_for(lambda:not training.run_alive(read_json(self.window.training_dir/'status.json',{})))
                if reopened: reopened.close()


if __name__=='__main__':unittest.main()
