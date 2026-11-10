"""CPU-only sidebar and temporary-chat behavior; no CUDA process is launched."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from gui import APP_NAME
from gui.app import MainWindow
from gui.storage import read_json, atomic_json, new_chat
from gui.widgets import STYLE


class ChatNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(STYLE)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="trance-navigation-")
        self.state=Path(self.temp.name)
        self.runs=patch("gui.app.latest_run",return_value=None)
        self.runs.start()
        self.window=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"))
        self.window.timer.stop()
        self.generate=patch.object(self.window.engine,"generate")
        self.mock_generate=self.generate.start()
        self.window.model_combo.clear()
        self.window.model_combo.addItem("CPU mock model","/unused-cpu-model.bin")
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.generate.stop();self.runs.stop();self.temp.cleanup()

    def reply(self,text,answer="CPU mock reply"):
        self.window.composer.setPlainText(text)
        QTest.keyClick(self.window.composer,Qt.Key.Key_Return)
        self.mock_generate.assert_called()
        self.window._delta(answer)
        self.window._finished({"reason":"end","generated_tokens":3,"elapsed_seconds":.01})

    def test_sidebar_actions_navigate_to_the_requested_pages(self):
        self.assertEqual(self.window.windowTitle(),APP_NAME)
        sidebar=self.window.new_chat_button.parentWidget()
        entries=[]
        for i in range(sidebar.layout().count()):
            widget=sidebar.layout().itemAt(i).widget()
            if isinstance(widget,(QLabel,QPushButton)) and widget.objectName() in ("eyebrow","nav"):
                entries.append(widget.text().rstrip("  ▸▾"))
        self.assertEqual(entries[:10],["CHAT","Temporary chat","New chat","Chat history","TRAIN","Train","Configs","TOOLS","Compare","Documentation"])
        QTest.mouseClick(self.window.nav[2],Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.pages.currentIndex(),2)
        QTest.mouseClick(self.window.nav[1],Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.pages.currentIndex(),1)
        QTest.mouseClick(self.window.new_chat_button,Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.pages.currentIndex(),0)
        self.assertTrue(self.window.new_chat_button.isChecked())

    def test_temporary_chat_keeps_context_but_never_enters_disk_history(self):
        saved=read_json(self.state/"chats.json",[])
        QTest.mouseClick(self.window.temporary_chat_button,Qt.MouseButton.LeftButton)
        temporary_id=self.window.current_chat["id"]
        self.assertTrue(self.window.temporary_notice.isVisible())
        self.assertGreaterEqual(self.window.temporary_notice.height(),self.window.temporary_notice.heightForWidth(self.window.temporary_notice.width()))
        self.assertEqual(self.window.chat_list.currentRow(),-1)
        self.reply("private first message")
        self.reply("private followup")
        context=self.mock_generate.call_args.args[1]
        self.assertEqual([m["role"] for m in context],["user","assistant","user"])
        self.assertNotIn(temporary_id,[c["id"] for c in self.window.chats])
        self.assertEqual(read_json(self.state/"chats.json",[]),saved)
        self.window.close()
        self.assertEqual(read_json(self.state/"chats.json",[]),saved)

    def test_history_restores_saved_chat_and_discards_temporary_content(self):
        self.reply("saved greeting")
        saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True)
        self.reply("temporary secret")
        temporary_id=self.window.current_chat["id"]
        QTest.mouseClick(self.window.history_button,Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.current_chat["id"],temporary_id)
        self.window.chat_list.setCurrentRow(0)
        self.assertEqual(self.window.current_chat["id"],saved_id)
        self.assertFalse(self.window.temporary_notice.isVisible())
        self.assertTrue(self.window.chat_list.isVisible())
        self.assertTrue(self.window.history_button.isChecked())
        self.assertNotIn("temporary secret",str(read_json(self.state/"chats.json")))

    def test_new_chat_after_temporary_chat_saves_only_the_new_conversation(self):
        self.window._new_chat(temporary=True)
        self.reply("temporary secret")
        self.window._new_chat()
        self.reply("persist this")
        saved=read_json(self.state/"chats.json")
        self.assertEqual(saved[0]["messages"][0]["content"],"persist this")
        self.assertNotIn("temporary secret",str(saved))

    def test_chat_heading_follows_new_saved_and_temporary_contexts(self):
        self.assertEqual(self.window.chat_title.text(),"New Chat")
        self.reply("saved conversation")
        self.assertEqual(self.window.chat_title.text(),"Chat")
        saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True)
        self.assertEqual(self.window.chat_title.text(),"Temporary Chat")
        self.reply("temporary message")
        self.assertEqual(self.window.chat_title.text(),"Temporary Chat")
        self.window._show_history()
        self.window.chat_list.setCurrentRow(0)
        self.assertEqual(self.window.current_chat["id"],saved_id)
        self.assertEqual(self.window.chat_title.text(),"Chat")
        self.window._new_chat()
        self.assertEqual(self.window.chat_title.text(),"New Chat")

    def test_embedded_send_stop_and_header_export_keep_their_behavior(self):
        for widget in (self.window.send_button,self.window.stop_button):
            self.assertTrue(self.window.composer_panel.isAncestorOf(widget))
        self.assertFalse(self.window.composer_panel.isAncestorOf(self.window.model_controls))
        self.assertFalse(self.window.transcript.toPlainText().strip())
        self.window.composer.setPlainText("send from the embedded button")
        QTest.mouseClick(self.window.send_button,Qt.MouseButton.LeftButton)
        self.assertEqual(self.mock_generate.call_args.args[1][-1]["content"],"send from the embedded button")
        self.assertTrue(self.window.stop_button.isEnabled())
        with patch.object(self.window.engine,"cancel") as cancel:
            QTest.mouseClick(self.window.stop_button,Qt.MouseButton.LeftButton)
            cancel.assert_called_once()
        self.window._delta("mock reply")
        self.window._finished({"reason":"end","generated_tokens":3,"elapsed_seconds":.01})
        self.assertFalse(self.window.statusBar().isVisible())
        path=self.state/"export.json"
        with patch("gui.app.QFileDialog.getSaveFileName",return_value=(str(path),"JSON")):
            QTest.mouseClick(self.window.chat_export_button,Qt.MouseButton.LeftButton)
        self.assertEqual(read_json(path)["messages"][-1]["content"],"mock reply")

    def test_message_box_grows_for_wrapped_text_and_shrinks_when_cleared(self):
        self.app.processEvents()
        initial=self.window.composer.height()
        self.assertLess(initial,60)
        QTest.keyClicks(self.window.composer,"first line")
        QTest.keyClick(self.window.composer,Qt.Key.Key_Return,Qt.KeyboardModifier.ShiftModifier)
        self.assertIn("\n",self.window.composer.toPlainText())
        self.mock_generate.assert_not_called()
        self.window.composer.setPlainText("A longer message that needs to wrap. "*100)
        self.app.processEvents()
        self.assertGreater(self.window.composer.height(),initial)
        self.assertGreater(self.window.composer.verticalScrollBar().maximum(),0)
        self.window.composer.clear();self.app.processEvents()
        self.assertEqual(self.window.composer.height(),initial)

    def test_empty_drafts_only_enter_history_when_a_message_is_sent(self):
        self.assertEqual(self.window.chats,[])
        self.assertEqual(read_json(self.state/"chats.json",[]),[])
        for _ in range(3): self.window._new_chat()
        draft_id=self.window.current_chat["id"]
        self.assertEqual(self.window.chat_list.count(),0)
        self.assertEqual(read_json(self.state/"chats.json",[]),[])
        self.window.composer.setPlainText("unsent draft")
        self.window._new_chat()
        self.assertNotEqual(self.window.current_chat["id"],draft_id)
        self.assertEqual(self.window.chats,[])
        self.reply("first sent message")
        saved=read_json(self.state/"chats.json")
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]["messages"][0]["content"],"first sent message")
        self.reply("second sent message")
        self.assertEqual(len(read_json(self.state/"chats.json")),1)
        self.window._new_chat();self.window.close()
        self.assertEqual(len(read_json(self.state/"chats.json")),1)

    def test_temporary_notice_occupies_the_empty_canvas_and_yields_to_messages(self):
        self.window._new_chat(temporary=True);self.app.processEvents()
        self.assertEqual(self.window.temporary_notice.text(),"Chat Not Saved to History. Cleared when you switch chats or close the app.")
        self.assertTrue(self.window.temporary_notice.isVisible())
        center=self.window.temporary_notice.mapTo(self.window.chat_content,self.window.temporary_notice.rect().center())
        canvas=self.window.chat_content.rect().center()
        self.assertLess(abs(center.x()-canvas.x()),3)
        self.assertLess(abs(center.y()-canvas.y()),3)
        self.reply("temporary conversation")
        self.assertFalse(self.window.temporary_notice.isVisible())
        self.assertTrue(self.window.transcript.isVisible())

    def test_old_empty_history_entries_are_filtered_without_losing_messages(self):
        state=self.state/"legacy"
        sent=new_chat();sent["messages"]=[{"role":"user","content":"keep this message"}]
        atomic_json(state/"chats.json",[new_chat(),sent,new_chat()])
        window=MainWindow(state_dir=state,binary=Path("/unused-cpu-test-engine"))
        window.timer.stop()
        try:
            self.assertEqual([c["id"] for c in window.chats],[sent["id"]])
        finally: window.close()
        self.assertEqual(read_json(state/"chats.json"),[sent])

    def test_history_expansion_persists_immediately_and_across_navigation_and_restart(self):
        self.assertFalse(self.window.history_expanded)
        self.window.top_p.setValue(.85)
        QTest.mouseClick(self.window.history_button,Qt.MouseButton.LeftButton)
        self.assertTrue(read_json(self.state/"settings.json")["history_expanded"])
        self.window._new_chat(temporary=True);self.window._show_page(3)
        self.assertTrue(self.window.chat_list.isVisible())
        self.assertTrue(self.window.history_button.isChecked())
        self.window._show_history()
        self.assertFalse(read_json(self.state/"settings.json")["history_expanded"])
        self.window._show_history();self.window.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused"))
        reopened.timer.stop();reopened.show()
        try:
            self.assertTrue(reopened.chat_list.isVisible())
            self.assertTrue(reopened.history_button.isChecked())
            self.assertEqual(reopened.top_p.value(),.85)
        finally: reopened.close()

    def test_history_scrolls_without_pushing_navigation_outside_the_window(self):
        for i in range(60):
            chat=new_chat();chat.update(title=f"Saved chat {i}",messages=[{"role":"user","content":"hello"}])
            self.window.chats.append(chat)
        self.window._refresh_chats();self.window._show_history()
        self.window.resize(1000,690);QTest.qWait(50)
        self.assertLessEqual(self.window.chat_list.height(),180)
        self.assertGreater(self.window.chat_list.verticalScrollBar().maximum(),0)
        self.assertLess(self.window.gpu_status.mapTo(self.window,self.window.gpu_status.rect().bottomRight()).y(),self.window.height())
        bar=self.window.chat_list.verticalScrollBar();bar.setValue(20)
        self.window._refresh_chats()
        self.assertEqual(bar.value(),20)

    def test_context_menu_targets_clicked_chat_and_right_click_preserves_temporary_chat(self):
        self.reply("saved chat");saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True);self.window._show_history();QTest.qWait(20)
        temporary_id=self.window.current_chat["id"]
        position=self.window.chat_list.visualItemRect(self.window.chat_list.item(0)).center()
        QTest.mousePress(self.window.chat_list.viewport(),Qt.MouseButton.RightButton,pos=position)
        QTest.mouseRelease(self.window.chat_list.viewport(),Qt.MouseButton.RightButton,pos=position)
        self.assertEqual(self.window.current_chat["id"],temporary_id)
        for chosen,method in ((sentinel.rename,"_rename_chat"),(sentinel.delete,"_delete_chat")):
            with patch("gui.app.QMenu") as menu,patch.object(self.window,method) as callback:
                menu.return_value.addAction.side_effect=[sentinel.rename,sentinel.delete]
                menu.return_value.exec.return_value=chosen
                self.window._chat_context_menu(position)
                self.assertEqual([c.args[0] for c in menu.return_value.addAction.call_args_list],["Rename chat…","Delete chat"])
                callback.assert_called_once_with(saved_id)

    def test_rename_persists_while_temporary_chat_remains_private(self):
        self.reply("saved chat");saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True);self.reply("temporary secret")
        with patch("gui.app.QInputDialog.getText",return_value=("  My Aurora 🌌  ",True)):
            self.window._rename_chat(saved_id)
        saved=read_json(self.state/"chats.json")
        self.assertEqual(saved[0]["title"],"My Aurora 🌌")
        self.assertNotIn("temporary secret",str(saved))
        self.assertTrue(self.window.current_chat["temporary"])
        self.assertEqual(self.window.chat_list.item(0).text(),"My Aurora 🌌")

    def test_cancelled_or_empty_rename_preserves_title(self):
        self.reply("original title");chat=self.window.current_chat
        for result in (("replacement",False),(" \n ",True)):
            with patch("gui.app.QInputDialog.getText",return_value=result):self.window._rename_chat(chat["id"])
        self.assertEqual(chat["title"],"original title")
        self.assertEqual(read_json(self.state/"chats.json")[0]["title"],"original title")

    def test_deleting_active_chat_selects_neighbor_then_leaves_an_unsaved_draft(self):
        self.reply("older");older_id=self.window.current_chat["id"]
        self.window._new_chat();self.reply("newer");newer_id=self.window.current_chat["id"]
        self.window._delete_chat(newer_id)
        self.assertEqual(self.window.current_chat["id"],older_id)
        self.window._delete_chat(older_id)
        self.assertEqual(read_json(self.state/"chats.json"),[])
        self.assertEqual(self.window.chat_title.text(),"New Chat")
        self.assertEqual(self.window.chat_list.count(),0)

    def test_deleting_an_unrelated_chat_preserves_temporary_messages(self):
        self.reply("saved");saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True);self.reply("private content")
        temporary_id=self.window.current_chat["id"]
        self.window._delete_chat(saved_id)
        self.assertEqual(self.window.current_chat["id"],temporary_id)
        self.assertEqual(self.window.current_chat["messages"][0]["content"],"private content")
        self.assertEqual(read_json(self.state/"chats.json"),[])

    def test_failed_history_writes_roll_back_rename_and_delete(self):
        self.reply("keep me");chat=self.window.current_chat
        with patch("gui.app.atomic_json",side_effect=OSError("test disk failure")):
            with patch("gui.app.QInputDialog.getText",return_value=("replacement",True)):
                self.window._rename_chat(chat["id"])
            self.window._delete_chat(chat["id"])
        self.assertEqual(chat["title"],"keep me")
        self.assertEqual(self.window.chats,[chat])
        self.assertEqual(read_json(self.state/"chats.json"),[chat])


if __name__=="__main__":unittest.main()
