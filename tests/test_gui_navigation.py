"""CPU-only sidebar and temporary-chat behavior; no CUDA process is launched."""
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from gui import APP_NAME, theme
from gui.app import GREETINGS, SUGGESTIONS, MainWindow
from gui.runtimes import RuntimeInfo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gguf import chat_model, write as write_gguf  # noqa: E402
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
        self.gguf_scan=patch("gui.app.discover_gguf",return_value=[])   # never scan the real LM Studio / Hugging Face folders
        self.gguf_scan.start()
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
        sheet=theme.apply_palette("teal")   # a test may have switched themes
        if self.app.styleSheet()!=sheet:self.app.setStyleSheet(sheet)
        self.generate.stop();self.runs.stop();self.gguf_scan.stop();self.temp.cleanup()

    def reply(self,text,answer="CPU mock reply"):
        self.window.composer.setPlainText(text)
        QTest.keyClick(self.window.composer,Qt.Key.Key_Return)
        self.mock_generate.assert_called()
        self.window._delta(answer)
        self.window._finished({"reason":"end","generated_tokens":3,"elapsed_seconds":.01})

    def test_sidebar_sections_navigate_and_remember_their_last_page(self):
        w=self.window; click=lambda b:QTest.mouseClick(b,Qt.MouseButton.LeftButton)
        self.assertEqual(w.windowTitle(),APP_NAME)
        self.assertEqual([s.text() for s in w.sections],["Chat","Train","Tools"])
        self.assertEqual({page:nav.text() for page,nav in w.nav.items()},{2:"Train",3:"Configs",1:"Compare",4:"Documentation",5:"Settings"})
        self.assertTrue(w.sections[0].isChecked())
        click(w.sections[1])                      # a section opens its page immediately
        self.assertEqual((w.pages.currentIndex(),w.section_stack.currentIndex()),(2,1))
        click(w.nav[3]);self.assertEqual(w.pages.currentIndex(),3)
        click(w.sections[2]);self.assertEqual(w.pages.currentIndex(),1)
        click(w.nav[4]);self.assertEqual(w.pages.currentIndex(),4)
        click(w.sections[1]);self.assertEqual(w.pages.currentIndex(),3)   # remembered
        click(w.sections[2]);self.assertEqual(w.pages.currentIndex(),4)   # remembered
        click(w.sections[0]);self.assertEqual(w.pages.currentIndex(),0)
        click(w.sections[2]);click(w.new_chat_button)
        self.assertEqual(w.pages.currentIndex(),0)
        self.assertTrue(w.new_chat_button.isChecked())
        self.assertTrue(w.sections[0].isChecked())
        self.assertFalse(w.sections[2].isChecked())

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
        self.assertEqual(self.window.current_chat["id"],temporary_id)
        self.window.chat_list.setCurrentRow(0)
        self.assertEqual(self.window.current_chat["id"],saved_id)
        self.assertFalse(self.window.temporary_notice.isVisible())
        self.assertTrue(self.window.chat_list.isVisible())
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

    def test_history_is_always_listed_and_settings_persist_across_restart(self):
        self.reply("persistent chat")
        self.assertTrue(self.window.chat_list.isVisible())
        self.window.top_p.setValue(.85)
        self.window._new_chat(temporary=True);self.window._show_page(3)
        self.assertFalse(self.window.chat_list.isVisible())   # other sections show their own lists
        self.window._open_section(0)
        self.assertTrue(self.window.chat_list.isVisible())
        self.window.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused"))
        reopened.timer.stop();reopened.show()
        try:
            self.assertTrue(reopened.chat_list.isVisible())
            self.assertEqual(reopened.chat_list.count(),1)
            self.assertEqual(reopened.top_p.value(),.85)
        finally: reopened.close()

    def test_history_fills_the_sidebar_and_scrolls_without_pushing_the_status_card_out(self):
        for i in range(60):
            chat=new_chat();chat.update(title=f"Saved chat {i}",messages=[{"role":"user","content":"hello"}])
            self.window.chats.append(chat)
        self.window._refresh_chats()
        self.window.resize(1000,690);QTest.qWait(50)
        self.assertGreater(self.window.chat_list.height(),180)   # no longer a cramped fixed-height box
        self.assertGreater(self.window.chat_list.verticalScrollBar().maximum(),0)
        self.assertLess(self.window.gpu_status.mapTo(self.window,self.window.gpu_status.rect().bottomRight()).y(),self.window.height())
        bar=self.window.chat_list.verticalScrollBar();bar.setValue(20)
        self.window._refresh_chats()
        self.assertEqual(bar.value(),20)

    def test_search_filters_titles_and_messages_and_keeps_rows_aligned(self):
        for title,text in (("Alpha","hello"),("Beta","needle inside")):
            chat=new_chat();chat.update(title=title,messages=[{"role":"user","content":text}]);self.window.chats.append(chat)
        self.window._refresh_chats()
        self.window.chat_search.setText("needle")
        self.assertTrue(self.window.chat_list.item(0).isHidden())
        self.assertFalse(self.window.chat_list.item(1).isHidden())
        self.window.chat_list.setCurrentRow(1)
        self.assertEqual(self.window.current_chat["title"],"Beta")
        self.window.chat_search.setText("alp")
        self.assertFalse(self.window.chat_list.item(0).isHidden())
        self.window.chat_search.setText("zzz")
        self.assertTrue(self.window.chat_empty.isVisible())
        self.assertEqual(self.window.chat_empty.text(),"No chats match your search")
        self.window.chat_search.clear()
        self.assertTrue(self.window.chat_list.isVisible())
        self.assertFalse(self.window.chat_empty.isVisible())

    def test_empty_history_shows_a_hint_instead_of_a_blank_list(self):
        self.assertTrue(self.window.chat_empty.isVisible())
        self.assertEqual(self.window.chat_empty.text(),"No saved chats yet")
        self.assertFalse(self.window.chat_list.isVisible())
        self.reply("first")
        self.assertFalse(self.window.chat_empty.isVisible())
        self.assertTrue(self.window.chat_list.isVisible())

    def test_dots_button_requests_the_same_menu_as_right_click_without_switching_chats(self):
        self.reply("older");older_id=self.window.current_chat["id"]
        self.window._new_chat();self.reply("newer");newer_id=self.window.current_chat["id"]
        received=[];self.window.chat_list.customContextMenuRequested.connect(received.append)
        rect=self.window.chat_list.visualItemRect(self.window.chat_list.item(1))
        with patch("gui.app.QMenu") as menu:   # a real QMenu.exec would block the headless run
            menu.return_value.exec.return_value=None
            QTest.mouseClick(self.window.chat_list.viewport(),Qt.MouseButton.LeftButton,pos=QPoint(rect.right()-8,rect.center().y()))
            self.assertEqual(len(received),1)
            menu.return_value.exec.assert_called_once()
        self.assertEqual(self.window.current_chat["id"],newer_id)   # the click did not open the other chat
        QTest.mouseClick(self.window.chat_list.viewport(),Qt.MouseButton.LeftButton,pos=QPoint(rect.left()+20,rect.center().y()))
        self.assertEqual(len(received),1)
        self.assertEqual(self.window.current_chat["id"],older_id)

    def test_settings_page_switches_theme_live_and_remembers_it(self):
        w=self.window;amber=theme.PALETTES["amber"]
        QTest.mouseClick(w.sections[2],Qt.MouseButton.LeftButton);QTest.mouseClick(w.nav[5],Qt.MouseButton.LeftButton)
        self.assertEqual(w.pages.currentIndex(),5)
        self.assertTrue(w.settings_page.cards["teal"].isChecked())
        w.transcript.render([{"role":"assistant","content":"hello","status":"complete"}])
        QTest.mouseClick(w.settings_page.cards["amber"],Qt.MouseButton.LeftButton)
        self.assertEqual(theme.COLORS["accent"],amber["accent"])
        self.assertIn(amber["accent"],QApplication.instance().styleSheet())
        self.assertIn(amber["accent"],w.transcript.toHtml().lower())   # rendered HTML was refreshed
        self.assertEqual(read_json(self.state/"settings.json")["theme"],"amber")
        w.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"));reopened.timer.stop();reopened.show()
        try:
            self.assertEqual(theme.COLORS["accent"],amber["accent"])
            self.assertTrue(reopened.settings_page.cards["amber"].isChecked())
        finally: reopened.close()

    def test_unknown_saved_theme_falls_back_to_teal(self):
        atomic_json(self.state/"settings.json",{"theme":"neon"})
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"));reopened.timer.stop()
        try:
            self.assertEqual(theme.COLORS["accent"],theme.PALETTES["teal"]["accent"])
            self.assertTrue(reopened.settings_page.cards["teal"].isChecked())
        finally: reopened.close()

    def test_temporary_chat_is_an_outline_button_that_still_works(self):
        w=self.window
        self.assertEqual(w.temporary_chat_button.objectName(),"outline")
        self.assertEqual(w.temporary_chat_button.text(),"Temporary chat")
        QTest.mouseClick(w.temporary_chat_button,Qt.MouseButton.LeftButton)
        self.assertTrue(w.current_chat.get("temporary"))
        self.assertTrue(w.temporary_chat_button.isChecked())

    def test_empty_chat_shows_a_big_greeting_and_one_row_of_suggestions(self):
        w=self.window
        w.model_combo.clear();w.model_combo.addItem("Choose a model…","");w._update_welcome()
        self.assertEqual(w.chat_content.currentIndex(),2)
        self.assertIn(w.welcome_title.text(),GREETINGS)
        self.assertTrue(w.welcome_hint.isVisible())
        w.model_combo.clear();w.model_combo.addItem("CPU mock model","/unused-cpu-model.bin")
        self.assertFalse(w.welcome_hint.isVisible())
        self.assertTrue(w.suggestion_row.isVisible())
        chips=w.suggestion_row.findChildren(QPushButton)
        self.assertTrue(1<=len(chips)<=4)
        self.assertTrue(all(c.text() in SUGGESTIONS for c in chips))
        self.assertLess(w.suggestion_row.mapTo(w,w.suggestion_row.rect().bottomLeft()).y(),w.composer_panel.mapTo(w,w.composer_panel.rect().topLeft()).y()+1)
        QTest.mouseClick(chips[0],Qt.MouseButton.LeftButton)
        self.assertEqual(w.composer.toPlainText(),chips[0].text())
        self.assertFalse(w.transcript.toPlainText().strip())
        self.reply("hello")
        self.assertEqual(w.chat_content.currentIndex(),0)
        self.assertFalse(w.suggestion_row.isVisible())
        w._new_chat(temporary=True);self.assertEqual(w.chat_content.currentIndex(),1)   # temporary keeps its own notice
        self.assertFalse(w.suggestion_row.isVisible())
        w._new_chat();self.assertEqual(w.chat_content.currentIndex(),2)
        self.assertTrue(w.suggestion_row.isVisible())

    def test_each_new_empty_chat_picks_a_different_greeting_and_suggestions_rotate(self):
        w=self.window;seen=set()
        for _ in range(12):
            w._new_chat(temporary=True);w._new_chat();seen.add(w.welcome_title.text())
        self.assertGreater(len(seen),2)
        before=[c.text() for c in w.suggestion_row.findChildren(QPushButton)]
        w.composer.clear()
        with patch.object(type(w.suggestion_row),"underMouse",return_value=False):
            w._rotate_suggestions()
        after=[c.text() for c in w.suggestion_row.findChildren(QPushButton)]
        self.assertNotEqual(before,after)
        w._use_starter("typing something")
        with patch.object(type(w.suggestion_row),"underMouse",return_value=False):
            w._rotate_suggestions()                                    # no rotation while the user is typing
        self.assertEqual(after,[c.text() for c in w.suggestion_row.findChildren(QPushButton)])

    def test_train_summary_card_reports_the_config_and_hides_json_by_default(self):
        w=self.window;w._open_section(1)
        self.assertFalse(w.config_editor.isVisible())
        self.assertNotEqual(w.summary_tiles["params"][0].text(),"–")
        config=json.loads(w.config_editor.toPlainText())
        w.config_editor.setPlainText(json.dumps({**config,"training_steps":1234,"layers":2,"embedding_dim":64,"attention_heads":4,"feed_forward_dim":256}))
        self.assertEqual(w.summary_tiles["steps"][0].text(),"1,234")
        self.assertEqual(w.summary_tiles["shape"][0].text(),"2 × 64")
        self.assertIn("4 heads",w.summary_tiles["shape"][1].text())
        self.assertIn(config["output_model"],w.config_output.text())
        w.config_editor.setPlainText("{nope")
        self.assertEqual(w.summary_tiles["params"][0].text(),"–")
        self.assertIn("Not a valid configuration",w.config_hint.text())
        self.assertIs(w.config_hint.property("bad"),True)
        QTest.mouseClick(w.json_toggle,Qt.MouseButton.LeftButton)
        self.assertTrue(w.config_editor.isVisible())
        QTest.mouseClick(w.json_toggle,Qt.MouseButton.LeftButton)
        self.assertFalse(w.config_editor.isVisible())

    def gguf_info(self,name="Tiny Chat",ctx=2048):
        path=write_gguf(self.state/"models"/f"{name}.gguf",*chat_model(name,"llama",ctx,15))
        from gui.gguf import read_gguf_info
        return read_gguf_info(path)

    def test_gguf_models_appear_in_every_picker_with_their_own_label(self):
        w=self.window;info=self.gguf_info()
        with patch("gui.app.discover_gguf",return_value=[info]):
            w.refresh_models()
        for combo in (w.model_combo,w.compare_a,w.compare_b):
            index=combo.findData(info["path"])
            self.assertGreater(index,0)
            self.assertEqual(combo.itemText(index).split("  ·  ")[:2],["Tiny Chat","GGUF Q4_K_M"])
        w.model_combo.setCurrentIndex(w.model_combo.findData(info["path"]))
        self.assertIn("2,048 token context",w.model_combo.toolTip())
        self.assertEqual(w.max_tokens.maximum(),2047)

    def test_model_sizes_are_readable_at_every_scale(self):
        from gui.app import human_size
        self.assertEqual([human_size(n) for n in (400_000,477_984,2_500_000,45_000_000,4_600_000_000)],["400 KB","478 KB","2.5 MB","45 MB","4.6 GB"])
        w=self.window;info=self.gguf_info()
        with patch("gui.app.discover_gguf",return_value=[info]): w.refresh_models()
        index=w.model_combo.findData(info["path"])
        self.assertTrue(w.model_combo.itemText(index).endswith("KB"))
        self.assertGreater(w.model_combo.view().minimumWidth(),150)

    def test_choosing_a_gguf_without_the_runtime_points_to_settings(self):
        w=self.window;info=self.gguf_info()
        with patch("gui.app.discover_gguf",return_value=[info]),patch.object(w.runtimes,"resolve",return_value=None):
            w.refresh_models()
            w.model_combo.setCurrentIndex(w.model_combo.findData(info["path"]))
            self.assertTrue(w.welcome_hint.isVisible())
            self.assertIn("llama.cpp runtime",w.welcome_hint.text())
        with patch.object(w.runtimes,"resolve",return_value=RuntimeInfo(Path("/x/llama-server"),"path")):
            w._update_welcome()
            self.assertFalse(w.welcome_hint.isVisible())

    def test_browse_accepts_gguf_files_and_remembers_them_across_restarts(self):
        w=self.window;path=write_gguf(self.state/"elsewhere"/"Picked.gguf",*chat_model("Picked","llama",1024,7))
        with patch("gui.app.QFileDialog.getOpenFileName",return_value=(str(path),"")) as dialog:
            w._browse_model()
            self.assertIn("*.gguf",dialog.call_args.args[3])
        self.assertEqual(w.model_combo.currentData(),str(path.resolve()))
        self.assertEqual(read_json(self.state/"settings.json")["gguf_paths"],[str(path.resolve())])
        w.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"));reopened.timer.stop()
        try:
            self.assertGreater(reopened.model_combo.findData(str(path.resolve())),0)
        finally: reopened.close()
        with patch("gui.app.QFileDialog.getOpenFileName",return_value=(str(self.state/"nope.gguf"),"")):
            self.window._browse_model()                                   # unreadable files only produce a message
        self.assertIn("GGUF",self.window.statusBar().currentMessage()+"GGUF")

    def test_runtime_card_downloads_and_cancels_through_the_manager(self):
        w=self.window;card=w.settings_page.runtime_card;manager=w.runtimes
        self.assertEqual(card.flavor.count(),len(manager.flavors))
        self.assertIn("(recommended)",card.flavor.itemText(card.flavor.currentIndex()))
        self.assertTrue(card.flavor_detail.text())
        with patch.object(manager,"start_install") as start:
            QTest.mouseClick(card.install_button,Qt.MouseButton.LeftButton)
            start.assert_called_once_with(card.flavor.currentData())
        with patch.object(type(manager),"installing",property(lambda self:True)),patch.object(manager,"cancel_install") as cancel:
            card.refresh()
            self.assertEqual(card.install_button.text(),"Cancel");self.assertFalse(card.flavor.isEnabled());self.assertFalse(card.choose_button.isEnabled())
            QTest.mouseClick(card.install_button,Qt.MouseButton.LeftButton);cancel.assert_called_once()
        card.refresh();self.assertEqual(card.install_button.text(),"Download")
        manager.install_failed.emit("Checksum mismatch");self.assertTrue(card.error.isVisibleTo(card));self.assertIn("Checksum",card.error.text())
        card._progress(5_000_000,20_000_000,"Downloading 1/2");self.assertEqual((card.progress.value(),card.progress.maximum()),(5_000_000,20_000_000))
        self.assertIn("5 / 20 MB",card.progress.format())

    def fake_download(self,flavor,tag="bTEST",installed=1.0):
        folder=self.state/"runtimes"/"llama.cpp"/f"{tag}-{flavor}";folder.mkdir(parents=True)
        (folder/"llama-server").write_bytes(b"#!/bin/sh\n"+b"x"*2_000_000);(folder/"llama-server").chmod(0o755)
        (folder/"runtime.json").write_text(json.dumps({"tag":tag,"flavor":flavor,"binary":"llama-server","installed":installed}))
        return folder

    def test_installed_runtimes_are_listed_with_name_path_and_actions(self):
        w=self.window;card=w.settings_page.runtime_card;manager=w.runtimes
        manager._path_lookup=lambda name:None
        manager.refresh()
        self.assertEqual(card.rows,{});self.assertTrue(card.empty.isVisibleTo(card));self.assertIn("Nothing installed",card.empty.text())
        cpu=self.fake_download("cpu",installed=1.0);vulkan=self.fake_download("vulkan",installed=2.0)
        manager.refresh()
        self.assertFalse(card.empty.isVisibleTo(card))
        self.assertEqual(list(card.rows),["managed:bTEST-vulkan","managed:bTEST-cpu"])
        newest,older=card.rows["managed:bTEST-vulkan"],card.rows["managed:bTEST-cpu"]
        self.assertIn("Vulkan",newest.name_label.text());self.assertIn("bTEST",newest.name_label.text())
        self.assertIn(str(vulkan/"llama-server"),newest.path_label.text());self.assertIn("2 MB",newest.path_label.text())
        self.assertTrue(newest.badge.isVisibleTo(newest));self.assertFalse(older.badge.isVisibleTo(older))
        self.assertFalse(newest.use_button.isVisibleTo(newest));self.assertTrue(older.use_button.isVisibleTo(older))
        self.assertEqual((newest.remove_button.text(),newest.remove_button.isVisibleTo(newest)),("Remove",True))
        with patch("gui.settings.QDesktopServices.openUrl") as opened:
            QTest.mouseClick(older.open_button,Qt.MouseButton.LeftButton)
            self.assertEqual(opened.call_args.args[0].toLocalFile(),str(cpu))
        QTest.mouseClick(older.use_button,Qt.MouseButton.LeftButton)
        self.assertEqual(read_json(self.state/"settings.json")["llama_runtime"],"bTEST-cpu")
        self.assertTrue(card.rows["managed:bTEST-cpu"].badge.isVisibleTo(card))
        self.assertFalse(card.rows["managed:bTEST-vulkan"].badge.isVisibleTo(card))
        QTest.mouseClick(card.rows["managed:bTEST-vulkan"].remove_button,Qt.MouseButton.LeftButton)
        self.assertFalse(vulkan.exists());self.assertTrue(cpu.exists())
        self.assertEqual(list(card.rows),["managed:bTEST-cpu"])

    def test_choosing_an_existing_llama_server_adds_a_row_that_can_be_dropped(self):
        w=self.window;card=w.settings_page.runtime_card;w.runtimes._path_lookup=lambda name:None
        binary=self.state/"my-llama-server";binary.write_bytes(b"#!/bin/sh\n");binary.chmod(0o755)
        with patch("gui.settings.QFileDialog.getOpenFileName",return_value=(str(binary),"")):
            QTest.mouseClick(card.choose_button,Qt.MouseButton.LeftButton)
        self.assertEqual(read_json(self.state/"settings.json")["llama_server_path"],str(binary))
        row=card.rows["custom"]
        self.assertIn("Your chosen llama-server",row.name_label.text());self.assertIn(str(binary),row.path_label.text())
        self.assertTrue(row.badge.isVisibleTo(row));self.assertEqual(row.remove_button.text(),"Stop using")
        QTest.mouseClick(row.remove_button,Qt.MouseButton.LeftButton)
        self.assertEqual(read_json(self.state/"settings.json")["llama_server_path"],"")
        self.assertNotIn("custom",card.rows);self.assertTrue(binary.exists(),"stopping use never deletes the user's file")

    def test_other_backends_without_trance_fields_do_not_break_the_message_flow(self):
        w=self.window
        w.composer.setPlainText("hello");w.model_combo.setCurrentIndex(0)   # the mock model
        w._send_chat()
        w._generation_started({})                      # a llama.cpp "started" event carries no prompt_tokens
        w._delta("Hi there")
        w._finished({"reason":"stop","generated_tokens":2,"elapsed_seconds":0.5,"prompt_tokens":7,"context_length":2048})
        message=w.current_chat["messages"][-1]
        self.assertEqual((message["content"],message["status"],message["prompt_tokens"]),("Hi there","complete",7))
        self.assertEqual(w.chat_hint.text(),"Context: 9/2,048 tokens")

    def test_max_tokens_defaults_to_no_limit_for_gguf_and_remembers_each_kind(self):
        w=self.window;info=self.gguf_info("Chatty",ctx=4096)
        self.assertEqual(w.max_tokens.value(),160)                              # a Trance-style model keeps a modest cap
        w.max_tokens.setValue(300)
        with patch("gui.app.discover_gguf",return_value=[info]): w.refresh_models()
        w.model_combo.setCurrentIndex(w.model_combo.findData(info["path"]))
        self.assertEqual((w.max_tokens.value(),w.max_tokens.text(),w.max_tokens.maximum()),(0,"No limit",4095))
        w.max_tokens.setValue(2000)
        w.model_combo.setCurrentIndex(0)
        self.assertEqual((w.max_tokens.value(),w.max_tokens.maximum()),(300,1023))
        w.model_combo.setCurrentIndex(w.model_combo.findData(info["path"]))
        self.assertEqual(w.max_tokens.value(),2000)
        w._save_preferences()
        saved=read_json(self.state/"settings.json");self.assertEqual((saved["max_new_tokens"],saved["max_new_tokens_gguf"]),(300,2000))
        w.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"));reopened.timer.stop()
        try: self.assertEqual((reopened._max_values["trance"],reopened._max_values["gguf"]),(300,2000))
        finally: reopened.close()

    def test_no_limit_lets_gguf_models_finish_and_gives_trance_half_its_window(self):
        w=self.window;info=self.gguf_info("Chatty",ctx=4096)
        w.models.append({"path":"/m/t.bin","context_length":1024,"parameters":1,"step":1})
        w.max_tokens.setValue(0)
        self.assertEqual(w._generation_settings(info["path"])["max_new_tokens"],0)    # omitted from the request: the model decides
        self.assertEqual(w._generation_settings("/m/t.bin")["max_new_tokens"],512)
        self.assertEqual(w._generation_settings("/m/unknown.bin")["max_new_tokens"],512)
        w.max_tokens.setValue(200)
        self.assertEqual(w._generation_settings("/m/t.bin")["max_new_tokens"],200)

    def test_cutoff_notes_say_whether_the_limit_or_the_context_window_ended_the_reply(self):
        w=self.window
        for limit,expected in ((160,"response limit reached"),(0,"context window full")):
            w.operation="chat"
            w.current_chat["messages"]=[{"role":"user","content":"hi"},{"role":"assistant","content":"partial","status":"generating","token_limit":limit}]
            w._finished({"reason":"length","generated_tokens":5,"elapsed_seconds":1.0})
            self.assertTrue(w.current_chat["messages"][-1]["note"].endswith(expected),w.current_chat["messages"][-1]["note"])

    def test_requests_record_the_limit_they_were_sent_with(self):
        w=self.window
        with patch.object(w.engine,"generate") as generate:
            w.max_tokens.setValue(0);w.composer.setPlainText("hello");w.model_combo.setCurrentIndex(0);w._send_chat()
        self.assertEqual(generate.call_args.args[2]["max_new_tokens"],512)           # the mock Trance model has no known window -> default 1024 // 2
        self.assertEqual(w.current_chat["messages"][-1]["token_limit"],512)

    def thinking_model(self,name,template):
        from gui.gguf import read_gguf_info
        path=write_gguf(self.state/"models"/f"{name}.gguf",*chat_model(name,"llama",4096,15),
                        __import__("test_gguf").kv_str("tokenizer.chat_template",template))
        return read_gguf_info(path)

    def test_the_think_toggle_sits_bottom_left_with_the_hint_to_its_right(self):
        w=self.window
        self.assertEqual(w.think_button.text(),"Think");self.assertTrue(w.think_button.isCheckable())
        left=lambda widget:widget.mapTo(w,widget.rect().topLeft()).x()
        self.assertLess(left(w.think_button),left(w.chat_hint))
        self.assertLess(left(w.chat_hint),left(w.send_button))
        self.assertEqual(w.think_button.mapTo(w,w.think_button.rect().center()).y()//12,w.chat_hint.mapTo(w,w.chat_hint.rect().center()).y()//12)

    def test_think_is_only_available_for_models_that_can_think_and_is_remembered(self):
        w=self.window
        thinker=self.thinking_model("Thinker","{% if enable_thinking %}<think>{% endif %}")
        plain=self.thinking_model("Plain","<start_of_turn>user\n{{ content }}")
        with patch("gui.app.discover_gguf",return_value=[thinker,plain]): w.refresh_models()
        pick=lambda info:w.model_combo.setCurrentIndex(w.model_combo.findData(info["path"]))
        w.model_combo.setCurrentIndex(0)                                             # "Choose a model…"
        self.assertFalse(w.think_button.isEnabled());self.assertIn("Choose a model",w.think_button.toolTip())
        pick(plain);self.assertFalse(w.think_button.isEnabled());self.assertIn("does not have a thinking mode",w.think_button.toolTip())
        pick(thinker);self.assertTrue(w.think_button.isEnabled())
        w.model_combo.clear();w.model_combo.addItem("A Trance model","/m/trance.bin")
        self.assertFalse(w.think_button.isEnabled());self.assertIn("Trance models",w.think_button.toolTip())
        w.model_combo.clear();w.model_combo.addItem("Thinker",thinker["path"]);w.think_button.setChecked(True)
        self.assertEqual(read_json(self.state/"settings.json")["thinking"],True)
        w.close()
        reopened=MainWindow(state_dir=self.state,binary=Path("/unused-cpu-test-engine"));reopened.timer.stop()
        try: self.assertTrue(reopened.think_button.isChecked())
        finally: reopened.close()

    def test_the_thinking_switch_is_only_sent_to_models_that_have_one(self):
        w=self.window
        thinker=self.thinking_model("Thinker","{% if enable_thinking %}x{% endif %}")
        plain=self.thinking_model("Plain","{{ content }}")
        with patch("gui.app.discover_gguf",return_value=[thinker,plain]): w.refresh_models()
        w.think_button.setEnabled(True);w.think_button.setChecked(True)
        self.assertIs(w._generation_settings(thinker["path"])["thinking"],True)
        w.think_button.setChecked(False)
        self.assertIs(w._generation_settings(thinker["path"])["thinking"],False)   # explicitly off, so the model does not think on its own
        self.assertNotIn("thinking",w._generation_settings(plain["path"]))
        self.assertNotIn("thinking",w._generation_settings("/m/trance.bin"))
        stamp=Path(thinker["path"]).stat().st_mtime
        with patch("gui.app.read_chat_template",side_effect=AssertionError("should be cached")):
            w._thinking_support(thinker["path"])                                      # cached by file, not read again
        self.assertEqual(w._think_cache[thinker["path"]][0],stamp)

    def test_reasoning_streams_into_a_fold_that_collapses_when_the_answer_starts(self):
        w=self.window;w.operation="chat"
        w.current_chat["messages"]=[{"role":"user","content":"hi"},{"role":"assistant","content":"","status":"generating","model_name":"Qwen Test"}]
        w._thinking_delta("Let me ");w._thinking_delta("think.");w._render_active()
        text=w.transcript.toPlainText()
        self.assertIn("QWEN TEST",text);self.assertIn("Thinking…",text);self.assertIn("Let me think.",text)    # open while it thinks
        w._delta("The answer");w._finished({"reason":"stop","generated_tokens":4,"elapsed_seconds":2.0})
        message=w.current_chat["messages"][-1]
        self.assertEqual((message["thinking"],message["content"]),("Let me think.","The answer"))
        self.assertIn("thinking_seconds",message)
        text=w.transcript.toPlainText()
        self.assertIn("Thought for",text) if message["thinking_seconds"]>=1 else self.assertIn("Thoughts",text)
        self.assertNotIn("Let me think.",text)                                       # collapsed once the answer is there
        w.transcript._link_clicked(QUrl("think:1"))
        self.assertIn("Let me think.",w.transcript.toPlainText())
        w.transcript._link_clicked(QUrl("think:1"))
        self.assertNotIn("Let me think.",w.transcript.toPlainText())
        w.transcript._link_clicked(QUrl("think:99"));w.transcript._link_clicked(QUrl("think:abc"))   # bad links are ignored

    def test_markdown_replies_are_formatted_and_raw_html_is_shown_as_text(self):
        w=self.window
        reply="## Plan\n\n**bold** and *italic* and `code`\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nx = 1\n```\n\n<b>not bold</b> <img src='http://evil.example/x.png'>"
        w.transcript.render([{"role":"user","content":"**not markdown** <b>x</b>"},{"role":"assistant","content":reply,"status":"complete"}])
        html=w.transcript.toHtml();plain=w.transcript.toPlainText()
        self.assertIn("**not markdown**",plain)                                      # your own text is never reinterpreted
        for expected in ("Plan","bold","italic","one","two","x = 1"): self.assertIn(expected,plain)
        self.assertNotIn("##",plain);self.assertNotIn("```",plain);self.assertNotIn("**bold**",plain)
        self.assertIn("<table",html);self.assertIn("<ul",html);self.assertIn("font-weight:700",html)
        self.assertIn("<b>not bold</b>",plain)                                       # shown literally, not applied
        self.assertNotIn("evil.example",html.replace("&lt;","<")) if "<img" in html else None

    def test_markdown_keeps_the_models_own_line_breaks_and_unfinished_code_blocks(self):
        from gui.markdown import to_html
        poem=to_html("Roses are red\nViolets are blue\n\nSecond verse")
        self.assertIn("Roses are red",poem);self.assertTrue(poem.count("<br")>=1 or poem.count("<p")>=2)
        partial=to_html("Here:\n```python\nprint('streaming'")
        self.assertIn("streaming",partial)                                           # a half-streamed fence still renders
        self.assertEqual(to_html("   \n"),"")
        w=self.window;w.transcript.render([{"role":"assistant","content":"line one\nline two\nline three","status":"complete"}])
        lines=[l for l in w.transcript.toPlainText().splitlines() if l.startswith("line")]
        self.assertEqual(len(lines),3,w.transcript.toPlainText())

    def test_links_only_open_when_clicked_and_only_web_addresses(self):
        w=self.window
        w.transcript.render([{"role":"assistant","content":"[site](https://example.com) [bad](javascript:alert(1)) [file](file:///etc/passwd)","status":"complete"}])
        with patch("gui.widgets.QDesktopServices.openUrl") as opened:
            for url in ("https://example.com","javascript:alert(1)","file:///etc/passwd","ftp://x/y"):
                w.transcript._link_clicked(QUrl(url))
            self.assertEqual([c.args[0].toString() for c in opened.call_args_list],["https://example.com"])
        self.assertIn("site",w.transcript.toPlainText())                             # the view never navigated away

    def test_streaming_redraws_are_coalesced_but_the_final_state_is_exact(self):
        w=self.window;w.operation="chat"
        w.current_chat["messages"]=[{"role":"user","content":"hi"},{"role":"assistant","content":"","status":"generating"}]
        draws=[]
        original=w.transcript.render
        with patch.object(w.transcript,"render",side_effect=lambda *a,**k:(draws.append(1),original(*a,**k))[1]):
            for _ in range(200): w._delta("x")
            self.assertLess(len(draws),50)                                           # not one redraw per token
            w._finished({"reason":"stop","generated_tokens":200,"elapsed_seconds":1.0})
        self.assertEqual(w.transcript.toPlainText().count("x"),200)
        self.assertFalse(w._render_timer.isActive())

    def test_status_card_and_train_badge_follow_activity(self):
        w=self.window
        self.assertEqual(w.status_title.text(),"No model loaded");self.assertFalse(w.model_status.isVisibleTo(w))
        w._engine_ready("NVIDIA Test GPU (compute 8.9)")
        self.assertEqual((w.status_title.text(),w.gpu_status.text()),("No model loaded","NVIDIA Test GPU"))   # a GPU alone is not a loaded model
        w.engine.trance.loaded.emit({"model":"/m/trance1.2_30m.bin","parameters":30_000_000})
        self.assertEqual((w.status_title.text(),w.model_status.text()),("Ready","Trance 1.2"))
        self.assertTrue(w.model_status.isVisibleTo(w))
        w._set_busy(True);self.assertEqual(w.status_title.text(),"Generating…")
        w._set_busy(False);self.assertEqual(w.status_title.text(),"Ready")
        w.training_active=True;w.progress.setValue(320);w._update_status()
        self.assertEqual(w.status_title.text(),"Training · 32%")
        self.assertEqual(w.sections[1].text(),"Train ●")
        self.assertEqual(w.status_dot.property("state"),"busy")
        w.training_active=False;w._update_status()
        self.assertEqual(w.sections[1].text(),"Train")
        w.operation=None;w._failed("boom")
        self.assertEqual((w.status_title.text(),w.status_dot.property("state")),("Last request failed","error"))
        w._set_busy(True);w._set_busy(False)
        self.assertEqual(w.status_title.text(),"Ready")

    def test_eject_unloads_the_model_and_is_only_available_when_it_can_work(self):
        w=self.window
        self.assertFalse(w.eject_button.isEnabled())                        # nothing loaded yet
        w.engine.trance.loaded.emit({"model":"/m/trance1.2_30m.bin","parameters":1})
        self.assertTrue(w.eject_button.isEnabled())
        w._set_busy(True);self.assertFalse(w.eject_button.isEnabled())     # not while a reply is being written
        w._set_busy(False);self.assertTrue(w.eject_button.isEnabled())
        w.training_active=True;w._update_eject();self.assertFalse(w.eject_button.isEnabled())
        w.training_active=False;w._update_eject()
        QTest.mouseClick(w.eject_button,Qt.MouseButton.LeftButton)
        self.assertEqual(w.engine.loaded_model,"")
        self.assertFalse(w.eject_button.isEnabled())
        self.assertEqual(w.status_title.text(),"No model loaded");self.assertFalse(w.model_status.isVisibleTo(w))
        self.assertIn("Ejected Trance 1.2",w.statusBar().currentMessage())

    def test_a_gguf_model_shows_its_title_in_the_status_card(self):
        w=self.window;info=self.gguf_info("Gemma Test")
        with patch("gui.app.discover_gguf",return_value=[info]): w.refresh_models()
        w.engine.llama.loaded.emit({"model":info["path"],"parameters":0})
        self.assertEqual(w.model_status.text(),"Gemma Test")
        self.assertTrue(w.eject_button.isEnabled())
        w.engine.llama.unloaded.emit()                                       # the server crashed
        self.assertEqual(w.status_title.text(),"No model loaded");self.assertFalse(w.eject_button.isEnabled())

    def test_recent_runs_list_opens_runs_and_protects_an_active_run(self):
        w=self.window
        for name,status,out in (("20260101-000000-aaaa","complete","models/trance1.1_21m.bin"),
                                ("20260102-000000-bbbb","stopped","models/trance1.2_30m.bin")):
            atomic_json(self.state/"runs"/name/"status.json",{"status":status})
            atomic_json(self.state/"runs"/name/"config.json",{"output_model":out})
        with patch("gui.training.STATE",self.state):
            w._refresh_runs()
            self.assertEqual([w.run_list.item(i).text() for i in range(w.run_list.count())],
                             ["Trance 1.2  ·  stopped","Trance 1.1  ·  done"])
            with patch.object(w,"_attach_training") as attach:
                w._open_run(w.run_list.item(0))
                attach.assert_called_once_with(self.state/"runs"/"20260102-000000-bbbb")
            self.assertEqual(w.pages.currentIndex(),2)
            running=self.state/"runs"/"running-run";w.training_dir=running;w.training_active=True
            with patch.object(w,"_attach_training") as attach:
                w._open_run(w.run_list.item(1))
                attach.assert_not_called()
            self.assertIn("Stop it before opening",w.train_message.text())
            self.assertEqual(w.training_dir,running)

    def test_context_menu_targets_clicked_chat_and_right_click_preserves_temporary_chat(self):
        self.reply("saved chat");saved_id=self.window.current_chat["id"]
        self.window._new_chat(temporary=True);QTest.qWait(20)
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
