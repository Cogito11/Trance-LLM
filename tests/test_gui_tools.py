"""CPU-only config creation and offline documentation workflow tests."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from gui.app import MainWindow
from gui.config_builder import ConfigBuilder,save_new_config
from gui.documentation import Documentation,load_articles
from gui.storage import read_json
from gui.widgets import STYLE


class ConfigBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);cls.app.setStyleSheet(STYLE)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="trance-config-test-");self.root=Path(self.temp.name)
        data=self.root/"data/trance12";data.mkdir(parents=True)
        for name in ("train.txt","validation.txt"):(data/name).write_text("<user>\nHi\n<assistant>\nHello\n<eot>\n")
        self.paths=patch("gui.storage.ROOT",self.root);self.paths.start()
        self.builder=ConfigBuilder(root=self.root)

    def tearDown(self):
        self.builder.close();self.paths.stop();self.temp.cleanup()

    def test_save_roundtrip_does_not_replace_config_or_model(self):
        config=self.builder.config();self.builder.save()
        path=self.builder.saved_path
        self.assertEqual(read_json(path),config)
        self.assertFalse((self.root/config["output_model"]).exists())
        before=path.read_bytes()
        self.builder.fields["training_steps"].setValue(1234);self.builder.save()
        self.assertIn("already exists",self.builder.message.text())
        self.assertEqual(path.read_bytes(),before)

    def test_invalid_heads_missing_files_and_existing_output_are_rejected(self):
        self.builder.fields["embedding_dim"].setValue(129)
        self.assertFalse(self.builder.save_button.isEnabled())
        self.builder.fields["embedding_dim"].setValue(128)
        self.builder.fields["train_data"].setText("missing.txt");self.builder.save()
        self.assertIn("Missing",self.builder.message.text());self.assertIsNone(self.builder.saved_path)
        self.builder.fields["train_data"].setText("data/trance12/train.txt")
        output=self.root/"models/trance-experiment.bin";output.parent.mkdir();output.write_bytes(b"preserve")
        self.builder.save();self.assertIn("already exists",self.builder.message.text())
        self.assertEqual(output.read_bytes(),b"preserve")

    def test_template_rejects_unsupported_or_clamped_fields(self):
        original=self.builder.config()
        with self.assertRaises(ValueError):self.builder.set_config({**original,"dropout":.1})
        with self.assertRaises(ValueError):self.builder.set_config({**original,"layers":1000})
        self.assertEqual(self.builder.config(),original)

    def test_filename_cannot_escape_configs_and_name_suggests_output(self):
        with self.assertRaises(ValueError):save_new_config(self.root,"../escape",self.builder.config())
        self.builder.name.setText("trance1.3")
        self.assertEqual(self.builder.fields["output_model"].text(),"models/trance1.3.bin")
        self.builder.fields["output_model"].setText("models/custom.bin")
        self.builder.name.setText("another-name")
        self.assertEqual(self.builder.fields["output_model"].text(),"models/custom.bin")

    def test_save_and_open_selects_train_without_launching_any_engine(self):
        with patch("gui.app.ROOT",self.root),patch("gui.app.latest_run",return_value=None),patch("gui.app.launch_training") as launch:
            window=MainWindow(state_dir=self.root/"gui-state",binary=Path("/unused-engine"));window.timer.stop()
            try:
                window._show_page(3);window.config_builder.save()
                self.assertEqual(window.pages.currentIndex(),3)
                path=window.config_builder.saved_path;self.assertIsNotNone(path)
                window.config_builder.use_button.click()
                self.assertEqual(window.pages.currentIndex(),2)
                self.assertEqual(window.config_combo.currentData(),str(path))
                self.assertEqual(json.loads(window.config_editor.toPlainText()),read_json(path))
                launch.assert_not_called()
            finally:window.close()


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_all_articles_and_related_links_exist(self):
        articles=load_articles();ids={a["id"] for a in articles}
        self.assertGreaterEqual(len(articles),14)
        for article in articles:
            self.assertTrue(article["body"].startswith("# "))
            for target in re.findall(r"article://([a-z-]+)",article["body"]):self.assertIn(target,ids)

    def test_search_and_internal_link_navigation_are_offline(self):
        widget=Documentation()
        try:
            widget.search.setText("overfitting")
            self.assertIn("overfitting",widget.reader.toPlainText().casefold())
            self.assertEqual(widget.current_id,"losses")
            with patch("gui.documentation.QDesktopServices.openUrl") as external:
                widget._link(QUrl("article://attention"));external.assert_not_called()
                self.assertEqual(widget.current_id,"attention")
            widget.search.setText("no-such-concept-zzzzz")
            self.assertIn("No articles match",widget.reader.toPlainText())
            widget.search.clear();self.assertTrue(widget.current_id)
        finally:widget.close()


if __name__=="__main__":unittest.main()
