"""CPU-only checks; do not start inference or alter a running training job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gui import runtime


class PackagingRuntimeTests(unittest.TestCase):
    def test_source_worker_keeps_legacy_module_launch(self):
        with patch.object(runtime,"FROZEN",False):
            self.assertEqual(runtime.supervisor_arguments(Path('/tmp/example-run')),
                             ['-m','gui.training_worker','/tmp/example-run'])

    def test_frozen_worker_has_explicit_project_and_worker_mode(self):
        with patch.object(runtime,"FROZEN",True):
            args=runtime.supervisor_arguments(Path('/tmp/example-run'))
            self.assertEqual(args[:2],['--project',str(runtime.PROJECT)])
            self.assertEqual(args[2:],['--training-worker','/tmp/example-run'])

    def test_appimage_worker_gets_an_independent_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'Trance.AppImage';file.touch()
            with patch.object(runtime,"FROZEN",True),patch.dict(os.environ,{'APPIMAGE':str(file)}):
                self.assertEqual(runtime.supervisor_program(),str(file))

    def test_explicit_workspace_resolves_before_storage_import(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-m','gui','--project',tmp,'--diagnostics'],cwd=root,capture_output=True,text=True,check=True)
            value=json.loads(result.stdout)
            self.assertEqual(value['project'],str(Path(tmp).resolve()))
            self.assertFalse(value['frozen'])


if __name__=='__main__':unittest.main()
