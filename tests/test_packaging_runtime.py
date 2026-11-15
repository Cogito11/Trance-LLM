"""CPU-only checks; do not start inference or alter a running training job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import importlib.util

from gui import bootstrap, runtime


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



class ReleaseLayoutTests(unittest.TestCase):
    def test_appimage_uses_the_folders_beside_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); image=folder/"TranceLLM.AppImage"; image.touch()
            with patch.dict(os.environ,{"APPIMAGE":str(image)}):
                self.assertIsNone(bootstrap.appimage_project())            # no configs/ or models/ beside it yet
                (folder/"configs").mkdir()
                self.assertIsNone(bootstrap.appimage_project())            # both are needed
                (folder/"models").mkdir()
                self.assertEqual(bootstrap.appimage_project(),folder.resolve())
            with patch.dict(os.environ,{},clear=False):
                os.environ.pop("APPIMAGE",None)
                self.assertIsNone(bootstrap.appimage_project())            # only set when running from an AppImage

    def test_appimage_in_the_dist_folder_does_not_hijack_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist=Path(tmp)/"dist"; dist.mkdir(); image=dist/"TranceLLM.AppImage"; image.touch()
            with patch.dict(os.environ,{"APPIMAGE":str(image)}):
                self.assertIsNone(bootstrap.appimage_project())

    @staticmethod
    def build_script():
        spec=importlib.util.spec_from_file_location("build_appimage_under_test",Path(runtime.PROJECT)/"tools"/"build_appimage.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

    def test_cublas_is_found_where_the_cuda_toolkit_or_an_override_puts_it(self):
        script=self.build_script()
        with tempfile.TemporaryDirectory() as tmp:
            override=Path(tmp)/"override"; toolkit=Path(tmp)/"cuda-12.6"
            (override).mkdir(); (toolkit/"lib64").mkdir(parents=True)
            (toolkit/"lib64"/"libcublas.so.12").write_text("toolkit"); (override/"libcublas.so.12").write_text("override")
            env={"CUDA_PATH":str(toolkit)}
            self.assertEqual(script.find_cublas("libcublas.so.12",env),toolkit/"lib64"/"libcublas.so.12")
            env["TRANCE_CUBLAS_DIR"]=str(override)
            self.assertEqual(script.find_cublas("libcublas.so.12",env),override/"libcublas.so.12")     # an explicit choice wins
            self.assertIsNone(script.find_cublas("libdoesnotexist.so.1",{"CUDA_PATH":str(toolkit)}))
            self.assertEqual(script.cublas_search_dirs({})[-1],Path("/lib/x86_64-linux-gnu"))          # the old Ubuntu location is still a fallback

    def test_cublas_license_text_is_collected_from_packages_or_the_toolkit(self):
        script=self.build_script()
        with tempfile.TemporaryDirectory() as tmp:
            docs=Path(tmp)/"doc"; (docs/"libcublas-12-6").mkdir(parents=True); (docs/"libcublas-12-6"/"copyright").write_text("c")
            toolkit=Path(tmp)/"cuda"; toolkit.mkdir(); (toolkit/"EULA.txt").write_text("eula")
            found=dict(script.find_cublas_licenses({"TRANCE_DOC_ROOT":str(docs),"CUDA_PATH":str(toolkit)}))
            self.assertEqual(sorted(found),["cuda-EULA.txt","libcublas-12-6-copyright.txt"])
            self.assertEqual(script.find_cublas_licenses({"TRANCE_DOC_ROOT":str(Path(tmp)/"none")}) if not Path("/usr/local/cuda/EULA.txt").exists() else [],[])


if __name__=='__main__':unittest.main()
