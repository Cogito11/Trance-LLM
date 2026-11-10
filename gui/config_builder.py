"""Guided configuration creation; saving never starts a training process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QLineEdit,
    QComboBox,QPushButton,QSpinBox,QDoubleSpinBox,QTabWidget,QPlainTextEdit,QFileDialog,QScrollArea)

from . import storage

INTEGER_FIELDS = {
    "vocab_size":("Vocabulary size",259,1024,"Number of byte and BPE tokens. Trance supports 259–1024."),
    "context_length":("Context length",2,1024,"Tokens in each training window. Longer contexts use more memory."),
    "embedding_dim":("Model width",1,4096,"Numbers in each token vector. Must be divisible by the attention heads."),
    "layers":("Transformer layers",1,64,"Number of attention and feed-forward blocks."),
    "attention_heads":("Attention heads",1,256,"Parallel attention groups. Must divide the model width."),
    "feed_forward_dim":("Feed-forward width",1,16384,"Hidden width inside each block's feed-forward network."),
    "batch_size":("Batch size",1,64,"Sequences per optimizer step. Larger batches require more GPU memory."),
    "training_steps":("Training steps",1,2147483647,"Optimizer updates. For resume, this is the additional step budget."),
    "eval_interval":("Evaluate every",0,2147483647,"Steps between validation checks. 0 disables periodic checks."),
    "checkpoint_interval":("Checkpoint every",0,2147483647,"Steps between saved optimizer checkpoints. 0 disables periodic saves."),
    "seed":("Random seed",0,2147483647,"Controls initialization and training-data sampling."),
}
PATH_FIELDS = {
    "train_data":("Training data","Conversation-format UTF-8 files. Multiple paths are comma-separated."),
    "validation_data":("Validation data","Held-out conversation files, or leave blank to omit validation."),
    "output_model":("Output model","A new model filename, typically models/name.bin. Existing models are protected."),
    "resume_model":("Resume checkpoint","Optional optimizer checkpoint with matching architecture. Leave blank for fresh training."),
}


def parameter_count(config):
    d=config["embedding_dim"];f=config["feed_forward_dim"];v=config["vocab_size"]
    return 2*v*d+config["context_length"]*d+2*d+v+config["layers"]*(4*d*d+2*d*f+9*d+f)


def save_new_config(root: Path, name: str, config: dict) -> Path:
    name=name.strip()
    if name.endswith(".json"):name=name[:-5]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",name):
        raise ValueError("Use a config name of 1–64 letters, numbers, dots, underscores or hyphens")
    storage.validate_config(config)
    target=root/"configs"/(name+".json");target.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=target.parent,delete=False) as f:
            temporary=Path(f.name);json.dump(config,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        # Publish a complete file without replacing any existing config,
        # including one created between validation and publication.
        os.link(temporary,target)
    except FileExistsError:
        raise ValueError("That config already exists. Choose a new name to preserve it") from None
    finally:
        if temporary:temporary.unlink(missing_ok=True)
    return target


class ConfigBuilder(QWidget):
    saved = Signal(str)
    open_training = Signal(str)

    def __init__(self,parent=None,*,root=None):
        super().__init__(parent);self.root=Path(root or storage.ROOT);self.fields={};self._loading=False;self.saved_path=None
        layout=QVBoxLayout(self);layout.setContentsMargins(30,28,30,24);layout.setSpacing(16)
        title=QLabel("Create a config");title.setObjectName("title");layout.addWidget(title)
        row=QHBoxLayout();self.preset=QComboBox();self.preset.addItems(["Small experiment","Tiny learning model","Trance 1.2 architecture"])
        row.addWidget(self.preset,1);load=QPushButton("Load template…");load.clicked.connect(self._load_template);row.addWidget(load);layout.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel("Config name"));self.name=QLineEdit("trance-experiment");row.addWidget(self.name,1);layout.addLayout(row)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        groups=(
            ("Model",("vocab_size","context_length","embedding_dim","layers","attention_heads","feed_forward_dim")),
            ("Training",("batch_size","training_steps","learning_rate","gradient_clip","eval_interval","checkpoint_interval","seed")),
            ("Data & output",tuple(PATH_FIELDS)),
        )
        for title,keys in groups:
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setObjectName("formScroll")
            page=QWidget();form=QFormLayout(page);form.setContentsMargins(20,20,20,20);form.setVerticalSpacing(18)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key in keys:
                if key in INTEGER_FIELDS:
                    text,lo,hi,help_text=INTEGER_FIELDS[key];field=QSpinBox();field.setRange(lo,hi);field.valueChanged.connect(self._changed)
                    form.addRow(text,field)
                elif key in PATH_FIELDS:
                    text,help_text=PATH_FIELDS[key];field=QLineEdit();field.textChanged.connect(self._changed)
                    cell=QWidget();row=QHBoxLayout(cell);row.setContentsMargins(0,0,0,0);row.addWidget(field,1)
                    browse=QPushButton("Browse…");browse.clicked.connect(lambda checked=False,k=key:self._browse(k));row.addWidget(browse);form.addRow(text,cell)
                else:
                    field=QDoubleSpinBox();field.setDecimals(8);field.setRange(.00000001,1 if key=="learning_rate" else 10000)
                    field.setSingleStep(.00001 if key=="learning_rate" else .1);field.valueChanged.connect(self._changed)
                    text="Learning rate" if key=="learning_rate" else "Gradient clip"
                    help_text="Size of each Adam update. Lower is slower but can be more stable." if key=="learning_rate" else "Limits the total gradient magnitude before Adam updates."
                    form.addRow(text,field)
                field.setToolTip(help_text);field.setAccessibleName(text);self.fields[key]=field
                if not isinstance(field,QLineEdit):field.setMaximumWidth(280)
            scroll.setWidget(page);self.tabs.addTab(scroll,title)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview.setFont(QFont("DejaVu Sans Mono",10));self.tabs.addTab(self.preview,"JSON preview")
        self.summary=QLabel();self.summary.setObjectName("eyebrow");self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.message=QLabel("Saving a config does not start training.");self.message.setObjectName("muted");self.message.setWordWrap(True);layout.addWidget(self.message)
        row=QHBoxLayout();row.addStretch();self.use_button=QPushButton("Open in Train");self.use_button.setEnabled(False)
        self.use_button.clicked.connect(lambda:self.open_training.emit(str(self.saved_path)))
        row.addWidget(self.use_button);self.save_button=QPushButton("Save config");self.save_button.setObjectName("primary");self.save_button.clicked.connect(self.save);row.addWidget(self.save_button);layout.addLayout(row)
        self._auto_output="models/trance-experiment.bin"
        self.name.textChanged.connect(self._name_changed);self.preset.currentIndexChanged.connect(self._preset_changed);self._preset_changed(0)

    def config(self):
        config={}
        for key,field in self.fields.items():
            value=field.text().strip() if isinstance(field,QLineEdit) else field.value()
            if key=="resume_model" and not value:continue
            config[key]=value
        return config

    def set_config(self,config):
        unsupported=set(config)-storage.CONFIG_KEYS
        if unsupported:raise ValueError("Unsupported settings: "+", ".join(sorted(unsupported)))
        # Validate structures before putting anything into bounded widgets,
        # so loading cannot silently clamp architecture or numeric values.
        storage.validate_config(config,check_output=False,check_files=False)
        for key,field in self.fields.items():
            value=config.get(key,"")
            if not isinstance(field,QLineEdit) and not field.minimum()<=value<=field.maximum():
                raise ValueError(f"{key} is outside the config builder's supported range")
        self._loading=True
        try:
            for key,field in self.fields.items():
                value=config.get(key,"")
                if isinstance(field,QLineEdit):field.setText(value)
                else:field.setValue(value)
        finally:self._loading=False
        self._changed()

    def _preset_changed(self,index):
        config={"vocab_size":512,"context_length":256,"embedding_dim":128,"layers":4,"attention_heads":4,"feed_forward_dim":512,
            "batch_size":2,"training_steps":20000,"eval_interval":500,"checkpoint_interval":2000,"learning_rate":.0002,"gradient_clip":1.,"seed":4004,
            "train_data":"data/trance12/train.txt","validation_data":"data/trance12/validation.txt","output_model":"models/"+(self.name.text().strip() or "trance-experiment")+".bin"}
        if index==1:
            config.update(vocab_size=259,context_length=64,embedding_dim=48,layers=2,attention_heads=4,feed_forward_dim=192,training_steps=1000,eval_interval=100,checkpoint_interval=500,learning_rate=.001)
        elif index==2:
            config.update(vocab_size=1024,context_length=1024,embedding_dim=448,layers=12,attention_heads=8,feed_forward_dim=1792,batch_size=4,training_steps=50000,eval_interval=1000,checkpoint_interval=5000)
        self.set_config(config)

    def _name_changed(self):
        field=self.fields["output_model"]
        if field.text()==self._auto_output:
            self._auto_output="models/"+(self.name.text().strip() or "trance-experiment")+".bin"
            field.setText(self._auto_output)
        self._changed()

    def _changed(self):
        if self._loading:return
        self.saved_path=None;self.use_button.setEnabled(False)
        config=self.config();self.preview.setPlainText(json.dumps(config,indent=2))
        count=parameter_count(config)
        self.summary.setText(f"{count/1e6:.2f}M parameters · weights alone {count*4/1048576:.1f} MiB · {config['training_steps']:,} steps")
        try:
            storage.validate_config(config,check_files=False)
            name=self.name.text().strip()
            if name.endswith(".json"):name=name[:-5]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",name):raise ValueError("Choose a config name using letters, numbers, dots, underscores or hyphens")
            self.message.setText("Choose existing data files before saving. Saving does not start training.")
            self.save_button.setEnabled(True)
        except ValueError as error:
            self.message.setText(str(error));self.save_button.setEnabled(False)

    def _relative(self,path):
        path=Path(path).resolve()
        try:return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:return str(path)

    def _browse(self,key):
        if key in ("train_data","validation_data"):
            names,_=QFileDialog.getOpenFileNames(self,"Choose conversation data",str(self.root/"data"),"Text files (*.txt);;All files (*)")
            if names:self.fields[key].setText(",".join(self._relative(p) for p in names))
        elif key=="output_model":
            name,_=QFileDialog.getSaveFileName(self,"Choose a new output model",str(self.root/"models/trance-experiment.bin"),"Trance models (*.bin)")
            if name:self.fields[key].setText(self._relative(name))
        else:
            name,_=QFileDialog.getOpenFileName(self,"Choose optimizer checkpoint",str(self.root/"models"),"Trance checkpoints (*.bin)")
            if name:self.fields[key].setText(self._relative(name))

    def _load_template(self):
        name,_=QFileDialog.getOpenFileName(self,"Load config as a template",str(self.root/"configs"),"JSON configs (*.json)")
        if not name:return
        try:
            config=json.loads(Path(name).read_text(encoding="utf-8"));config["output_model"]="models/"+(self.name.text().strip() or "trance-experiment")+".bin"
            self.set_config(config)
        except (ValueError,OSError,TypeError) as error:self.message.setText(str(error))

    def save(self):
        try:
            path=save_new_config(self.root,self.name.text(),self.config());self.saved_path=path
            self.message.setText(f"Saved {path.name}. Open in Train to review it; training starts only when you click Start training.")
            self.use_button.setEnabled(True);self.saved.emit(str(path))
        except (ValueError,OSError) as error:self.message.setText(str(error))
