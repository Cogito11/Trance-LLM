"""Offline, searchable articles bundled with the desktop application."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt,QUrl
from .theme import COLORS as C
from PySide6.QtGui import QDesktopServices,QColor,QPalette,QTextCursor,QTextCharFormat
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel,QLineEdit,QTreeWidget,QTreeWidgetItem,QTextBrowser,QSplitter

ARTICLES=Path(__file__).parent/"articles"


def load_articles(directory=ARTICLES):
    directory=Path(directory).resolve();index=json.loads((directory/"index.json").read_text(encoding="utf-8"));result=[]
    ids=set()
    for entry in index:
        if entry["id"] in ids:raise ValueError("Duplicate documentation article ID")
        ids.add(entry["id"])
        path=(directory/entry["file"]).resolve()
        if not path.is_relative_to(directory):raise ValueError("Article path must remain inside the documentation folder")
        body=path.read_text(encoding="utf-8")
        result.append({**entry,"body":body,"search":(entry["title"]+" "+entry["section"]+" "+body).casefold()})
    return result


class Documentation(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.articles=load_articles();self.by_id={a["id"]:a for a in self.articles};self.current_id=None
        layout=QVBoxLayout(self);layout.setContentsMargins(30,28,30,24);layout.setSpacing(16)
        title=QLabel("Documentation");title.setObjectName("title");layout.addWidget(title)
        self.search=QLineEdit();self.search.setPlaceholderText("Search articles, concepts or config settings…");layout.addWidget(self.search)
        splitter=QSplitter();layout.addWidget(splitter,1)
        self.tree=QTreeWidget();self.tree.setHeaderHidden(True);self.tree.setMinimumWidth(230);splitter.addWidget(self.tree)
        self.reader=QTextBrowser();self.reader.setObjectName("articleReader");self.reader.setOpenLinks(False);self.reader.setOpenExternalLinks(False)
        palette=self.reader.palette();palette.setColor(QPalette.ColorRole.Link,QColor(C["accent"]));self.reader.setPalette(palette)
        self.reader.anchorClicked.connect(self._link);splitter.addWidget(self.reader);splitter.setStretchFactor(1,1);splitter.setSizes([260,710])
        self.footer=QLabel("Articles are available offline. External sources open only when you click their links.");self.footer.setObjectName("muted");self.footer.setWordWrap(True);layout.addWidget(self.footer)
        self.tree.currentItemChanged.connect(self._selected);self.search.textChanged.connect(self._filter);self._filter("")

    def _filter(self,text):
        terms=text.casefold().split();previous=self.current_id;self.tree.blockSignals(True);self.tree.clear();groups={};first=None;selected=None
        matches=[article for article in self.articles if all(term in article["search"] for term in terms)]
        if terms:matches.sort(key=lambda a:0 if all(term in a["title"].casefold() for term in terms) else 1)
        for article in matches:
            section=article["section"]
            if section not in groups:
                item=QTreeWidgetItem([section]);item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.tree.addTopLevelItem(item);item.setExpanded(True);groups[section]=item
            item=QTreeWidgetItem([article["title"]]);item.setData(0,Qt.ItemDataRole.UserRole,article["id"]);item.setToolTip(0,article["title"])
            groups[section].addChild(item)
            if first is None:first=item
            if article["id"]==previous:selected=item
        self.tree.blockSignals(False)
        item=selected or first
        if item:self.tree.setCurrentItem(item)
        else:
            self.current_id=None;self.reader.setPlainText("No articles match this search. Try a concept such as attention, tokenizer or validation.")

    def _selected(self,item,previous=None):
        if not item:return
        article_id=item.data(0,Qt.ItemDataRole.UserRole)
        if article_id not in self.by_id:return
        self.current_id=article_id;self.reader.setMarkdown(self.by_id[article_id]["body"])
        document=self.reader.document();block=document.begin();links=[]
        while block.isValid():
            cursor=QTextCursor(block);format=block.blockFormat();format.setBottomMargin(8);cursor.setBlockFormat(format)
            fragments=block.begin()
            while not fragments.atEnd():
                fragment=fragments.fragment()
                if fragment.isValid() and fragment.charFormat().isAnchor():links.append((fragment.position(),fragment.length()))
                fragments+=1
            block=block.next()
        for position,length in links:
            cursor=QTextCursor(document);cursor.setPosition(position);cursor.setPosition(position+length,QTextCursor.MoveMode.KeepAnchor)
            format=QTextCharFormat();format.setForeground(QColor(C["accent"]));cursor.mergeCharFormat(format)
        self.reader.verticalScrollBar().setValue(0)

    def restyle(self):
        """Re-apply theme colors after a palette change."""
        palette=self.reader.palette();palette.setColor(QPalette.ColorRole.Link,QColor(C["accent"]));self.reader.setPalette(palette)
        item=self.tree.currentItem()
        if item:self._selected(item)

    def open_article(self,article_id):
        if article_id not in self.by_id:return
        self.search.clear()
        for i in range(self.tree.topLevelItemCount()):
            group=self.tree.topLevelItem(i)
            for j in range(group.childCount()):
                item=group.child(j)
                if item.data(0,Qt.ItemDataRole.UserRole)==article_id:
                    self.tree.setCurrentItem(item);self.tree.scrollToItem(item);return

    def _link(self,url:QUrl):
        if url.scheme()=="article":self.open_article(url.host() or url.path().lstrip("/"))
        elif url.scheme() in ("https","http"):QDesktopServices.openUrl(url)
