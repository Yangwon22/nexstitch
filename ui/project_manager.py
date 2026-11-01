"""NexStitch — Project Manager (startup screen)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFrame, QFileDialog, QMenu, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor

from core.project import Project, NXS_FILENAME
from core.recent_projects import load_recent, add_recent, remove_recent


class ProjectManagerDialog(QDialog):
    """Startup window — pick a recent project or create / open one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NexStitch")
        self.setFixedSize(600, 500)
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowCloseButtonHint
        )

        self._project: Project | None = None
        self._build_ui()
        self._load_recent()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(88)
        header.setStyleSheet("background:#111111; border-bottom:1px solid #1e1e1e;")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(28, 16, 28, 16)
        hl.setSpacing(3)

        lbl_title = QLabel("NexStitch")
        lbl_title.setFont(QFont("Rajdhani", 22, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color:#00ff88; background:transparent; letter-spacing:3px;")
        hl.addWidget(lbl_title)

        lbl_sub = QLabel("RED Komodo  ·  5-Camera Cylindrical Stitching")
        lbl_sub.setStyleSheet("color:#555555; font-size:10px; background:transparent;")
        hl.addWidget(lbl_sub)

        root.addWidget(header)

        # ── Recent projects section ──────────────────────────────
        body = QFrame()
        body.setStyleSheet("background:#181818;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(28, 20, 28, 16)
        bl.setSpacing(10)

        lbl_recent = QLabel("RECENT PROJECTS")
        lbl_recent.setStyleSheet(
            "color:#666666; font-size:9px; font-weight:bold; letter-spacing:2px;")
        bl.addWidget(lbl_recent)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget {"
            "  background:#131313; border:1px solid #222222;"
            "  border-radius:4px; outline:none; }"
            "QListWidget::item {"
            "  padding:0; border-bottom:1px solid #1a1a1a; }"
            "QListWidget::item:selected {"
            "  background:#1e2e1e; border-left:2px solid #00ff88; }"
            "QListWidget::item:hover:!selected { background:#1a1a1a; }"
        )
        self._list.setIconSize(QSize(0, 0))
        self._list.setSpacing(0)
        self._list.itemDoubleClicked.connect(self._open_selected)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        bl.addWidget(self._list, stretch=1)

        self._lbl_empty = QLabel("No recent projects.  Create or open one below.")
        self._lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_empty.setStyleSheet("color:#444444; font-size:10px;")
        self._lbl_empty.setVisible(False)
        bl.addWidget(self._lbl_empty)

        root.addWidget(body, stretch=1)

        # ── Footer buttons ───────────────────────────────────────
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setStyleSheet(
            "background:#111111; border-top:1px solid #1e1e1e;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(28, 0, 28, 0)
        fl.setSpacing(10)

        btn_new = QPushButton("New Project")
        btn_new.setFixedHeight(36)
        btn_new.setMinimumWidth(140)
        btn_new.setStyleSheet(
            "QPushButton { background:#00aa55; color:#000000; font-weight:bold;"
            "  font-size:11px; border:none; border-radius:4px; }"
            "QPushButton:hover { background:#00ff88; }"
        )
        btn_new.clicked.connect(self._new_project)
        fl.addWidget(btn_new)

        btn_open = QPushButton("Open Project…")
        btn_open.setFixedHeight(36)
        btn_open.setMinimumWidth(140)
        btn_open.setStyleSheet(
            "QPushButton { background:#1e1e1e; color:#cccccc;"
            "  font-size:11px; border:1px solid #333333; border-radius:4px; }"
            "QPushButton:hover { border-color:#555555; color:#ffffff; }"
        )
        btn_open.clicked.connect(self._open_project)
        fl.addWidget(btn_open)

        fl.addStretch()

        btn_open_sel = QPushButton("Open Selected  ▶")
        btn_open_sel.setFixedHeight(36)
        btn_open_sel.setMinimumWidth(140)
        btn_open_sel.setStyleSheet(
            "QPushButton { background:#1a2a1a; color:#00ff88;"
            "  font-size:11px; border:1px solid #00ff88; border-radius:4px; }"
            "QPushButton:hover { background:#223322; }"
            "QPushButton:disabled { color:#2a4a2a; border-color:#1a3a1a;"
            "  background:#111111; }"
        )
        btn_open_sel.clicked.connect(self._open_selected)
        fl.addWidget(btn_open_sel)
        self._btn_open_sel = btn_open_sel

        root.addWidget(footer)

        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._on_selection_changed()

    # ── data ────────────────────────────────────────────────────────────────

    def _load_recent(self):
        self._list.clear()
        entries = load_recent()
        if not entries:
            self._list.setVisible(False)
            self._lbl_empty.setVisible(True)
            return

        self._list.setVisible(True)
        self._lbl_empty.setVisible(False)

        for entry in entries:
            nxs_str  = entry.get("nxs", "")
            name     = entry.get("name", Path(nxs_str).parent.name)
            modified = entry.get("modified", "")
            exists   = Path(nxs_str).exists()

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, nxs_str)
            item.setSizeHint(QSize(0, 56))

            w = _RecentItem(name, nxs_str, modified, exists)
            self._list.addItem(item)
            self._list.setItemWidget(item, w)

            if not exists:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_selection_changed(self):
        sel = self._list.selectedItems()
        enabled = bool(sel) and bool(
            sel[0].flags() & Qt.ItemFlag.ItemIsEnabled)
        self._btn_open_sel.setEnabled(enabled)

    def _open_selected(self, *_):
        sel = self._list.selectedItems()
        if not sel:
            return
        nxs_str = sel[0].data(Qt.ItemDataRole.UserRole)
        if not nxs_str:
            return
        self._load_project(Path(nxs_str))

    def _new_project(self):
        from ui.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.project:
            proj = dlg.project
            add_recent(proj)
            self._project = proj
            self.accept()

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            str(Path.home()),
            f"NexStitch Project (*{NXS_FILENAME});;All files (*)"
        )
        if path:
            self._load_project(Path(path))

    def _load_project(self, nxs_path: Path):
        try:
            proj = Project.load(nxs_path)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error",
                                 f"Could not open project:\n{e}")
            return
        add_recent(proj)
        self._project = proj
        self.accept()

    def _show_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        nxs_str = item.data(Qt.ItemDataRole.UserRole)

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background:#1e1e1e; color:#cccccc; border:1px solid #333333; }"
            "QMenu::item:selected { background:#2a2a2a; color:#ffffff; }"
        )
        act_open   = menu.addAction("Open")
        menu.addSeparator()
        act_remove = menu.addAction("Remove from list")

        act_open.setEnabled(bool(item.flags() & Qt.ItemFlag.ItemIsEnabled))

        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen == act_open:
            self._load_project(Path(nxs_str))
        elif chosen == act_remove:
            remove_recent(nxs_str)
            self._load_recent()

    # ── result ───────────────────────────────────────────────────────────────

    @property
    def selected_project(self) -> Project | None:
        return self._project


# ── Custom list item widget ───────────────────────────────────────────────────

class _RecentItem(QFrame):
    """Single row in the recent-projects list."""

    def __init__(self, name: str, path_str: str, modified: str,
                 exists: bool, parent=None):
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setStyleSheet("background:transparent; border:none;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)

        name_color  = "#ffffff" if exists else "#555555"
        path_color  = "#666666" if exists else "#3a3a3a"
        date_color  = "#444444"

        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(
            f"color:{name_color}; font-size:11px; font-weight:bold;"
            " background:transparent;")
        lay.addWidget(lbl_name)

        # Second row: path + date
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.setContentsMargins(0, 0, 0, 0)

        suffix = "  [not found]" if not exists else ""
        lbl_path = QLabel(path_str + suffix)
        lbl_path.setStyleSheet(
            f"color:{path_color}; font-size:8px;"
            " font-family:'JetBrains Mono','Consolas',monospace;"
            " background:transparent;")
        lbl_path.setSizePolicy(
            lbl_path.sizePolicy().horizontalPolicy(),
            lbl_path.sizePolicy().verticalPolicy())
        row2.addWidget(lbl_path, stretch=1)

        if modified:
            date_str = modified.replace("T", "  ")[:16]
            lbl_date = QLabel(date_str)
            lbl_date.setStyleSheet(
                f"color:{date_color}; font-size:8px;"
                " font-family:'JetBrains Mono','Consolas',monospace;"
                " background:transparent;")
            row2.addWidget(lbl_date)

        lay.addLayout(row2)
