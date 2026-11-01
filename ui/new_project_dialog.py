"""NexStitch — New Project dialog."""
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog,
    QFrame,
)
from PyQt6.QtCore import Qt

from core.project import CAMERA_RIGS, Project


_DEFAULT_LOC = str(Path.home() / "Documents")


class NewProjectDialog(QDialog):
    """Modal dialog for creating a new NexStitch project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setFixedSize(480, 260)
        self.setModal(True)

        self._project: Project | None = None
        self._location = Path(_DEFAULT_LOC)

        self._build_ui()
        self._refresh_preview()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(0)

        # Header
        hdr = QLabel("New Project")
        hdr.setStyleSheet("color:#ffffff; font-size:15px; font-weight:bold;"
                          " margin-bottom:14px;")
        root.addWidget(hdr)

        # Form grid
        gl = QGridLayout()
        gl.setSpacing(8)
        gl.setColumnStretch(1, 1)

        # Name
        gl.addWidget(self._field_label("Name"), 0, 0)
        self._edit_name = QLineEdit()
        self._edit_name.setPlaceholderText("e.g.  Shoot_20240318")
        self._edit_name.setFixedHeight(28)
        self._edit_name.textChanged.connect(self._refresh_preview)
        gl.addWidget(self._edit_name, 0, 1, 1, 2)

        # Location
        gl.addWidget(self._field_label("Location"), 1, 0)
        self._edit_loc = QLineEdit(str(self._location))
        self._edit_loc.setFixedHeight(28)
        self._edit_loc.textChanged.connect(self._on_loc_edited)
        gl.addWidget(self._edit_loc, 1, 1)

        btn_browse = QPushButton("…")
        btn_browse.setFixedSize(32, 28)
        btn_browse.clicked.connect(self._browse)
        gl.addWidget(btn_browse, 1, 2)

        # Camera rig
        gl.addWidget(self._field_label("Camera Rig"), 2, 0)
        self._cmb_rig = QComboBox()
        self._cmb_rig.setFixedHeight(28)
        for rig in CAMERA_RIGS:
            self._cmb_rig.addItem(rig)
        gl.addWidget(self._cmb_rig, 2, 1, 1, 2)

        root.addLayout(gl)
        root.addSpacing(10)

        # Path preview
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#2a2a2a;")
        root.addWidget(sep)
        root.addSpacing(6)

        self._lbl_preview = QLabel()
        self._lbl_preview.setStyleSheet(
            "color:#666666; font-size:9px;"
            " font-family:'JetBrains Mono','Consolas',monospace;")
        self._lbl_preview.setWordWrap(True)
        root.addWidget(self._lbl_preview)

        root.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedWidth(90)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self._btn_create = QPushButton("Create")
        self._btn_create.setFixedWidth(90)
        self._btn_create.setDefault(True)
        self._btn_create.setStyleSheet(
            "QPushButton { background:#00aa55; color:#000000; font-weight:bold;"
            " border:none; border-radius:3px; }"
            "QPushButton:hover { background:#00ff88; }"
            "QPushButton:disabled { background:#1e3a2a; color:#555555; }"
        )
        self._btn_create.clicked.connect(self._create)
        btn_row.addWidget(self._btn_create)

        root.addLayout(btn_row)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#888888; font-size:10px;")
        lbl.setFixedWidth(88)
        return lbl

    # ── slots ────────────────────────────────────────────────────────────────

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Project Location", str(self._location))
        if d:
            self._location = Path(d)
            self._edit_loc.setText(d)

    def _on_loc_edited(self, text: str):
        self._location = Path(text) if text else Path(_DEFAULT_LOC)
        self._refresh_preview()

    def _refresh_preview(self):
        name = self._edit_name.text().strip()
        if name:
            preview = str(self._location / name) + "/"
            self._lbl_preview.setText(f"→  {preview}")
            self._btn_create.setEnabled(True)
        else:
            self._lbl_preview.setText("Enter a project name above")
            self._btn_create.setEnabled(False)

    def _create(self):
        name = self._edit_name.text().strip()
        if not name:
            return
        loc  = Path(self._edit_loc.text().strip()) if self._edit_loc.text().strip() \
               else self._location
        rig  = self._cmb_rig.currentText()
        try:
            self._project = Project.create(loc, name, rig)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Could not create project:\n{e}")
            return
        self.accept()

    # ── result ───────────────────────────────────────────────────────────────

    @property
    def project(self) -> Project | None:
        return self._project
