from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
import numpy as np
from ui.utils import numpy_to_pixmap, fit_pixmap
from core.settings import CAM_ROLES


class ThumbCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background:#2d2d2d; border:1px solid #3a3a3a; border-radius:4px; }")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setMinimumHeight(80)
        self._img_lbl.setStyleSheet("background:#111111; border:none;")
        self._img_lbl.setText("—")
        lay.addWidget(self._img_lbl)

        name_lbl = QLabel(f"CAM{cam_id}  {CAM_ROLES.get(cam_id, '')}")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("color:#aaaaaa; font-size:10px; border:none;")
        lay.addWidget(name_lbl)

        self._info_lbl = QLabel("—")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_lbl.setStyleSheet("color:#666666; font-size:9px; border:none;")
        lay.addWidget(self._info_lbl)

    def set_frame(self, rgb: np.ndarray):
        pm = numpy_to_pixmap(rgb)
        self._img_lbl.setPixmap(fit_pixmap(pm, 200, 120))
        h, w = rgb.shape[:2]
        self._info_lbl.setText(f"{w}×{h}")

    def set_selected(self, selected: bool):
        border = "#00ff88" if selected else "#3a3a3a"
        self.setStyleSheet(
            f"QFrame {{ background:#2d2d2d; border:2px solid {border}; border-radius:4px; }}"
        )

    def mousePressEvent(self, ev):
        self.clicked.emit(self.cam_id)


class ThumbnailPanel(QWidget):
    camera_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(160)
        self.setMaximumWidth(240)
        self._cards: dict[int, ThumbCard] = {}
        self._selected = 0
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(6)

        hdr = QLabel("CAMERAS")
        hdr.setObjectName("section_header")
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        self._vbox = QVBoxLayout(container)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(6)

        for cam_id in range(1, 6):
            card = ThumbCard(cam_id)
            card.clicked.connect(self._on_card_clicked)
            self._cards[cam_id] = card
            self._vbox.addWidget(card)

        self._vbox.addStretch()
        scroll.setWidget(container)
        lay.addWidget(scroll)

    def set_frame(self, cam_id: int, rgb: np.ndarray):
        if cam_id in self._cards:
            self._cards[cam_id].set_frame(rgb)

    def set_selected(self, cam_id: int):
        if self._selected and self._selected in self._cards:
            self._cards[self._selected].set_selected(False)
        self._selected = cam_id
        if cam_id in self._cards:
            self._cards[cam_id].set_selected(True)

    def _on_card_clicked(self, cam_id: int):
        self.set_selected(cam_id)
        self.camera_selected.emit(cam_id)
