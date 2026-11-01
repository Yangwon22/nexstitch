"""멀티캠 뷰 — 5개 카메라 동시 표시 + 오버랩 시각화 + 조정 패널"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QSlider, QSpinBox, QPushButton,
    QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QColor

from ui.utils import numpy_to_pixmap, fit_pixmap
from core.settings import CAM_ROLES
from core.calibration import load_rig_layout, save_rig_layout

# Grid position: cam_id -> (row, col)
#  [CAM3 좌2] [CAM1 정면] [CAM4 우1]
#             [CAM2 좌1]  [CAM5 우2]
GRID_POS = {3: (0, 0), 1: (0, 1), 4: (0, 2), 2: (1, 1), 5: (1, 2)}

# Overlap edge indicators (which cameras show green on which edge)
_LEFT_OV_CAMS  = {1, 4, 5}   # left-edge overlap indicator
_RIGHT_OV_CAMS = {3, 1, 2}   # right-edge overlap indicator

_DEFAULT_ADJ = {"dx": 0, "dy": 0, "rotation": 0.0, "brightness": 0.0, "contrast": 1.0}

_MONO_STYLE = "font-family:'JetBrains Mono','Consolas',monospace; font-size:10px; color:#00ff88;"


class CamCell(QFrame):
    """Single camera preview cell with overlap overlay and drag support."""
    clicked = pyqtSignal(int)
    dragged = pyqtSignal(int, int, int)   # cam_id, ddx, ddy

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self._selected = False
        self._overlap_l = 0
        self._overlap_r = 0
        self._rgb: np.ndarray | None = None
        self._drag_origin: QPoint | None = None

        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._apply_style()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background:#0d0d0d; border:none;")
        self._img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img_lbl.setMinimumHeight(80)
        lay.addWidget(self._img_lbl, stretch=1)

        role = CAM_ROLES.get(cam_id, "")
        name_lbl = QLabel(f"CAM{cam_id}  {role}")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("color:#888888; font-size:10px; background:transparent; border:none;")
        lay.addWidget(name_lbl)

    # ── public ───────────────────────────────────────────────
    def set_frame(self, rgb: np.ndarray | None):
        self._rgb = rgb
        self._refresh()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._apply_style()
        self.setCursor(
            Qt.CursorShape.SizeAllCursor if sel else Qt.CursorShape.PointingHandCursor
        )

    def set_overlap(self, left_pct: int, right_pct: int):
        self._overlap_l = left_pct
        self._overlap_r = right_pct
        self._refresh()

    # ── mouse ────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.cam_id)
            if self._selected:
                self._drag_origin = ev.position().toPoint()

    def mouseMoveEvent(self, ev):
        if self._selected and self._drag_origin is not None:
            delta = ev.position().toPoint() - self._drag_origin
            self._drag_origin = ev.position().toPoint()
            # Scale: cell pixel → canvas pixel (approx 3× for 4K→thumbnail ratio)
            self.dragged.emit(self.cam_id, delta.x() * 3, delta.y() * 3)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None

    # ── internal ─────────────────────────────────────────────
    def _apply_style(self):
        if self._selected:
            border = "2px solid #ffdd00"
        else:
            border = "1px solid #3a3a3a"
        self.setStyleSheet(
            f"QFrame {{ background:#1e1e1e; border:{border}; border-radius:4px; }}"
        )

    def _refresh(self):
        if self._rgb is None:
            self._img_lbl.setText("—")
            self._img_lbl.setPixmap(QPixmap())
            return
        w_cell = max(self._img_lbl.width(), 160)
        h_cell = max(self._img_lbl.height(), 80)
        pm = fit_pixmap(numpy_to_pixmap(self._rgb), w_cell, h_cell)
        if self._overlap_l > 0 or self._overlap_r > 0:
            pm = pm.copy()
            p = QPainter(pm)
            ov = QColor(0, 220, 100, 70)
            if self._overlap_l > 0:
                p.fillRect(0, 0, int(pm.width() * self._overlap_l / 100), pm.height(), ov)
            if self._overlap_r > 0:
                p.fillRect(
                    pm.width() - int(pm.width() * self._overlap_r / 100), 0,
                    int(pm.width() * self._overlap_r / 100), pm.height(), ov,
                )
            p.end()
        self._img_lbl.setPixmap(pm)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refresh()


class AdjustPanel(QWidget):
    """Per-camera adjustment sliders."""
    changed        = pyqtSignal(int, dict)  # cam_id, params
    save_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cam_id: int | None = None
        self._blocking = False
        self._build_ui()
        self.setEnabled(False)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._title = QLabel("카메라를 선택하세요")
        self._title.setStyleSheet("color:#00ff88; font-size:11px; font-weight:bold;")
        lay.addWidget(self._title)

        # ── Position offset ──────────────────────────────────
        gb_pos = QGroupBox("위치 오프셋 (px)")
        pos_lay = QGridLayout(gb_pos)
        pos_lay.setSpacing(4)

        pos_lay.addWidget(QLabel("X (dx)"), 0, 0)
        self._spin_dx = QSpinBox()
        self._spin_dx.setRange(-2000, 2000)
        self._spin_dx.valueChanged.connect(self._emit)
        pos_lay.addWidget(self._spin_dx, 0, 1)

        pos_lay.addWidget(QLabel("Y (dy)"), 1, 0)
        self._spin_dy = QSpinBox()
        self._spin_dy.setRange(-2000, 2000)
        self._spin_dy.valueChanged.connect(self._emit)
        pos_lay.addWidget(self._spin_dy, 1, 1)
        lay.addWidget(gb_pos)

        # ── Rotation ─────────────────────────────────────────
        gb_rot = QGroupBox("회전 (°)")
        rot_lay = QVBoxLayout(gb_rot)
        rot_lay.setSpacing(2)
        self._sld_rot = QSlider(Qt.Orientation.Horizontal)
        self._sld_rot.setRange(-300, 300)   # ×0.1 deg
        self._lbl_rot = QLabel("0.0°")
        self._lbl_rot.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_rot.setStyleSheet(_MONO_STYLE)
        self._sld_rot.valueChanged.connect(self._on_rot)
        rot_lay.addWidget(self._sld_rot)
        rot_lay.addWidget(self._lbl_rot)
        lay.addWidget(gb_rot)

        # ── Brightness / Contrast ────────────────────────────
        gb_col = QGroupBox("밝기 / 대비")
        col_lay = QGridLayout(gb_col)
        col_lay.setSpacing(4)

        col_lay.addWidget(QLabel("밝기"), 0, 0)
        self._sld_br = QSlider(Qt.Orientation.Horizontal)
        self._sld_br.setRange(-100, 100)
        self._lbl_br = QLabel("0")
        self._lbl_br.setFixedWidth(36)
        self._lbl_br.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_br.setStyleSheet(_MONO_STYLE)
        self._sld_br.valueChanged.connect(self._on_br)
        col_lay.addWidget(self._sld_br, 0, 1)
        col_lay.addWidget(self._lbl_br, 0, 2)

        col_lay.addWidget(QLabel("대비"), 1, 0)
        self._sld_ct = QSlider(Qt.Orientation.Horizontal)
        self._sld_ct.setRange(50, 200)      # ×0.01 → 0.50–2.00
        self._sld_ct.setValue(100)
        self._lbl_ct = QLabel("1.00×")
        self._lbl_ct.setFixedWidth(40)
        self._lbl_ct.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_ct.setStyleSheet(_MONO_STYLE)
        self._sld_ct.valueChanged.connect(self._on_ct)
        col_lay.addWidget(self._sld_ct, 1, 1)
        col_lay.addWidget(self._lbl_ct, 1, 2)
        lay.addWidget(gb_col)

        btn_reset = QPushButton("리셋")
        btn_reset.clicked.connect(self._reset)
        lay.addWidget(btn_reset)

        btn_save = QPushButton("저장  (rig_layout.json)")
        btn_save.setObjectName("accent")
        btn_save.clicked.connect(self.save_requested)
        lay.addWidget(btn_save)

        lay.addStretch()

    # ── public ───────────────────────────────────────────────
    def load_cam(self, cam_id: int, params: dict):
        self._cam_id = cam_id
        self._blocking = True
        self._title.setText(f"CAM{cam_id}  {CAM_ROLES.get(cam_id, '')}")
        self._spin_dx.setValue(int(params.get("dx", 0)))
        self._spin_dy.setValue(int(params.get("dy", 0)))
        self._sld_rot.setValue(int(params.get("rotation", 0.0) * 10))
        self._sld_br.setValue(int(params.get("brightness", 0.0)))
        self._sld_ct.setValue(int(params.get("contrast", 1.0) * 100))
        self._blocking = False
        self.setEnabled(True)

    def apply_drag(self, ddx: int, ddy: int):
        """Called when user drags the camera cell."""
        self._spin_dx.setValue(self._spin_dx.value() + ddx)
        self._spin_dy.setValue(self._spin_dy.value() + ddy)

    def current_params(self) -> dict:
        return {
            "dx":         self._spin_dx.value(),
            "dy":         self._spin_dy.value(),
            "rotation":   self._sld_rot.value() / 10.0,
            "brightness": float(self._sld_br.value()),
            "contrast":   self._sld_ct.value() / 100.0,
        }

    # ── slots ─────────────────────────────────────────────────
    def _on_rot(self, v: int):
        self._lbl_rot.setText(f"{v / 10:.1f}°")
        self._emit()

    def _on_br(self, v: int):
        self._lbl_br.setText(str(v))
        self._emit()

    def _on_ct(self, v: int):
        self._lbl_ct.setText(f"{v / 100:.2f}×")
        self._emit()

    def _emit(self):
        if self._blocking or self._cam_id is None:
            return
        self.changed.emit(self._cam_id, self.current_params())

    def _reset(self):
        if self._cam_id is None:
            return
        self._blocking = True
        self._spin_dx.setValue(0)
        self._spin_dy.setValue(0)
        self._sld_rot.setValue(0)
        self._sld_br.setValue(0)
        self._sld_ct.setValue(100)
        self._blocking = False
        self._emit()


class MultiCamView(QWidget):
    """5-camera grid with overlap visualization and per-cam adjustment panel."""
    stitch_requested = pyqtSignal()
    layout_changed   = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout: dict = {str(i): _DEFAULT_ADJ.copy() for i in range(1, 6)}
        self._selected_cam: int | None = None
        self._cells: dict[int, CamCell] = {}

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self.stitch_requested)

        self._build_ui()
        self._load_saved()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: overlap bar + camera grid ──────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 6, 4, 6)
        left_lay.setSpacing(6)

        ov_row = QHBoxLayout()
        lbl_ov = QLabel("오버랩 표시:")
        lbl_ov.setStyleSheet("color:#888888; font-size:10px;")
        ov_row.addWidget(lbl_ov)
        self._sld_ov = QSlider(Qt.Orientation.Horizontal)
        self._sld_ov.setRange(0, 50)
        self._sld_ov.setValue(20)
        self._sld_ov.valueChanged.connect(self._on_overlap_changed)
        ov_row.addWidget(self._sld_ov)
        self._lbl_ov = QLabel("20%")
        self._lbl_ov.setFixedWidth(32)
        self._lbl_ov.setStyleSheet(_MONO_STYLE)
        ov_row.addWidget(self._lbl_ov)
        left_lay.addLayout(ov_row)

        grid_w = QWidget()
        self._grid = QGridLayout(grid_w)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(0, 0, 0, 0)

        for cam_id, (row, col) in GRID_POS.items():
            cell = CamCell(cam_id)
            cell.clicked.connect(self._on_cell_clicked)
            cell.dragged.connect(self._on_cell_dragged)
            self._cells[cam_id] = cell
            self._grid.addWidget(cell, row, col)

        # Equal stretch for all columns/rows
        for c in range(3):
            self._grid.setColumnStretch(c, 1)
        for r in range(2):
            self._grid.setRowStretch(r, 1)

        # Transparent placeholder for empty cell (row=1, col=0)
        ph = QWidget()
        ph.setStyleSheet("background:transparent;")
        self._grid.addWidget(ph, 1, 0)

        left_lay.addWidget(grid_w, stretch=1)
        root.addWidget(left, stretch=1)

        # ── Right: adjustment panel ───────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#3a3a3a;")
        root.addWidget(sep)

        self._adj = AdjustPanel()
        self._adj.setFixedWidth(210)
        self._adj.changed.connect(self._on_adj_changed)
        self._adj.save_requested.connect(self.save_layout)
        root.addWidget(self._adj)

        self._update_overlaps()

    # ── public ───────────────────────────────────────────────
    def set_frames(self, frames: dict):
        for cam_id, cell in self._cells.items():
            cell.set_frame(frames.get(cam_id))

    def get_layout(self) -> dict:
        return dict(self._layout)

    def save_layout(self):
        save_rig_layout(self._layout)

    # ── slots ────────────────────────────────────────────────
    def _on_cell_clicked(self, cam_id: int):
        if self._selected_cam is not None and self._selected_cam in self._cells:
            self._cells[self._selected_cam].set_selected(False)
        self._selected_cam = cam_id
        self._cells[cam_id].set_selected(True)
        self._adj.load_cam(cam_id, self._layout.get(str(cam_id), _DEFAULT_ADJ.copy()))

    def _on_cell_dragged(self, cam_id: int, ddx: int, ddy: int):
        if cam_id == self._selected_cam:
            # apply_drag updates spinboxes → valueChanged → _adj.changed → _on_adj_changed
            self._adj.apply_drag(ddx, ddy)

    def _on_adj_changed(self, cam_id: int, params: dict):
        self._layout[str(cam_id)] = params
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _on_overlap_changed(self, val: int):
        self._lbl_ov.setText(f"{val}%")
        self._update_overlaps()

    # ── internal ─────────────────────────────────────────────
    def _update_overlaps(self):
        pct = self._sld_ov.value()
        for cam_id, cell in self._cells.items():
            l = pct if cam_id in _LEFT_OV_CAMS  else 0
            r = pct if cam_id in _RIGHT_OV_CAMS else 0
            cell.set_overlap(l, r)

    def _load_saved(self):
        loaded = load_rig_layout()
        if loaded:
            self._layout = loaded
        self._update_overlaps()
