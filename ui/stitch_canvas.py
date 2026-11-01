"""
NexStitch  ─  StitchCanvas
Mistika VR style: canvas = always latest stitching result
"""
import copy
import math
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGraphicsScene,
    QGraphicsView, QGraphicsPixmapItem, QGraphicsItem, QGraphicsObject,
    QGraphicsTextItem,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QPushButton,
    QFrame, QGroupBox, QCheckBox, QComboBox, QInputDialog,
    QMessageBox, QSizePolicy, QGridLayout, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QImage, QColor, QKeyEvent, QPainter, QBrush, QPen, QFont

from ui.utils import numpy_to_pixmap
from ui.workers import CylStitchWorker
from core.settings import CAM_ROLES, ADJ_PAIRS
from core.calibration import (
    load_cyl_params, save_cyl_params,
    list_presets, load_preset, save_preset, delete_preset,
    load_output_resolution, save_output_resolution,
    load_color_params, save_color_params,
)
from core.stitch import render_overlays, improve_yaw
from core.color import apply_output_color, default_color_params

PREVIEW_SCALE = 0.25   # default quality (fallback constant only)
DRAFT_SCALE   = 0.10   # fast preview during active drag
_MONO = "font-family:'JetBrains Mono','Consolas',monospace; font-size:10px; color:#00ff88;"

_QUALITY_PRESETS = [
    ("Draft   0.10×", 0.10),
    ("Low     0.25×", 0.25),
    ("Medium  0.50×", 0.50),
    ("High    0.75×", 0.75),
    ("Full    1.00×", 1.00),
]
_CAM_COLORS = {1:"#ffffff", 2:"#ffffff", 3:"#ffffff", 4:"#ffffff", 5:"#ffffff"}

_RES_PRESETS = [
    ("7680 × 2160  (8K Ultrawide)", 7680, 2160),
    ("5760 × 1080  (6K Ultrawide)", 5760, 1080),
    ("3840 × 1080  (4K Ultrawide)", 3840, 1080),
    ("3840 ×  720  (4K Wide)",      3840,  720),
    ("Custom",                         0,    0),
]


def rgba_to_pixmap(rgba: np.ndarray) -> QPixmap:
    h, w = rgba.shape[:2]
    qi = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


# ─────────────────────────────────────────────────────────────
# Canvas view (zoom + pan, keyboard shortcuts)
# ─────────────────────────────────────────────────────────────
class CanvasView(QGraphicsView):
    zoom_to_cam_requested = pyqtSignal(int)          # cam_id
    cam_dropped           = pyqtSignal(int, float, float)  # cam_id, scene_x, scene_y

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor(0x0a, 0x0a, 0x0a)))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self._panning   = False
        self._pan_last  = None
        self._space_on  = False

    def fit_all(self):
        self.fitInView(self.scene().sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k == Qt.Key.Key_A:
            self.fit_all()
        elif k == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_on = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev: QKeyEvent):
        if ev.key() == Qt.Key.Key_Space:
            self._space_on = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.MiddleButton or
                (ev.button() == Qt.MouseButton.LeftButton and self._space_on)):
            self._panning  = True
            self._pan_last = ev.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_last:
            d = ev.pos() - self._pan_last
            self._pan_last = ev.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._panning:
            self._panning  = False
            self._pan_last = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_on
                else Qt.CursorShape.ArrowCursor)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    # ── drag-and-drop from source panel ──────────────────────
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat("application/x-nexstitch-cam"):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat("application/x-nexstitch-cam"):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        if ev.mimeData().hasFormat("application/x-nexstitch-cam"):
            cam_id    = int(ev.mimeData().data("application/x-nexstitch-cam").data())
            scene_pos = self.mapToScene(ev.position().toPoint())
            self.cam_dropped.emit(cam_id, scene_pos.x(), scene_pos.y())
            ev.acceptProposedAction()


# ─────────────────────────────────────────────────────────────
# Camera icon  (draggable overlay on the stitch canvas)
# ─────────────────────────────────────────────────────────────
class CamIcon(QGraphicsObject):
    drag_delta = pyqtSignal(int, float, float)   # cam_id, dx, dy (scene coords)
    drag_ended = pyqtSignal(int)                  # cam_id — emitted on mouse release after drag
    clicked    = pyqtSignal(int)                  # cam_id

    W, H = 64, 28

    def __init__(self, cam_id: int, color: str, parent=None):
        super().__init__(parent)
        self.cam_id    = cam_id
        self._color    = QColor(color)
        self._selected = False
        self._drag_last: QPointF | None = None
        self._moved    = False           # True once any mouse-move delta is emitted
        self.setZValue(20)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def boundingRect(self) -> QRectF:
        return QRectF(-self.W / 2, -self.H / 2, self.W, self.H)

    def paint(self, painter: QPainter, option, widget=None):
        r = QRectF(-self.W / 2, -self.H / 2, self.W, self.H)
        painter.fillRect(r, QColor(0, 0, 0, 190))
        pen_color = QColor("#00ff88") if self._selected else self._color
        painter.setPen(QPen(pen_color, 2))
        painter.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4, 4)
        painter.setPen(QPen(pen_color))
        font = QFont("Consolas", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, f"CAM{self.cam_id}")

    def set_selected(self, v: bool):
        self._selected = v
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_last = ev.scenePos()
            self._moved     = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.clicked.emit(self.cam_id)
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_last is not None:
            delta           = ev.scenePos() - self._drag_last
            self._drag_last = ev.scenePos()
            self._moved     = True
            self.drag_delta.emit(self.cam_id, delta.x(), delta.y())
            ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            was_dragging    = self._moved
            self._drag_last = None
            self._moved     = False
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            if was_dragging:
                self.drag_ended.emit(self.cam_id)
            ev.accept()


# ─────────────────────────────────────────────────────────────
# Parameter panel (right side)
# ─────────────────────────────────────────────────────────────
class ParamPanel(QWidget):
    params_changed      = pyqtSignal()
    slider_drag_started = pyqtSignal()   # any YPR slider pressed
    slider_drag_ended   = pyqtSignal()   # any YPR slider released

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#181818;")
        self._cam_id   = 1
        self._blocking = False
        self._params   = load_cyl_params()
        self._build_ui()
        self._load_cam(1)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Camera selector ──
        cam_row = QHBoxLayout()
        cam_row.setSpacing(3)
        self._cam_btns: dict[int, QPushButton] = {}
        for cam_id in range(1, 6):
            btn = QPushButton(f"{cam_id}")
            btn.setFixedSize(36, 26)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { border:1px solid #ffffff33; border-radius:3px; color:#888888; }"
                "QPushButton:checked { background:#00ff8822; border-color:#00ff88; color:#00ff88; font-weight:bold; }"
            )
            btn.clicked.connect(lambda _, c=cam_id: self._select_cam(c))
            cam_row.addWidget(btn)
            self._cam_btns[cam_id] = btn
        cam_row.addStretch()
        lay.addLayout(cam_row)

        self._lbl_role = QLabel("")
        self._lbl_role.setStyleSheet("color:#00ff88; font-size:10px; font-weight:bold;")
        lay.addWidget(self._lbl_role)

        # ── Enabled ──
        self._chk_enabled = QCheckBox("Enabled")
        self._chk_enabled.setChecked(True)
        self._chk_enabled.toggled.connect(self._on_enabled)
        lay.addWidget(self._chk_enabled)

        def _sep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.HLine)
            s.setStyleSheet("color:#2a2a2a;"); return s

        lay.addWidget(_sep())

        # ── YPR sliders ──
        gb_ypr = QGroupBox("Rotation (YPR)")
        gb_ypr.setStyleSheet("QGroupBox { color:#666666; font-size:9px; }")
        gl = QGridLayout(gb_ypr)
        gl.setSpacing(4)

        self._sl_yaw,   self._sb_yaw,   _ = self._make_ypr_row(gl, 0, "Yaw",   -1800, 1800,  0)
        self._sl_pitch, self._sb_pitch, _ = self._make_ypr_row(gl, 1, "Pitch",  -900,  900,   0)
        self._sl_roll,  self._sb_roll,  _ = self._make_ypr_row(gl, 2, "Roll",   -450,  450,   0)
        lay.addWidget(gb_ypr)

        # ── Lens ──
        gb_lens = QGroupBox("Lens Distortion")
        gb_lens.setStyleSheet("QGroupBox { color:#666666; font-size:9px; }")
        ll = QGridLayout(gb_lens)
        ll.setSpacing(4)

        ll.addWidget(QLabel("k1"), 0, 0)
        self._dsb_k1 = QDoubleSpinBox()
        self._dsb_k1.setRange(-1.0, 1.0); self._dsb_k1.setSingleStep(0.001)
        self._dsb_k1.setDecimals(4); self._dsb_k1.setStyleSheet(_MONO)
        self._dsb_k1.valueChanged.connect(self._on_lens_changed)
        ll.addWidget(self._dsb_k1, 0, 1)

        ll.addWidget(QLabel("k2"), 1, 0)
        self._dsb_k2 = QDoubleSpinBox()
        self._dsb_k2.setRange(-1.0, 1.0); self._dsb_k2.setSingleStep(0.001)
        self._dsb_k2.setDecimals(4); self._dsb_k2.setStyleSheet(_MONO)
        self._dsb_k2.valueChanged.connect(self._on_lens_changed)
        ll.addWidget(self._dsb_k2, 1, 1)
        lay.addWidget(gb_lens)

        # ── Blend ──
        gb_bl = QGroupBox("Blending")
        gb_bl.setStyleSheet("QGroupBox { color:#666666; font-size:9px; }")
        bl = QGridLayout(gb_bl)

        bl.addWidget(QLabel("Feather W"), 0, 0)
        self._sb_feather = QSpinBox()
        self._sb_feather.setRange(0, 500); self._sb_feather.setValue(200)
        self._sb_feather.setStyleSheet(_MONO)
        self._sb_feather.valueChanged.connect(lambda _: self.params_changed.emit())
        bl.addWidget(self._sb_feather, 0, 1)

        bl.addWidget(QLabel("Opacity"), 1, 0)
        self._sl_opacity = QSlider(Qt.Orientation.Horizontal)
        self._sl_opacity.setRange(0, 100); self._sl_opacity.setValue(100)
        self._lbl_opacity = QLabel("100%"); self._lbl_opacity.setStyleSheet(_MONO)
        self._sl_opacity.valueChanged.connect(
            lambda v: (self._lbl_opacity.setText(f"{v}%"), self.params_changed.emit()))
        bl.addWidget(self._sl_opacity, 1, 1); bl.addWidget(self._lbl_opacity, 1, 2)
        lay.addWidget(gb_bl)

        # ── Focal ──
        gb_f = QGroupBox("Focal Length (Global)")
        gb_f.setStyleSheet("QGroupBox { color:#666666; font-size:9px; }")
        fl = QHBoxLayout(gb_f)
        fl.addWidget(QLabel("focal"))
        self._sb_focal = QDoubleSpinBox()
        self._sb_focal.setRange(100, 10000); self._sb_focal.setValue(1100)
        self._sb_focal.setDecimals(1); self._sb_focal.setStyleSheet(_MONO)
        self._sb_focal.valueChanged.connect(lambda _: self.params_changed.emit())
        fl.addWidget(self._sb_focal)
        lay.addWidget(gb_f)

        # ── Reset + Save ──
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self._reset_cam)
        lay.addWidget(btn_reset)

        btn_reset_all = QPushButton("Reset All (Defaults)")
        btn_reset_all.clicked.connect(self._reset_all)
        lay.addWidget(btn_reset_all)

        lay.addStretch()

    def _make_ypr_row(self, grid, row, label, lo, hi, default):
        """Returns (slider, spinbox, label_widget)"""
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#888888; font-size:9px;")
        grid.addWidget(lbl, row, 0)

        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi); sl.setValue(default)
        grid.addWidget(sl, row, 1)

        sb = QDoubleSpinBox()
        sb.setRange(lo / 10.0, hi / 10.0)
        sb.setSingleStep(0.1); sb.setDecimals(1)
        sb.setFixedWidth(70); sb.setStyleSheet(_MONO)
        grid.addWidget(sb, row, 2)

        # Two-way sync
        sl.valueChanged.connect(
            lambda v, s=sb: (s.blockSignals(True), s.setValue(v/10.0), s.blockSignals(False),
                              self._on_ypr_changed()))
        sb.valueChanged.connect(
            lambda v, s=sl: (s.blockSignals(True), s.setValue(int(v*10)), s.blockSignals(False),
                              self._on_ypr_changed()))
        # Drag state — emitted to StitchCanvas for debounce logic
        sl.sliderPressed.connect(self.slider_drag_started)
        sl.sliderReleased.connect(self.slider_drag_ended)
        return sl, sb, lbl

    # ── public ────────────────────────────────────────────────
    def get_focal(self) -> float:
        return self._sb_focal.value()

    def get_feather_width(self) -> int:
        return self._sb_feather.value()

    def get_cam_params(self) -> dict:
        return copy.deepcopy(self._params)

    def set_cam_params(self, params: dict):
        """Set params from outside (e.g. improve_yaw result)."""
        self._params = copy.deepcopy(params)
        self._load_cam(self._cam_id)
        self.params_changed.emit()

    def load_external(self, params: dict):
        self._params = copy.deepcopy(params)
        focal = params.get("focal", 1100.0)
        self._sb_focal.blockSignals(True)
        self._sb_focal.setValue(focal)
        self._sb_focal.blockSignals(False)
        self._load_cam(self._cam_id)

    def update_cam_yaw_pitch(self, cam_id: int, yaw: float, pitch: float):
        """Directly update yaw/pitch for one camera (used during drag operations)."""
        key = f"cam{cam_id}"
        if key not in self._params:
            self._params[key] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                                  "k1": 0.0, "k2": 0.0, "enabled": True}
        self._params[key]["yaw"]   = yaw
        self._params[key]["pitch"] = pitch
        if cam_id == self._cam_id:
            self._blocking = True
            self._sl_yaw.setValue(int(yaw * 10));    self._sb_yaw.setValue(yaw)
            self._sl_pitch.setValue(int(pitch * 10)); self._sb_pitch.setValue(pitch)
            self._blocking = False
        self.params_changed.emit()

    # ── internal ──────────────────────────────────────────────
    def _select_cam(self, cam_id: int):
        self._cam_id = cam_id
        for cid, btn in self._cam_btns.items():
            btn.setChecked(cid == cam_id)
        self._load_cam(cam_id)

    def _load_cam(self, cam_id: int):
        key = f"cam{cam_id}"
        p   = self._params.get(key, {})
        role = CAM_ROLES.get(cam_id, "")
        self._lbl_role.setText(f"CAM{cam_id}  {role}")
        self._cam_btns[cam_id].setChecked(True)
        self._blocking = True
        self._chk_enabled.setChecked(p.get("enabled", True))
        yaw   = p.get("yaw",   0.0)
        pitch = p.get("pitch", 0.0)
        roll  = p.get("roll",  0.0)
        self._sl_yaw.setValue(int(yaw * 10));     self._sb_yaw.setValue(yaw)
        self._sl_pitch.setValue(int(pitch * 10)); self._sb_pitch.setValue(pitch)
        self._sl_roll.setValue(int(roll * 10));   self._sb_roll.setValue(roll)
        self._dsb_k1.setValue(p.get("k1", 0.0))
        self._dsb_k2.setValue(p.get("k2", 0.0))
        self._blocking = False

    def _collect_current(self):
        """Write widget values into _params for current cam."""
        key = f"cam{self._cam_id}"
        if key not in self._params:
            self._params[key] = {}
        p = self._params[key]
        p["yaw"]     = self._sb_yaw.value()
        p["pitch"]   = self._sb_pitch.value()
        p["roll"]    = self._sb_roll.value()
        p["k1"]      = self._dsb_k1.value()
        p["k2"]      = self._dsb_k2.value()
        p["enabled"] = self._chk_enabled.isChecked()

    def _on_ypr_changed(self):
        if self._blocking:
            return
        self._collect_current()
        self.params_changed.emit()

    def _on_lens_changed(self):
        if self._blocking:
            return
        self._collect_current()
        self.params_changed.emit()

    def _on_enabled(self, state: bool):
        if self._blocking:
            return
        key = f"cam{self._cam_id}"
        if key in self._params:
            self._params[key]["enabled"] = state
        self.params_changed.emit()

    def _reset_cam(self):
        from core.settings import DEFAULT_CYL_PARAMS
        key = f"cam{self._cam_id}"
        if key in DEFAULT_CYL_PARAMS:
            self._params[key] = copy.deepcopy(DEFAULT_CYL_PARAMS[key])
        self._load_cam(self._cam_id)
        self.params_changed.emit()

    def _reset_all(self):
        from core.settings import DEFAULT_CYL_PARAMS
        self._params = copy.deepcopy(DEFAULT_CYL_PARAMS)
        focal = self._params.get("focal", 1100.0)
        self._sb_focal.blockSignals(True)
        self._sb_focal.setValue(focal)
        self._sb_focal.blockSignals(False)
        self._load_cam(self._cam_id)
        self.params_changed.emit()


# ─────────────────────────────────────────────────────────────
# StitchCanvas  —  main widget
# ─────────────────────────────────────────────────────────────
class StitchCanvas(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: dict = {}
        self._worker: CylStitchWorker | None = None
        self._pending       = False      # recompute requested while worker is running
        self._pending_scale = PREVIEW_SCALE
        self._is_dragging   = False      # True while any slider or cam-icon is being dragged
        self._show_cam_labels = True     # toggle [C]
        self._show_bounds   = True
        self._show_feather  = True
        self._show_grid     = False
        self._last_cam_masks: dict = {}
        self._last_cx: float | None = None
        self._last_cy: float | None = None
        self._out_w = 7680
        self._out_h = 2160
        self._last_raw_rgb: np.ndarray | None = None   # stitch result before output color
        self._color_params: dict = default_color_params()
        self._preview_scale: float = PREVIEW_SCALE    # user-selected quality
        self._build_ui()
        self._load_params()
        self._load_resolution()
        self._load_color()
        self._load_preview_quality()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──
        tb_w = QWidget()
        tb_w.setFixedHeight(36)
        tb_w.setStyleSheet("background:#111111; border-bottom:1px solid #1e1e1e;")
        tb = QHBoxLayout(tb_w)
        tb.setContentsMargins(8, 4, 8, 4)
        tb.setSpacing(6)

        # View buttons
        btn_fit = QPushButton("[A] Fit All")
        btn_fit.setFixedHeight(24)
        btn_fit.clicked.connect(self._fit_all)
        tb.addWidget(btn_fit)

        tb.addSpacing(8)

        # Overlay toggles
        for key, label in (("bounds","[1] Edges"), ("feather","[2] Feather"), ("grid","[3] Grid")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key != "grid")
            btn.setFixedHeight(24)
            btn.toggled.connect(lambda v, k=key: self._on_overlay_toggled(k, v))
            tb.addWidget(btn)
            setattr(self, f"_btn_{key}", btn)

        # CAM Labels toggle
        self._btn_cam_labels = QPushButton("[C] CAM Labels")
        self._btn_cam_labels.setCheckable(True)
        self._btn_cam_labels.setChecked(True)
        self._btn_cam_labels.setFixedHeight(24)
        self._btn_cam_labels.setStyleSheet(
            "QPushButton { border:1px solid #333333; border-radius:3px; color:#888888; }"
            "QPushButton:checked { border-color:#00ff88; color:#00ff88; }"
        )
        self._btn_cam_labels.toggled.connect(self._on_cam_labels_toggled)
        tb.addWidget(self._btn_cam_labels)

        tb.addSpacing(12)

        btn_improve = QPushButton("Improve Offsets")
        btn_improve.setFixedHeight(24)
        btn_improve.clicked.connect(self._improve_offsets)
        tb.addWidget(btn_improve)

        tb.addSpacing(12)

        btn_save = QPushButton("[S] Save Calibration")
        btn_save.setFixedHeight(24)
        btn_save.clicked.connect(self._save_params)
        tb.addWidget(btn_save)

        # Presets menu
        btn_preset = QPushButton("Preset  ▼")
        btn_preset.setFixedHeight(24)
        btn_preset.clicked.connect(self._show_preset_menu)
        tb.addWidget(btn_preset)

        tb.addSpacing(8)

        # ── Preview quality selector ──
        lbl_qual = QLabel("Preview:")
        lbl_qual.setStyleSheet("color:#888888; font-size:9px;")
        tb.addWidget(lbl_qual)

        self._cmb_quality = QComboBox()
        self._cmb_quality.setFixedHeight(24)
        self._cmb_quality.setFixedWidth(122)
        self._cmb_quality.setStyleSheet("font-size:9px;")
        for qlabel, qscale in _QUALITY_PRESETS:
            self._cmb_quality.addItem(qlabel, qscale)
        self._cmb_quality.setCurrentIndex(1)   # default: Low 0.25
        self._cmb_quality.currentIndexChanged.connect(self._on_quality_changed)
        tb.addWidget(self._cmb_quality)

        tb.addSpacing(8)

        # ── Output resolution selector ──
        lbl_out = QLabel("Output:")
        lbl_out.setStyleSheet("color:#888888; font-size:9px;")
        tb.addWidget(lbl_out)

        self._cmb_res = QComboBox()
        self._cmb_res.setFixedHeight(24)
        self._cmb_res.setFixedWidth(210)
        self._cmb_res.setStyleSheet("font-size:9px;")
        for label, w, h in _RES_PRESETS:
            self._cmb_res.addItem(label, (w, h))
        self._cmb_res.currentIndexChanged.connect(self._on_res_preset_changed)
        tb.addWidget(self._cmb_res)

        self._sb_out_w = QSpinBox()
        self._sb_out_w.setRange(256, 32768)
        self._sb_out_w.setValue(7680)
        self._sb_out_w.setFixedWidth(68)
        self._sb_out_w.setFixedHeight(24)
        self._sb_out_w.setStyleSheet(_MONO)
        self._sb_out_w.setVisible(False)
        self._sb_out_w.editingFinished.connect(self._on_custom_res_changed)
        tb.addWidget(self._sb_out_w)

        self._lbl_res_x = QLabel("×")
        self._lbl_res_x.setStyleSheet("color:#888888; font-size:10px;")
        self._lbl_res_x.setVisible(False)
        tb.addWidget(self._lbl_res_x)

        self._sb_out_h = QSpinBox()
        self._sb_out_h.setRange(144, 16384)
        self._sb_out_h.setValue(2160)
        self._sb_out_h.setFixedWidth(62)
        self._sb_out_h.setFixedHeight(24)
        self._sb_out_h.setStyleSheet(_MONO)
        self._sb_out_h.setVisible(False)
        self._sb_out_h.editingFinished.connect(self._on_custom_res_changed)
        tb.addWidget(self._sb_out_h)

        tb.addStretch()

        self._lbl_status = QLabel("No frames")
        self._lbl_status.setStyleSheet("color:#444444; font-size:10px;")
        tb.addWidget(self._lbl_status)

        root.addWidget(tb_w)

        # ── Content splitter ──
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Canvas
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor(0x0a, 0x0a, 0x0a)))

        self._bg_item = QGraphicsPixmapItem()
        self._bg_item.setZValue(0)
        self._scene.addItem(self._bg_item)

        self._ov_item = QGraphicsPixmapItem()
        self._ov_item.setZValue(10)
        self._scene.addItem(self._ov_item)

        self._view = CanvasView(self._scene)
        splitter.addWidget(self._view)

        # Right panel: tab container (Params | Color)
        from ui.color_panel import ColorPanel
        self._params_panel = ParamPanel()
        self._params_panel.params_changed.connect(self._on_params_changed)

        self._color_panel = ColorPanel()
        self._color_panel.output_changed.connect(self._on_output_color_changed)
        self._color_panel.per_cam_changed.connect(self._on_per_cam_color_changed)
        self._color_panel.match_requested.connect(self._on_color_match_requested)

        right_tabs = QTabWidget()
        right_tabs.setFixedWidth(340)
        right_tabs.setStyleSheet(
            "QTabWidget::pane { border:none; background:#181818; }"
            "QTabBar::tab { background:#111111; color:#666666; font-size:9px;"
            "  padding:4px 10px; border:1px solid #1e1e1e; }"
            "QTabBar::tab:selected { background:#181818; color:#00ff88;"
            "  border-bottom:1px solid #181818; }"
        )
        right_tabs.addTab(self._params_panel, "Params")
        right_tabs.addTab(self._color_panel, "Color")
        splitter.addWidget(right_tabs)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 340])
        root.addWidget(splitter, stretch=1)

        # Initial placeholder scene so icons are visible before first stitch
        self._scene.setSceneRect(0, 0, 1280, 400)

        # Watermark: resolution label in top-right of canvas
        self._watermark = QGraphicsTextItem()
        self._watermark.setZValue(30)
        self._watermark.setDefaultTextColor(QColor(255, 255, 255, 110))
        wm_font = QFont("Consolas", 9)
        self._watermark.setFont(wm_font)
        self._scene.addItem(self._watermark)

        # Camera icons — one per camera, always present on canvas
        self._cam_icons: dict[int, CamIcon] = {}
        for cam_id in range(1, 6):
            icon = CamIcon(cam_id, _CAM_COLORS[cam_id])
            icon.clicked.connect(self._on_cam_icon_click)
            icon.drag_delta.connect(self._on_cam_drag)
            self._scene.addItem(icon)
            self._cam_icons[cam_id] = icon

        self._view.cam_dropped.connect(self._on_cam_drop)
        # Reposition icons whenever any parameter changes (e.g. yaw slider)
        self._params_panel.params_changed.connect(self._update_cam_icon_positions)
        # Icon drag-end → trigger full-quality stitch
        for icon in self._cam_icons.values():
            icon.drag_ended.connect(self._on_cam_drag_end)

        # Debounce timer: during slider drag, wait 80 ms then render DRAFT quality
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._on_debounce_fire)

        # Slider drag state from ParamPanel
        self._params_panel.slider_drag_started.connect(self._on_drag_started)
        self._params_panel.slider_drag_ended.connect(self._on_drag_ended)

    # ── keyboard shortcuts ────────────────────────────────────
    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k == Qt.Key.Key_1:
            self._btn_bounds.toggle()
        elif k == Qt.Key.Key_2:
            self._btn_feather.toggle()
        elif k == Qt.Key.Key_3:
            self._btn_grid.toggle()
        elif k == Qt.Key.Key_C:
            self._btn_cam_labels.toggle()
        elif k == Qt.Key.Key_A:
            self._fit_all()
        elif k == Qt.Key.Key_S:
            self._save_params()
        elif k == Qt.Key.Key_Space:
            self._on_params_changed()   # force recompute
        else:
            super().keyPressEvent(ev)

    # ── public ────────────────────────────────────────────────
    def set_frames(self, frames: dict):
        self._frames = frames
        self._on_params_changed()

    def set_frames_preview(self, frames: dict):
        """During playback, lay out 5 camera frames horizontally — skip stitching/overlays."""
        if not frames:
            return
        strips = []
        target_h = 360
        for cam_id in sorted(frames.keys()):
            rgb = frames[cam_id]
            h, w = rgb.shape[:2]
            if h == 0:
                continue
            new_w = int(w * target_h / h)
            resized = cv2.resize(rgb, (new_w, target_h),
                                 interpolation=cv2.INTER_LINEAR)
            strips.append(resized)
        if not strips:
            return
        composite = np.concatenate(strips, axis=1)
        ch, cw = composite.shape[:2]
        self._scene.setSceneRect(0, 0, cw, ch)
        self._bg_item.setPixmap(numpy_to_pixmap(composite))
        self._bg_item.setPos(0, 0)
        self._ov_item.setPixmap(QPixmap())   # hide overlay
        self._set_status(f"Playing  {len(frames)}/5  (Preview)", "#888888")

    def get_params(self) -> dict:
        return self._params_panel.get_cam_params()

    def get_feather_width(self) -> int:
        return self._params_panel.get_feather_width()

    def get_color_params(self) -> dict:
        return copy.deepcopy(self._color_params)

    # ── camera icon management ────────────────────────────────
    def _update_cam_icon_positions(self):
        """Reposition camera icons to match current yaw/pitch params."""
        if not hasattr(self, '_cam_icons'):
            return
        params  = self._params_panel.get_cam_params()
        focal_s = self._params_panel.get_focal() * self._preview_scale
        if focal_s == 0:
            return
        sw = self._scene.width()
        sh = self._scene.height()
        cx = self._last_cx if self._last_cx is not None else (sw / 2 if sw > 0 else 640)
        cy = self._last_cy if self._last_cy is not None else (sh / 2 if sh > 0 else 200)
        for cam_id, icon in self._cam_icons.items():
            p     = params.get(f"cam{cam_id}", {})
            yaw   = p.get("yaw",   0.0)
            pitch = p.get("pitch", 0.0)
            # Cylindrical projection: X = cx + focal * θ_rad (linear)
            x = cx + focal_s * math.radians(yaw)
            y = cy + focal_s * math.radians(pitch)
            icon.setPos(x, y)
            icon.setVisible(p.get("enabled", True) and self._show_cam_labels)

    def _on_cam_icon_click(self, cam_id: int):
        """Select camera — yellow border on icon, sync right panel."""
        self._params_panel._select_cam(cam_id)
        for cid, icon in self._cam_icons.items():
            icon.set_selected(cid == cam_id)

    def _on_cam_drag(self, cam_id: int, dx: float, dy: float):
        """Convert canvas drag delta → yaw/pitch update (with instant re-stitch)."""
        self._is_dragging = True   # ensure dragging flag is set
        focal_s = self._params_panel.get_focal() * self._preview_scale
        if focal_s == 0:
            return
        d_yaw   = math.degrees(dx / focal_s)
        d_pitch = math.degrees(dy / focal_s)
        params  = self._params_panel.get_cam_params()
        p       = params.get(f"cam{cam_id}", {})
        new_yaw   = max(-180.0, min(180.0, p.get("yaw",   0.0) + d_yaw))
        new_pitch = max(-90.0,  min(90.0,  p.get("pitch", 0.0) + d_pitch))
        # update_cam_yaw_pitch emits params_changed → triggers stitch + icon reposition
        self._params_panel.update_cam_yaw_pitch(cam_id, new_yaw, new_pitch)

    def _on_cam_drop(self, cam_id: int, sx: float, sy: float):
        """Place camera at drop position, converting scene coords → yaw/pitch."""
        focal_s = self._params_panel.get_focal() * self._preview_scale
        if focal_s == 0:
            return
        sw = self._scene.width()
        sh = self._scene.height()
        cx = self._last_cx if self._last_cx is not None else (sw / 2 if sw > 0 else 640)
        cy = self._last_cy if self._last_cy is not None else (sh / 2 if sh > 0 else 200)
        yaw   = max(-180.0, min(180.0, math.degrees((sx - cx) / focal_s)))
        pitch = max(-90.0,  min(90.0,  math.degrees((sy - cy) / focal_s)))
        self._params_panel.update_cam_yaw_pitch(cam_id, yaw, pitch)
        self._on_cam_icon_click(cam_id)

    def _on_cam_labels_toggled(self, val: bool):
        self._show_cam_labels = val
        for icon in self._cam_icons.values():
            if icon.isVisible() != val:
                icon.setVisible(val)
        # Force re-apply enabled state when turning back on
        if val:
            self._update_cam_icon_positions()

    # ── overlay ───────────────────────────────────────────────
    def _on_overlay_toggled(self, key: str, val: bool):
        setattr(self, f"_show_{key}", val)
        self._refresh_overlays()

    def _refresh_overlays(self):
        if not self._last_cam_masks:
            return
        focal_s = self._params_panel.get_focal() * self._preview_scale
        pw, ph = int(self._scene.width()), int(self._scene.height())
        if pw < 4 or ph < 4:
            return
        ov = render_overlays(
            self._last_cam_masks, pw, ph,
            self._show_bounds, self._show_feather, self._show_grid,
            focal_s, self._last_cx or 0.0, self._last_cy or 0.0,
        )
        self._ov_item.setPixmap(rgba_to_pixmap(ov))

    # ── drag state ────────────────────────────────────────────
    def _on_drag_started(self):
        self._is_dragging = True

    def _on_drag_ended(self):
        """Slider released or cam-icon dropped — fire full-quality stitch immediately."""
        self._is_dragging = False
        self._debounce.stop()
        if not self._frames:
            return
        if self._worker and self._worker.isRunning():
            self._pending       = True
            self._pending_scale = self._preview_scale
            self._worker.request_abort()
        else:
            self._start_worker(self._preview_scale)

    def _on_cam_drag_end(self, _cam_id: int):
        self._on_drag_ended()

    def _on_debounce_fire(self):
        """80 ms of slider inactivity → render DRAFT quality preview."""
        if not self._frames:
            return
        if self._worker and self._worker.isRunning():
            self._pending       = True
            self._pending_scale = DRAFT_SCALE
        else:
            self._start_worker(DRAFT_SCALE)

    # ── computation ───────────────────────────────────────────
    def _on_params_changed(self):
        if not self._frames:
            return
        if self._is_dragging:
            # During drag: abort running worker, restart debounce (80 ms)
            if self._worker and self._worker.isRunning():
                self._worker.request_abort()
            self._pending = False   # debounce will schedule the next run
            self._debounce.start()  # (re)start — fires 80 ms after last change
        else:
            # Not dragging (spinbox, keyboard, etc.) → immediate full quality
            self._debounce.stop()
            if self._worker and self._worker.isRunning():
                self._pending       = True
                self._pending_scale = self._preview_scale
                self._worker.request_abort()
                return
            self._start_worker(self._preview_scale)

    def _start_worker(self, scale: float = PREVIEW_SCALE):
        params  = self._params_panel.get_cam_params()
        focal   = self._params_panel.get_focal()
        params["focal"] = focal
        feather = self._params_panel.get_feather_width()

        self._worker = CylStitchWorker(
            frames        = self._frames,
            cam_params    = params,
            focal         = focal,
            feather_width = feather,
            scale         = scale,
            color_params  = self._color_params,
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(
            lambda e: (self._set_status(f"Error: {e}", "#ff4444"),
                       self._check_pending()))
        self._worker.start()
        label = "Draft..." if scale < self._preview_scale else "Computing..."
        self._set_status(label, "#aaaaaa")

    def _on_result(self, rgb: np.ndarray, cam_masks: dict,
                   cx_p: float, cy_p: float):
        # Store pre-output-color result for fast re-apply without re-stitch
        self._last_raw_rgb = rgb.copy()
        # Apply output color correction
        out_p = self._color_params.get("output", {})
        display_rgb = apply_output_color(rgb, out_p) if out_p else rgb
        h, w = display_rgb.shape[:2]
        self._scene.setSceneRect(0, 0, w, h)
        self._bg_item.setPixmap(numpy_to_pixmap(display_rgb))
        self._bg_item.setPos(0, 0)
        self._ov_item.setPos(0, 0)
        self._last_cam_masks = cam_masks
        self._last_cx = cx_p
        self._last_cy = cy_p
        self._refresh_overlays()
        self._update_cam_icon_positions()
        self._update_watermark()
        scale = self._worker.scale if self._worker else self._preview_scale
        tag   = " [Draft]" if scale < self._preview_scale else ""
        self._set_status(f"Done  {w}×{h}px  (×{scale:.2f}){tag}", "#00ff88")
        self._check_pending()

    def _check_pending(self):
        if self._pending:
            self._pending = False
            scale, self._pending_scale = self._pending_scale, self._preview_scale
            self._start_worker(scale)

    # ── fit / zoom ────────────────────────────────────────────
    def _fit_all(self):
        self._view.fit_all()

    # ── calibration save ──────────────────────────────────────
    def _save_params(self):
        params = self._params_panel.get_cam_params()
        params["focal"] = self._params_panel.get_focal()
        save_cyl_params(params)
        save_color_params(self._color_params)
        self._set_status("Calibration saved", "#00ff88")
        self.status_changed.emit("Calibration saved")

    def _load_params(self):
        params = load_cyl_params()
        self._params_panel.load_external(params)
        self._update_cam_icon_positions()

    # ── color ─────────────────────────────────────────────────
    def _load_color(self):
        try:
            self._color_params = load_color_params()
        except Exception:
            self._color_params = default_color_params()
        if hasattr(self, "_color_panel"):
            self._color_panel.load_params(self._color_params)

    def _on_output_color_changed(self, color_params: dict):
        """Output slider changed — re-apply to cached raw RGB without re-stitching."""
        self._color_params = color_params
        if self._last_raw_rgb is not None:
            out_p = self._color_params.get("output", {})
            display_rgb = apply_output_color(self._last_raw_rgb, out_p)
            self._bg_item.setPixmap(numpy_to_pixmap(display_rgb))

    def _on_per_cam_color_changed(self, color_params: dict):
        """Per-cam slider changed — trigger DRAFT re-stitch."""
        self._color_params = color_params
        if not self._frames:
            return
        if self._worker and self._worker.isRunning():
            self._pending       = True
            self._pending_scale = DRAFT_SCALE
            self._worker.request_abort()
        else:
            self._start_worker(DRAFT_SCALE)

    def _on_color_match_requested(self):
        """Auto-match all cameras to CAM3 reference."""
        if not self._frames:
            self._set_status("No frames — load video first", "#ff4444")
            return
        from core.color import match_colors_to_ref
        matched = match_colors_to_ref(self._frames, ref_cam_id=3)
        for cam_id, pc_params in matched.items():
            self._color_params["per_cam"][cam_id] = pc_params
        if hasattr(self, "_color_panel"):
            self._color_panel.load_params(self._color_params)
        if not self._frames:
            return
        if self._worker and self._worker.isRunning():
            self._pending       = True
            self._pending_scale = DRAFT_SCALE
            self._worker.request_abort()
        else:
            self._start_worker(DRAFT_SCALE)

    # ── improve offsets ───────────────────────────────────────
    def _improve_offsets(self):
        if not self._frames:
            self._set_status("No frames", "#ff4444")
            return
        self._set_status("Improve Offsets Computing...", "#aaaaaa")
        params = self._params_panel.get_cam_params()
        focal  = self._params_panel.get_focal()
        try:
            updated = improve_yaw(self._frames, params, focal, ADJ_PAIRS)
            self._params_panel.set_cam_params(updated)
            self._set_status("Improve Offsets complete", "#00ff88")
        except Exception as e:
            self._set_status(f"Improve error: {e}", "#ff4444")

    # ── presets ───────────────────────────────────────────────
    def _show_preset_menu(self):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background:#1e1e1e; color:#cccccc; }"
                           "QMenu::item:selected { background:#2a2a2a; }")
        act_save = menu.addAction("Save current params as preset...")
        menu.addSeparator()
        presets = list_presets()
        load_acts = {}
        del_acts  = {}
        for name in presets:
            la = menu.addAction(f"▶  {name}")
            load_acts[la] = name
        if presets:
            menu.addSeparator()
            for name in presets:
                da = menu.addAction(f"✕  Delete {name}")
                del_acts[da] = name
        chosen = menu.exec(self.mapToGlobal(
            self._view.geometry().topRight()))
        if chosen is None:
            return
        if chosen == act_save:
            name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
            if ok and name.strip():
                params = self._params_panel.get_cam_params()
                params["focal"] = self._params_panel.get_focal()
                save_preset(name.strip(), params)
                self._set_status(f"Preset '{name}' saved", "#00ff88")
        elif chosen in load_acts:
            name   = load_acts[chosen]
            params = load_preset(name)
            if params:
                self._params_panel.load_external(params)
                self._set_status(f"Preset '{name}' loaded", "#00ff88")
        elif chosen in del_acts:
            name = del_acts[chosen]
            delete_preset(name)
            self._set_status(f"Preset '{name}' deleted", "#888888")

    # ── output resolution ─────────────────────────────────────
    def _load_resolution(self):
        """앱 시작시 저장된 출력 해상도를 로드하여 UI에 반영."""
        res = load_output_resolution()
        w, h = res["width"], res["height"]
        self._out_w = w
        self._out_h = h
        # 프리셋 매칭 시도
        for i, (_, pw, ph) in enumerate(_RES_PRESETS):
            if pw == w and ph == h:
                self._cmb_res.blockSignals(True)
                self._cmb_res.setCurrentIndex(i)
                self._cmb_res.blockSignals(False)
                self._sb_out_w.setVisible(False)
                self._lbl_res_x.setVisible(False)
                self._sb_out_h.setVisible(False)
                return
        # 매칭되는 프리셋 없음 → Custom
        self._cmb_res.blockSignals(True)
        self._cmb_res.setCurrentIndex(len(_RES_PRESETS) - 1)
        self._cmb_res.blockSignals(False)
        self._sb_out_w.setValue(w)
        self._sb_out_h.setValue(h)
        self._sb_out_w.setVisible(True)
        self._lbl_res_x.setVisible(True)
        self._sb_out_h.setVisible(True)

    def _on_res_preset_changed(self, idx: int):
        """프리셋 드롭다운 변경 핸들러."""
        _, w, h = _RES_PRESETS[idx]
        is_custom = (w == 0 and h == 0)
        self._sb_out_w.setVisible(is_custom)
        self._lbl_res_x.setVisible(is_custom)
        self._sb_out_h.setVisible(is_custom)
        if not is_custom:
            self._out_w = w
            self._out_h = h
            save_output_resolution(w, h)
            self._update_watermark()

    def _on_custom_res_changed(self):
        """Custom 해상도 스핀박스 확정 핸들러."""
        w = self._sb_out_w.value()
        h = self._sb_out_h.value()
        self._out_w = w
        self._out_h = h
        save_output_resolution(w, h)
        self._update_watermark()

    def _update_watermark(self):
        """캔버스 우상단에 현재 출력 해상도 워터마크 업데이트."""
        self._watermark.setPlainText(f"{self._out_w} × {self._out_h}")
        sr = self._scene.sceneRect()
        br = self._watermark.boundingRect()
        margin = 8
        self._watermark.setPos(
            sr.right() - br.width() - margin,
            sr.top() + margin,
        )

    # ── preview quality ───────────────────────────────────────
    def _load_preview_quality(self):
        import json
        from core.settings import CAL_DIR
        f = CAL_DIR / "preview_quality.json"
        scale = PREVIEW_SCALE
        if f.exists():
            try:
                scale = float(json.loads(f.read_text("utf-8")).get("scale", PREVIEW_SCALE))
            except Exception:
                pass
        self._preview_scale = scale
        for i, (_, s) in enumerate(_QUALITY_PRESETS):
            if abs(s - scale) < 0.001:
                self._cmb_quality.blockSignals(True)
                self._cmb_quality.setCurrentIndex(i)
                self._cmb_quality.blockSignals(False)
                return

    def _on_quality_changed(self, idx: int):
        _, scale = _QUALITY_PRESETS[idx]
        self._preview_scale = scale
        import json
        from core.settings import CAL_DIR
        (CAL_DIR / "preview_quality.json").write_text(
            json.dumps({"scale": scale}, indent=2), "utf-8"
        )
        if not self._frames:
            return
        if self._worker and self._worker.isRunning():
            self._pending       = True
            self._pending_scale = scale
            self._worker.request_abort()
        else:
            self._start_worker(scale)

    def _set_status(self, msg: str, color: str = "#888888"):
        self._lbl_status.setText(msg)
        self._lbl_status.setStyleSheet(f"color:{color}; font-size:10px;")
