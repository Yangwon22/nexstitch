"""
인터랙티브 스티칭 캔버스 뷰
QGraphicsScene 기반 — 5개 카메라 레이어 + 실시간 스티칭 프리뷰 배경
"""
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGraphicsScene, QGraphicsView,
    QGraphicsObject, QGraphicsPixmapItem, QGraphicsRectItem,
    QLabel, QSlider, QSpinBox, QPushButton, QGroupBox,
    QGridLayout, QCheckBox, QSizePolicy, QFrame, QStyleOptionGraphicsItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPixmap, QColor, QPen, QBrush, QPainter, QFont,
)

from ui.utils import numpy_to_pixmap, fit_pixmap
from core.settings import CAM_ROLES
from core.calibration import load_canvas_layout, save_canvas_layout

# ── Constants ────────────────────────────────────────────────
DISP_SCALE   = 0.10        # scene px = canvas px * DISP_SCALE
CANVAS_ORDER = [3, 2, 1, 4, 5]

_COLORS = {
    1: QColor(0, 200, 255, 200),
    2: QColor(100, 255, 100, 200),
    3: QColor(255, 160, 0,   200),
    4: QColor(255, 80,  80,  200),
    5: QColor(200, 100, 255, 200),
}
_DEFAULT_ADJ = {
    "dx": 0.0, "dy": 0.0, "rotation": 0.0,
    "scale": 1.0, "brightness": 0.0, "opacity": 0.8,
}
_MONO = "font-family:'JetBrains Mono','Consolas',monospace; font-size:10px; color:#00ff88;"


# ── Camera layer ─────────────────────────────────────────────
class CamLayer(QGraphicsObject):
    """Draggable camera frame item on the canvas."""
    drag_delta = pyqtSignal(int, float, float)  # cam_id, ddx_scene, ddy_scene
    clicked    = pyqtSignal(int)

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id     = cam_id
        self._pm: QPixmap | None = None
        self._bw        = 160
        self._bh        = 90
        self._selected  = False
        self._opacity_v = 0.8
        self._drag_last: QPointF | None = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setZValue(cam_id)          # stacking by cam_id
        self.setAcceptHoverEvents(True)

    # ── public ───────────────────────────────────────────────
    def set_frame(self, rgb: np.ndarray, disp_scale: float):
        fh, fw = rgb.shape[:2]
        tw, th = max(4, int(fw * disp_scale)), max(4, int(fh * disp_scale))
        thumb  = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        self._pm = numpy_to_pixmap(thumb)
        self._bw = tw
        self._bh = th
        self.prepareGeometryChange()
        self.update()

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()

    def set_opacity_val(self, v: float):
        self._opacity_v = max(0.1, min(1.0, v))
        self.update()

    # ── QGraphicsItem interface ───────────────────────────────
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._bw, self._bh)

    def paint(self, painter: QPainter,
              option: QStyleOptionGraphicsItem, widget=None):
        # Thumbnail
        if self._pm:
            painter.setOpacity(self._opacity_v)
            painter.drawPixmap(0, 0, self._pm)
        painter.setOpacity(1.0)

        # Border
        color = _COLORS.get(self.cam_id, QColor(180, 180, 180))
        if self._selected:
            pen = QPen(QColor(255, 221, 0), 2)
        else:
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5))

        # Label
        lbl_color = QColor(255, 221, 0) if self._selected else color
        painter.setPen(QPen(lbl_color))
        painter.setFont(QFont("Rajdhani", 9, QFont.Weight.Bold))
        role = CAM_ROLES.get(self.cam_id, "")
        painter.drawText(QRectF(3, 2, self._bw - 6, 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                         f"CAM{self.cam_id} {role}")

    # ── drag ─────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_last = ev.scenePos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.clicked.emit(self.cam_id)
            ev.accept()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self._drag_last is not None:
            delta = ev.scenePos() - self._drag_last
            self._drag_last = ev.scenePos()
            self.moveBy(delta.x(), delta.y())
            self.drag_delta.emit(self.cam_id, delta.x(), delta.y())
            ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_last = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept()


# ── Graphics view with zoom + pan ────────────────────────────
class StitchView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor(0x0d, 0x0d, 0x0d)))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panning     = False
        self._pan_last    = None
        self._space_held  = False

    def fit_all(self):
        self.fitInView(self.scene().sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, ev):
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif ev.key() == Qt.Key.Key_0:
            self.fit_all()
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        is_pan = (ev.button() == Qt.MouseButton.MiddleButton or
                  (ev.button() == Qt.MouseButton.LeftButton and self._space_held))
        if is_pan:
            self._panning  = True
            self._pan_last = ev.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_last is not None:
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
                Qt.CursorShape.OpenHandCursor if self._space_held
                else Qt.CursorShape.ArrowCursor
            )
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)


# ── Right adjustment panel ───────────────────────────────────
class CanvasAdjustPanel(QWidget):
    changed        = pyqtSignal(int, dict)
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
        lay.setSpacing(5)

        self._title = QLabel("카메라를 선택하세요")
        self._title.setStyleSheet("color:#00ff88; font-size:11px; font-weight:bold;")
        lay.addWidget(self._title)

        # Position
        gb_pos = QGroupBox("위치 (px)")
        pg = QGridLayout(gb_pos)
        pg.setSpacing(3)
        pg.addWidget(QLabel("X"), 0, 0)
        self._spin_dx = QSpinBox(); self._spin_dx.setRange(-10000, 10000)
        self._spin_dx.valueChanged.connect(self._emit)
        pg.addWidget(self._spin_dx, 0, 1)
        pg.addWidget(QLabel("Y"), 1, 0)
        self._spin_dy = QSpinBox(); self._spin_dy.setRange(-10000, 10000)
        self._spin_dy.valueChanged.connect(self._emit)
        pg.addWidget(self._spin_dy, 1, 1)
        lay.addWidget(gb_pos)

        # Rotation + Scale
        gb_rs = QGroupBox("회전 / 스케일")
        rs = QGridLayout(gb_rs); rs.setSpacing(3)

        rs.addWidget(QLabel("회전°"), 0, 0)
        self._sld_rot = QSlider(Qt.Orientation.Horizontal)
        self._sld_rot.setRange(-300, 300)
        self._lbl_rot = QLabel("0.0°"); self._lbl_rot.setStyleSheet(_MONO)
        self._sld_rot.valueChanged.connect(
            lambda v: (self._lbl_rot.setText(f"{v/10:.1f}°"), self._emit()))
        rs.addWidget(self._sld_rot, 0, 1); rs.addWidget(self._lbl_rot, 0, 2)

        rs.addWidget(QLabel("스케일"), 1, 0)
        self._sld_sc = QSlider(Qt.Orientation.Horizontal)
        self._sld_sc.setRange(50, 200); self._sld_sc.setValue(100)
        self._lbl_sc = QLabel("1.00×"); self._lbl_sc.setStyleSheet(_MONO)
        self._sld_sc.valueChanged.connect(
            lambda v: (self._lbl_sc.setText(f"{v/100:.2f}×"), self._emit()))
        rs.addWidget(self._sld_sc, 1, 1); rs.addWidget(self._lbl_sc, 1, 2)
        lay.addWidget(gb_rs)

        # Brightness
        gb_br = QGroupBox("밝기 보정")
        br = QVBoxLayout(gb_br); br.setSpacing(2)
        self._sld_br = QSlider(Qt.Orientation.Horizontal)
        self._sld_br.setRange(-100, 100)
        self._lbl_br = QLabel("0"); self._lbl_br.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_br.setStyleSheet(_MONO)
        self._sld_br.valueChanged.connect(
            lambda v: (self._lbl_br.setText(str(v)), self._emit()))
        br.addWidget(self._sld_br); br.addWidget(self._lbl_br)
        lay.addWidget(gb_br)

        # Opacity (display only — blending visualization)
        gb_op = QGroupBox("레이어 투명도")
        op = QVBoxLayout(gb_op); op.setSpacing(2)
        self._sld_op = QSlider(Qt.Orientation.Horizontal)
        self._sld_op.setRange(10, 100); self._sld_op.setValue(80)
        self._lbl_op = QLabel("80%"); self._lbl_op.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_op.setStyleSheet(_MONO)
        self._sld_op.valueChanged.connect(
            lambda v: (self._lbl_op.setText(f"{v}%"), self._emit()))
        op.addWidget(self._sld_op); op.addWidget(self._lbl_op)
        lay.addWidget(gb_op)

        btn_reset = QPushButton("이 카메라 리셋")
        btn_reset.clicked.connect(self._reset)
        lay.addWidget(btn_reset)

        btn_save = QPushButton("저장  (canvas_layout.json)")
        btn_save.setObjectName("accent")
        btn_save.clicked.connect(self.save_requested)
        lay.addWidget(btn_save)

        lay.addStretch()

    # ── public ───────────────────────────────────────────────
    def load_cam(self, cam_id: int, params: dict):
        self._cam_id   = cam_id
        self._blocking = True
        self._title.setText(f"CAM{cam_id}  {CAM_ROLES.get(cam_id, '')}")
        self._spin_dx.setValue(int(params.get("dx", 0)))
        self._spin_dy.setValue(int(params.get("dy", 0)))
        self._sld_rot.setValue(int(params.get("rotation", 0.0) * 10))
        self._sld_sc.setValue(int(params.get("scale", 1.0) * 100))
        self._sld_br.setValue(int(params.get("brightness", 0.0)))
        self._sld_op.setValue(int(params.get("opacity", 0.8) * 100))
        self._blocking = False
        self.setEnabled(True)

    def apply_drag(self, ddx: float, ddy: float):
        self._spin_dx.setValue(int(self._spin_dx.value() + ddx))
        self._spin_dy.setValue(int(self._spin_dy.value() + ddy))

    def current_params(self) -> dict:
        return {
            "dx":         float(self._spin_dx.value()),
            "dy":         float(self._spin_dy.value()),
            "rotation":   self._sld_rot.value() / 10.0,
            "scale":      self._sld_sc.value() / 100.0,
            "brightness": float(self._sld_br.value()),
            "opacity":    self._sld_op.value() / 100.0,
        }

    def _emit(self):
        if self._blocking or self._cam_id is None:
            return
        self.changed.emit(self._cam_id, self.current_params())

    def _reset(self):
        if self._cam_id is None:
            return
        self._blocking = True
        self._spin_dx.setValue(0); self._spin_dy.setValue(0)
        self._sld_rot.setValue(0); self._sld_sc.setValue(100)
        self._sld_br.setValue(0);  self._sld_op.setValue(80)
        self._blocking = False
        self._emit()


# ── Main interactive canvas view ─────────────────────────────
class InteractiveCanvasView(QWidget):
    stitch_requested = pyqtSignal()
    layout_changed   = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout: dict  = {str(i): _DEFAULT_ADJ.copy() for i in range(1, 6)}
        self._layout["_crop"] = {"enabled": False, "ratio": 0}

        self._homo_data: dict | None = None
        self._frames:    dict        = {}
        self._frame_w    = 0
        self._frame_h    = 0
        self._disp_scale = DISP_SCALE
        self._selected_cam: int | None = None
        self._show_overlaps = True
        self._overlap_items: list = []
        self._layers: dict[int, CamLayer] = {}

        self._quick_preview = True   # show quick composite on drag

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.stitch_requested)

        self._build_ui()
        self._load_saved()

    # ── build ─────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Canvas side ───────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet("background:#1a1a1a; border-bottom:1px solid #2a2a2a;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(8, 2, 8, 2)
        tb.setSpacing(6)

        btn_fit = QPushButton("전체 보기  [0]")
        btn_fit.setFixedHeight(24)
        btn_fit.clicked.connect(self._fit_view)
        tb.addWidget(btn_fit)

        btn_reset_all = QPushButton("전체 리셋")
        btn_reset_all.setFixedHeight(24)
        btn_reset_all.clicked.connect(self._reset_all)
        tb.addWidget(btn_reset_all)

        tb.addSpacing(12)

        self._btn_ov = QPushButton("오버랩  ON")
        self._btn_ov.setCheckable(True)
        self._btn_ov.setChecked(True)
        self._btn_ov.setFixedHeight(24)
        self._btn_ov.toggled.connect(self._on_overlap_toggled)
        tb.addWidget(self._btn_ov)

        tb.addSpacing(12)

        lbl_c = QLabel("크롭:")
        lbl_c.setStyleSheet("color:#888888; font-size:10px;")
        tb.addWidget(lbl_c)

        self._chk_crop = QCheckBox("활성화")
        self._chk_crop.toggled.connect(self._on_crop_changed)
        tb.addWidget(self._chk_crop)

        self._sld_crop = QSlider(Qt.Orientation.Horizontal)
        self._sld_crop.setRange(0, 30)
        self._sld_crop.setFixedWidth(80)
        self._sld_crop.valueChanged.connect(self._on_crop_changed)
        tb.addWidget(self._sld_crop)

        self._lbl_crop = QLabel("0%")
        self._lbl_crop.setFixedWidth(28)
        self._lbl_crop.setStyleSheet(_MONO)
        tb.addWidget(self._lbl_crop)

        tb.addSpacing(12)

        self._btn_quick = QPushButton("빠른 미리보기  ON")
        self._btn_quick.setCheckable(True)
        self._btn_quick.setChecked(True)
        self._btn_quick.setFixedHeight(24)
        self._btn_quick.toggled.connect(self._on_quick_toggled)
        tb.addWidget(self._btn_quick)

        tb.addStretch()

        self._lbl_status = QLabel("카메라 클릭 후 드래그로 위치 조정  |  휠: 줌  |  스페이스+드래그: 패닝")
        self._lbl_status.setStyleSheet("color:#555555; font-size:10px;")
        tb.addWidget(self._lbl_status)

        left_lay.addWidget(toolbar)

        # Scene + View
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor(0x0d, 0x0d, 0x0d)))

        self._bg_item = QGraphicsPixmapItem()
        self._bg_item.setZValue(-10)
        self._scene.addItem(self._bg_item)

        self._view = StitchView(self._scene)
        left_lay.addWidget(self._view, stretch=1)

        root.addWidget(left, stretch=1)

        # ── Right panel ───────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#3a3a3a;")
        root.addWidget(sep)

        self._adj = CanvasAdjustPanel()
        self._adj.setFixedWidth(210)
        self._adj.changed.connect(self._on_adj_changed)
        self._adj.save_requested.connect(self.save_layout)
        root.addWidget(self._adj)

        # Create camera layers
        for cam_id in range(1, 6):
            layer = CamLayer(cam_id)
            layer.drag_delta.connect(self._on_drag_delta)
            layer.clicked.connect(self._on_layer_clicked)
            self._layers[cam_id] = layer
            self._scene.addItem(layer)

        self._layout_cameras_default()

    # ── public ────────────────────────────────────────────────
    def set_frames(self, frames: dict):
        self._frames = frames
        if frames:
            s = next(iter(frames.values()))
            self._frame_h, self._frame_w = s.shape[:2]
        self._update_thumbnails()
        if self._homo_data:
            self._rebuild_from_homo()
        else:
            self._quick_composite()

    def set_homo_data(self, homo_data: dict | None):
        self._homo_data = homo_data
        if homo_data:
            self._rebuild_from_homo()
        else:
            self._layout_cameras_default()

    def set_stitch_result(self, rgb: np.ndarray):
        sw, sh = int(self._scene.width()), int(self._scene.height())
        if sw < 4 or sh < 4:
            return
        h, w = rgb.shape[:2]
        scaled = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # Apply crop visualization
        crop = self._layout.get("_crop", {})
        if crop.get("enabled") and crop.get("ratio", 0) > 0:
            ratio = crop["ratio"] / 100.0
            cr    = int(scaled.shape[0] * ratio)
            if cr > 0:
                scaled[:cr, :] = 0
                scaled[-cr:, :] = 0

        self._bg_item.setPixmap(numpy_to_pixmap(scaled))
        self._lbl_status.setText(f"스티칭 완료  {w}×{h}px")

    def set_status(self, msg: str):
        self._lbl_status.setText(msg)

    def get_layout(self) -> dict:
        return dict(self._layout)

    def save_layout(self):
        save_canvas_layout(self._layout)

    # ── layout / position ─────────────────────────────────────
    def _layout_cameras_default(self):
        """Horizontal layout when no calibration data."""
        self._update_thumbnails()
        if self._frames:
            s      = next(iter(self._frames.values()))
            fh, fw = s.shape[:2]
            tw     = max(160, int(fw * self._disp_scale))
            th     = max(90,  int(fh * self._disp_scale))
        else:
            tw, th = 160, 90

        x = 10
        for cam_id in CANVAS_ORDER:
            adj = self._layout.get(str(cam_id), _DEFAULT_ADJ.copy())
            ox  = adj.get("dx", 0.0) * self._disp_scale
            oy  = adj.get("dy", 0.0) * self._disp_scale
            self._layers[cam_id].setPos(x + ox, 10 + oy)
            x  += tw + 10

        self._scene.setSceneRect(0, 0, x, th + 20)
        self._update_overlaps()
        self._quick_composite()
        QTimer.singleShot(150, self._fit_view)

    def _rebuild_from_homo(self):
        """Position layers based on H_abs + canvas_layout offsets."""
        if not self._homo_data or not self._frames:
            self._layout_cameras_default()
            return

        canvas = self._homo_data["canvas"]
        sw = canvas["out_w"] * self._disp_scale
        sh = canvas["out_h"] * self._disp_scale
        self._scene.setSceneRect(0, 0, sw, sh)
        self._bg_item.setPos(0, 0)

        self._update_thumbnails()

        for cam_id in range(1, 6):
            key = str(cam_id)
            if key not in self._homo_data["H_abs"]:
                continue
            H   = np.array(self._homo_data["H_abs"][key], np.float64)
            adj = self._layout.get(key, _DEFAULT_ADJ.copy())
            self._layers[cam_id].setPos(
                *self._cam_scene_pos(H, canvas, adj)
            )

        self._update_overlaps()
        self._quick_composite()
        QTimer.singleShot(150, self._fit_view)

    def _cam_scene_pos(self, H: np.ndarray, canvas: dict,
                       adj: dict) -> tuple[float, float]:
        """Top-left scene position for camera given its H and offsets."""
        tl_h  = H @ np.array([0.0, 0.0, 1.0])
        tl    = tl_h[:2] / tl_h[2]
        sx    = (tl[0] - canvas["x_min"] + adj.get("dx", 0.0)) * self._disp_scale
        sy    = (tl[1] - canvas["y_min"] + adj.get("dy", 0.0)) * self._disp_scale
        return sx, sy

    def _reposition_one(self, cam_id: int):
        if not self._homo_data:
            return
        canvas = self._homo_data["canvas"]
        key    = str(cam_id)
        if key not in self._homo_data["H_abs"]:
            return
        H   = np.array(self._homo_data["H_abs"][key], np.float64)
        adj = self._layout.get(key, _DEFAULT_ADJ.copy())
        self._layers[cam_id].setPos(*self._cam_scene_pos(H, canvas, adj))

    def _update_thumbnails(self):
        for cam_id, layer in self._layers.items():
            if cam_id in self._frames:
                layer.set_frame(self._frames[cam_id], self._disp_scale)

    def _update_overlaps(self):
        for it in self._overlap_items:
            self._scene.removeItem(it)
        self._overlap_items.clear()
        if not self._show_overlaps:
            return
        for i in range(len(CANVAS_ORDER) - 1):
            a, b = CANVAS_ORDER[i], CANVAS_ORDER[i + 1]
            if a not in self._layers or b not in self._layers:
                continue
            ra = self._layers[a].mapToScene(self._layers[a].boundingRect()).boundingRect()
            rb = self._layers[b].mapToScene(self._layers[b].boundingRect()).boundingRect()
            inter = ra.intersected(rb)
            if inter.isValid() and inter.width() > 1 and inter.height() > 1:
                ov = QGraphicsRectItem(inter)
                ov.setBrush(QBrush(QColor(0, 200, 80, 55)))
                ov.setPen(QPen(QColor(0, 220, 100, 140), 1))
                ov.setZValue(20)
                self._scene.addItem(ov)
                self._overlap_items.append(ov)

    def _quick_composite(self):
        """즉시 합성 — 워핑 없이 썸네일을 현재 위치에 겹쳐 그림.
        캘리브레이션 없이도 즉각적인 배치 피드백을 제공."""
        if not self._frames or not self._quick_preview:
            return
        sw = int(self._scene.width())
        sh = int(self._scene.height())
        if sw < 4 or sh < 4:
            return

        canvas = np.zeros((sh, sw, 3), np.uint8)

        for cam_id in CANVAS_ORDER:
            if cam_id not in self._frames or cam_id not in self._layers:
                continue
            layer = self._layers[cam_id]
            pos   = layer.pos()
            x, y  = int(pos.x()), int(pos.y())
            tw, th = layer._bw, layer._bh
            if tw < 1 or th < 1:
                continue

            # Scaled thumbnail (cached in layer, but re-use frame for accuracy)
            frame = self._frames[cam_id]
            thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            # Per-cam brightness
            br = self._layout.get(str(cam_id), {}).get("brightness", 0.0)
            if br != 0.0:
                thumb = np.clip(thumb.astype(np.int16) + int(br), 0, 255).astype(np.uint8)

            # Clip to canvas bounds
            x1 = max(0, x);   y1 = max(0, y)
            x2 = min(sw, x + tw); y2 = min(sh, y + th)
            if x2 <= x1 or y2 <= y1:
                continue
            sx1 = x1 - x;  sy1 = y1 - y
            sx2 = sx1 + (x2 - x1);  sy2 = sy1 + (y2 - y1)

            opacity = self._layout.get(str(cam_id), {}).get("opacity", 0.8)
            src = thumb[sy1:sy2, sx1:sx2].astype(np.float32)
            dst = canvas[y1:y2, x1:x2].astype(np.float32)
            # Where canvas is black (not yet painted) use full opacity
            has_content = dst.max(axis=2) > 0
            alpha = np.where(has_content[:, :, np.newaxis], opacity, 1.0)
            canvas[y1:y2, x1:x2] = (
                dst * (1.0 - alpha) + src * alpha
            ).clip(0, 255).astype(np.uint8)

        self._bg_item.setPixmap(numpy_to_pixmap(canvas))

    def _fit_view(self):
        self._view.fitInView(self._scene.sceneRect(),
                             Qt.AspectRatioMode.KeepAspectRatio)

    def _reset_all(self):
        for cam_id in range(1, 6):
            self._layout[str(cam_id)] = _DEFAULT_ADJ.copy()
        if self._homo_data:
            self._rebuild_from_homo()
        else:
            self._layout_cameras_default()
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    # ── slots ─────────────────────────────────────────────────
    def _on_layer_clicked(self, cam_id: int):
        if self._selected_cam is not None and self._selected_cam in self._layers:
            self._layers[self._selected_cam].set_selected(False)
        self._selected_cam = cam_id
        self._layers[cam_id].set_selected(True)
        self._adj.load_cam(cam_id, self._layout.get(str(cam_id), _DEFAULT_ADJ.copy()))

    def _on_drag_delta(self, cam_id: int, ddx_s: float, ddy_s: float):
        """Called on every mouse-move during drag (scene-space delta)."""
        adj = self._layout.setdefault(str(cam_id), _DEFAULT_ADJ.copy())
        adj["dx"] = adj.get("dx", 0.0) + ddx_s / self._disp_scale
        adj["dy"] = adj.get("dy", 0.0) + ddy_s / self._disp_scale
        # Keep layer opacity in sync
        self._layers[cam_id].set_opacity_val(adj.get("opacity", 0.8))
        # Update adj panel spinboxes without full reload (avoid fighting the drag)
        if cam_id == self._selected_cam:
            self._adj._blocking = True
            self._adj._spin_dx.setValue(int(adj["dx"]))
            self._adj._spin_dy.setValue(int(adj["dy"]))
            self._adj._blocking = False
        self._update_overlaps()
        self._quick_composite()          # ← 즉시 미리보기 갱신
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _on_adj_changed(self, cam_id: int, params: dict):
        self._layout[str(cam_id)] = params
        self._layers[cam_id].set_opacity_val(params.get("opacity", 0.8))
        self._reposition_one(cam_id)
        self._update_overlaps()
        self._quick_composite()          # ← 즉시 미리보기 갱신
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _on_quick_toggled(self, checked: bool):
        self._quick_preview = checked
        self._btn_quick.setText(f"빠른 미리보기  {'ON' if checked else 'OFF'}")
        if checked:
            self._quick_composite()

    def _on_overlap_toggled(self, checked: bool):
        self._show_overlaps = checked
        self._btn_ov.setText(f"오버랩  {'ON' if checked else 'OFF'}")
        self._update_overlaps()

    def _on_crop_changed(self):
        ratio   = self._sld_crop.value()
        enabled = self._chk_crop.isChecked()
        self._lbl_crop.setText(f"{ratio}%")
        self._layout["_crop"] = {"enabled": enabled, "ratio": ratio}
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _load_saved(self):
        loaded = load_canvas_layout()
        if loaded:
            self._layout = loaded
            # Restore opacity on layers
            for cam_id, layer in self._layers.items():
                op = self._layout.get(str(cam_id), {}).get("opacity", 0.8)
                layer.set_opacity_val(op)
        crop = self._layout.get("_crop", {})
        self._chk_crop.setChecked(crop.get("enabled", False))
        self._sld_crop.setValue(int(crop.get("ratio", 0)))
