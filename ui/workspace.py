"""
NexStitch  ─  Workspace UI
Canvas + camera panel + adjustment bar + action bar
"""
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QLabel, QSlider, QSpinBox, QPushButton, QFrame,
    QProgressBar, QDialog, QTextEdit, QComboBox,
    QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QMimeData, QPoint
from PyQt6.QtGui import QDrag, QColor, QPen, QBrush

from ui.utils import numpy_to_pixmap
from ui.canvas_view import (
    CamLayer, StitchView,
    DISP_SCALE, CANVAS_ORDER, _DEFAULT_ADJ, _COLORS,
)
from core.settings import CAM_ROLES, OUTPUT_DIR
from core.calibration import load_canvas_layout, save_canvas_layout
from ui.workers import RenderWorker

_MONO = "font-family:'JetBrains Mono','Consolas',monospace; font-size:10px; color:#00ff88;"


# ─────────────────────────────────────────────────────────────
# DraggableThumb
# ─────────────────────────────────────────────────────────────
class DraggableThumb(QFrame):
    """Camera thumbnail — drag to canvas to place."""
    selected = pyqtSignal(int)

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id          = cam_id
        self._placed         = False
        self._rgb: np.ndarray | None = None
        self._drag_start: QPoint | None = None
        self.setFixedSize(170, 100)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)

        self._lbl_img = QLabel()
        self._lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_img.setFixedSize(164, 82)
        self._lbl_img.setStyleSheet("background:#111111;")
        lay.addWidget(self._lbl_img)

        role = CAM_ROLES.get(cam_id, "")
        self._lbl_name = QLabel(f"CAM{cam_id}  {role}")
        self._lbl_name.setStyleSheet("color:#666666; font-size:9px;")
        lay.addWidget(self._lbl_name)

        self._apply_style()

    def set_frame(self, rgb: np.ndarray):
        self._rgb = rgb
        thumb = cv2.resize(rgb, (164, 82), interpolation=cv2.INTER_AREA)
        self._lbl_img.setPixmap(numpy_to_pixmap(thumb))

    def set_placed(self, placed: bool):
        self._placed = placed
        self._apply_style()
        role = CAM_ROLES.get(self.cam_id, "")
        self._lbl_name.setText(
            f"CAM{self.cam_id}  ✓ Placed" if placed
            else f"CAM{self.cam_id}  {role}"
        )

    def _apply_style(self):
        c = "#00ff88" if self._placed else "#2a2a2a"
        self.setStyleSheet(
            f"DraggableThumb {{ background:#1a1a1a; border:1px solid {c}; border-radius:3px; }}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start = ev.pos()
            self.selected.emit(self.cam_id)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if (self._drag_start is not None
                and (ev.pos() - self._drag_start).manhattanLength() > 6
                and ev.buttons() & Qt.MouseButton.LeftButton):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(str(self.cam_id))
            drag.setMimeData(mime)
            if self._rgb is not None:
                fh, fw = self._rgb.shape[:2]
                tw, th = 120, max(1, int(120 * fh / fw))
                thumb  = cv2.resize(self._rgb, (tw, th), interpolation=cv2.INTER_AREA)
                drag.setPixmap(numpy_to_pixmap(thumb))
                drag.setHotSpot(QPoint(tw // 2, th // 2))
            drag.exec(Qt.DropAction.CopyAction)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._drag_start = None
        super().mouseReleaseEvent(ev)


# ─────────────────────────────────────────────────────────────
# CameraPanel
# ─────────────────────────────────────────────────────────────
class CameraPanel(QWidget):
    camera_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(182)
        self.setStyleSheet("background:#161616; border-left:1px solid #2a2a2a;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(6)

        hdr = QLabel("CAMERAS")
        hdr.setStyleSheet("color:#444444; font-size:8px; letter-spacing:3px; font-weight:bold;")
        lay.addWidget(hdr)

        self._thumbs: dict[int, DraggableThumb] = {}
        for cam_id in range(1, 6):
            t = DraggableThumb(cam_id)
            t.selected.connect(self.camera_selected)
            lay.addWidget(t)
            self._thumbs[cam_id] = t

        lay.addStretch()

    def set_frame(self, cam_id: int, rgb: np.ndarray):
        if cam_id in self._thumbs:
            self._thumbs[cam_id].set_frame(rgb)

    def set_placed(self, cam_id: int, placed: bool):
        if cam_id in self._thumbs:
            self._thumbs[cam_id].set_placed(placed)

    def set_selected(self, cam_id: int):
        for cid, t in self._thumbs.items():
            placed = t._placed
            if cid == cam_id:
                c = "#00aaff"
            elif placed:
                c = "#00ff88"
            else:
                c = "#2a2a2a"
            t.setStyleSheet(
                f"DraggableThumb {{ background:#1a1a1a; border:1px solid {c}; border-radius:3px; }}"
            )


# ─────────────────────────────────────────────────────────────
# WorkspaceView  (StitchView + drag-and-drop target)
# ─────────────────────────────────────────────────────────────
class WorkspaceView(StitchView):
    camera_dropped = pyqtSignal(int, QPointF)

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasText():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        ev.acceptProposedAction()

    def dropEvent(self, ev):
        try:
            cam_id = int(ev.mimeData().text())
        except (ValueError, TypeError):
            return
        scene_pos = self.mapToScene(ev.position().toPoint())
        self.camera_dropped.emit(cam_id, scene_pos)
        ev.acceptProposedAction()


# ─────────────────────────────────────────────────────────────
# WorkspaceCanvas
# ─────────────────────────────────────────────────────────────
class WorkspaceCanvas(QWidget):
    camera_selected    = pyqtSignal(int)
    cam_params_changed = pyqtSignal(int, float, float)   # cam_id, new_dx, new_dy
    stitch_requested   = pyqtSignal()
    layout_changed     = pyqtSignal(dict)
    cam_placed         = pyqtSignal(int, bool)           # cam_id, placed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout:   dict = {str(i): _DEFAULT_ADJ.copy() for i in range(1, 6)}
        self._layout["_crop"] = {"enabled": False, "ratio": 0}
        self._homo_data: dict | None  = None
        self._frames:    dict         = {}
        self._placed:    set          = set()
        self._selected_cam: int | None = None
        self._show_overlaps = True
        self._overlap_items: list     = []
        self._layers:    dict[int, CamLayer] = {}
        self._quick_preview = True

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.stitch_requested)

        self._build_ui()
        self._load_saved()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Toolbar ──
        tb_w = QWidget()
        tb_w.setFixedHeight(30)
        tb_w.setStyleSheet("background:#181818; border-bottom:1px solid #252525;")
        tb = QHBoxLayout(tb_w)
        tb.setContentsMargins(8, 2, 8, 2)
        tb.setSpacing(6)

        btn_fit = QPushButton("[0] Fit All")
        btn_fit.setFixedHeight(22)
        btn_fit.clicked.connect(self._fit_view)
        tb.addWidget(btn_fit)

        btn_reset = QPushButton("Reset")
        btn_reset.setFixedHeight(22)
        btn_reset.clicked.connect(self._reset_all)
        tb.addWidget(btn_reset)

        tb.addSpacing(8)

        self._btn_ov = QPushButton("Overlap ●")
        self._btn_ov.setCheckable(True)
        self._btn_ov.setChecked(True)
        self._btn_ov.setFixedHeight(22)
        self._btn_ov.toggled.connect(self._on_overlap_toggled)
        tb.addWidget(self._btn_ov)

        self._btn_qp = QPushButton("즉시 미리보기 ●")
        self._btn_qp.setCheckable(True)
        self._btn_qp.setChecked(True)
        self._btn_qp.setFixedHeight(22)
        self._btn_qp.toggled.connect(self._on_quick_toggled)
        tb.addWidget(self._btn_qp)

        tb.addStretch()

        self._lbl_status = QLabel("카메라 패널에서 캔버스로 드래그하여 배치  │  휠: 줌  │  Space+드래그: 패닝")
        self._lbl_status.setStyleSheet("color:#3a3a3a; font-size:10px;")
        tb.addWidget(self._lbl_status)

        lay.addWidget(tb_w)

        # ── Scene ──
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor(0x0d, 0x0d, 0x0d)))
        self._scene.setSceneRect(0, 0, 1920, 1080)

        self._bg_item = QGraphicsPixmapItem()
        self._bg_item.setZValue(-10)
        self._scene.addItem(self._bg_item)

        self._view = WorkspaceView(self._scene)
        self._view.camera_dropped.connect(self._on_camera_dropped)
        lay.addWidget(self._view, stretch=1)

        # Create layers (hidden until placed)
        for cam_id in range(1, 6):
            layer = CamLayer(cam_id)
            layer.drag_delta.connect(self._on_drag_delta)
            layer.clicked.connect(self._on_layer_clicked)
            self._layers[cam_id] = layer
            self._scene.addItem(layer)
            layer.setVisible(False)

    # ── public ────────────────────────────────────────────────
    def set_frames(self, frames: dict):
        self._frames = frames
        self._update_thumbnails()
        if self._homo_data:
            self._place_all_from_homo()
        elif self._placed:
            self._quick_composite()

    def set_homo_data(self, homo_data: dict | None):
        self._homo_data = homo_data
        if homo_data:
            self._place_all_from_homo()

    def set_stitch_result(self, rgb: np.ndarray):
        sw, sh = int(self._scene.width()), int(self._scene.height())
        if sw < 4 or sh < 4:
            return
        scaled = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        self._bg_item.setPixmap(numpy_to_pixmap(scaled))
        h, w = rgb.shape[:2]
        self._lbl_status.setText(f"스티칭 완료  {w}×{h}px")

    def set_status(self, msg: str):
        self._lbl_status.setText(msg)

    def get_layout(self) -> dict:
        return dict(self._layout)

    def get_cam_params(self, cam_id: int) -> dict:
        return dict(self._layout.get(str(cam_id), _DEFAULT_ADJ.copy()))

    def set_cam_params(self, cam_id: int, params: dict):
        self._layout[str(cam_id)] = params
        self._layers[cam_id].set_opacity_val(params.get("opacity", 0.8))
        self._reposition_one(cam_id)
        self._update_overlaps()
        self._quick_composite()
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def save_layout(self):
        save_canvas_layout(self._layout)

    def select_camera(self, cam_id: int):
        self._on_layer_clicked(cam_id)

    def get_placed(self) -> set:
        return set(self._placed)

    def get_placed_positions(self) -> dict:
        """Returns {cam_id: (canvas_x, canvas_y)} in actual canvas pixels."""
        result = {}
        for cam_id in self._placed:
            layer = self._layers[cam_id]
            result[cam_id] = (
                layer.pos().x() / DISP_SCALE,
                layer.pos().y() / DISP_SCALE,
            )
        return result

    # ── placement ─────────────────────────────────────────────
    def _place_all_from_homo(self):
        if not self._homo_data or not self._frames:
            return
        canvas = self._homo_data["canvas"]
        sw = canvas["out_w"] * DISP_SCALE
        sh = canvas["out_h"] * DISP_SCALE
        self._scene.setSceneRect(0, 0, sw, sh)
        self._bg_item.setPos(0, 0)
        self._update_thumbnails()
        for cam_id in range(1, 6):
            key = str(cam_id)
            if key not in self._homo_data.get("H_abs", {}):
                continue
            H   = np.array(self._homo_data["H_abs"][key], np.float64)
            adj = self._layout.get(key, _DEFAULT_ADJ.copy())
            sx, sy = self._cam_scene_pos(H, canvas, adj)
            self._layers[cam_id].setPos(sx, sy)
            self._layers[cam_id].setVisible(True)
            self._placed.add(cam_id)
            self.cam_placed.emit(cam_id, True)
        self._update_overlaps()
        self._quick_composite()
        QTimer.singleShot(150, self._fit_view)

    def _on_camera_dropped(self, cam_id: int, scene_pos: QPointF):
        layer = self._layers[cam_id]
        if self._frames and cam_id in self._frames:
            s = self._frames[cam_id]
            fh, fw = s.shape[:2]
            tw = max(160, int(fw * DISP_SCALE))
            th = max(90,  int(fh * DISP_SCALE))
        else:
            tw, th = 160, 90
        sx = scene_pos.x() - tw / 2
        sy = scene_pos.y() - th / 2
        layer.setPos(sx, sy)
        layer.setVisible(True)
        self._placed.add(cam_id)

        # Update dx/dy relative to calibrated base position
        adj = self._layout.setdefault(str(cam_id), _DEFAULT_ADJ.copy())
        if self._homo_data:
            canvas = self._homo_data["canvas"]
            key = str(cam_id)
            if key in self._homo_data.get("H_abs", {}):
                H = np.array(self._homo_data["H_abs"][key], np.float64)
                base_sx, base_sy = self._cam_scene_pos(H, canvas, {"dx": 0, "dy": 0})
                adj["dx"] = (sx - base_sx) / DISP_SCALE
                adj["dy"] = (sy - base_sy) / DISP_SCALE
        else:
            adj["dx"] = sx / DISP_SCALE
            adj["dy"] = sy / DISP_SCALE

        self.cam_placed.emit(cam_id, True)
        self._select_layer(cam_id)
        self.camera_selected.emit(cam_id)
        self._update_overlaps()
        self._quick_composite()
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _cam_scene_pos(self, H: np.ndarray, canvas: dict,
                       adj: dict) -> tuple[float, float]:
        tl_h = H @ np.array([0.0, 0.0, 1.0])
        tl   = tl_h[:2] / tl_h[2]
        sx   = (tl[0] - canvas["x_min"] + adj.get("dx", 0.0)) * DISP_SCALE
        sy   = (tl[1] - canvas["y_min"] + adj.get("dy", 0.0)) * DISP_SCALE
        return sx, sy

    def _reposition_one(self, cam_id: int):
        if cam_id not in self._placed:
            return
        adj = self._layout.get(str(cam_id), _DEFAULT_ADJ.copy())
        if self._homo_data:
            key = str(cam_id)
            if key not in self._homo_data.get("H_abs", {}):
                return
            H      = np.array(self._homo_data["H_abs"][key], np.float64)
            canvas = self._homo_data["canvas"]
            self._layers[cam_id].setPos(*self._cam_scene_pos(H, canvas, adj))
        else:
            self._layers[cam_id].setPos(
                adj.get("dx", 0.0) * DISP_SCALE,
                adj.get("dy", 0.0) * DISP_SCALE,
            )

    def _update_thumbnails(self):
        for cam_id, layer in self._layers.items():
            if cam_id in self._frames:
                layer.set_frame(self._frames[cam_id], DISP_SCALE)

    def _update_overlaps(self):
        for it in self._overlap_items:
            self._scene.removeItem(it)
        self._overlap_items.clear()
        if not self._show_overlaps:
            return
        visible = [c for c in CANVAS_ORDER if c in self._placed]
        for i in range(len(visible) - 1):
            a, b = visible[i], visible[i + 1]
            if a not in self._layers or b not in self._layers:
                continue
            if not self._layers[a].isVisible() or not self._layers[b].isVisible():
                continue
            ra = self._layers[a].mapToScene(self._layers[a].boundingRect()).boundingRect()
            rb = self._layers[b].mapToScene(self._layers[b].boundingRect()).boundingRect()
            inter = ra.intersected(rb)
            if inter.isValid() and inter.width() > 1 and inter.height() > 1:
                ov = QGraphicsRectItem(inter)
                ov.setBrush(QBrush(QColor(0, 200, 80, 50)))
                ov.setPen(QPen(QColor(0, 220, 100, 130), 1))
                ov.setZValue(20)
                self._scene.addItem(ov)
                self._overlap_items.append(ov)

    def _quick_composite(self):
        if not self._frames or not self._quick_preview or not self._placed:
            return
        sw, sh = int(self._scene.width()), int(self._scene.height())
        if sw < 4 or sh < 4:
            return
        canvas_img = np.zeros((sh, sw, 3), np.uint8)
        for cam_id in CANVAS_ORDER:
            if cam_id not in self._placed or cam_id not in self._frames:
                continue
            layer = self._layers[cam_id]
            if not layer.isVisible():
                continue
            pos    = layer.pos()
            x, y   = int(pos.x()), int(pos.y())
            tw, th = layer._bw, layer._bh
            if tw < 1 or th < 1:
                continue
            frame = self._frames[cam_id]
            thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            br = self._layout.get(str(cam_id), {}).get("brightness", 0.0)
            if br != 0.0:
                thumb = np.clip(thumb.astype(np.int16) + int(br), 0, 255).astype(np.uint8)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(sw, x + tw), min(sh, y + th)
            if x2 <= x1 or y2 <= y1:
                continue
            sx1, sy1 = x1 - x, y1 - y
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            opacity = self._layout.get(str(cam_id), {}).get("opacity", 0.8)
            src = thumb[sy1:sy2, sx1:sx2].astype(np.float32)
            dst = canvas_img[y1:y2, x1:x2].astype(np.float32)
            has_content = dst.max(axis=2) > 0
            alpha = np.where(has_content[:, :, np.newaxis], opacity, 1.0)
            canvas_img[y1:y2, x1:x2] = (
                dst * (1.0 - alpha) + src * alpha
            ).clip(0, 255).astype(np.uint8)
        self._bg_item.setPixmap(numpy_to_pixmap(canvas_img))

    def _fit_view(self):
        self._view.fitInView(self._scene.sceneRect(),
                             Qt.AspectRatioMode.KeepAspectRatio)

    def _reset_all(self):
        for cam_id in range(1, 6):
            self._layout[str(cam_id)] = _DEFAULT_ADJ.copy()
        if self._homo_data:
            self._place_all_from_homo()
        else:
            for cam_id in self._placed:
                self._reposition_one(cam_id)
            self._quick_composite()
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _select_layer(self, cam_id: int):
        if self._selected_cam is not None and self._selected_cam in self._layers:
            self._layers[self._selected_cam].set_selected(False)
        self._selected_cam = cam_id
        if cam_id in self._layers:
            self._layers[cam_id].set_selected(True)

    # ── slots ─────────────────────────────────────────────────
    def _on_layer_clicked(self, cam_id: int):
        self._select_layer(cam_id)
        self.camera_selected.emit(cam_id)

    def _on_drag_delta(self, cam_id: int, ddx_s: float, ddy_s: float):
        adj = self._layout.setdefault(str(cam_id), _DEFAULT_ADJ.copy())
        adj["dx"] = adj.get("dx", 0.0) + ddx_s / DISP_SCALE
        adj["dy"] = adj.get("dy", 0.0) + ddy_s / DISP_SCALE
        self._layers[cam_id].set_opacity_val(adj.get("opacity", 0.8))
        self.cam_params_changed.emit(cam_id, adj["dx"], adj["dy"])
        self._update_overlaps()
        self._quick_composite()
        self.layout_changed.emit(self._layout)
        self._debounce.start()

    def _on_overlap_toggled(self, checked: bool):
        self._show_overlaps = checked
        self._btn_ov.setText(f"오버랩 {'●' if checked else '○'}")
        self._update_overlaps()

    def _on_quick_toggled(self, checked: bool):
        self._quick_preview = checked
        self._btn_qp.setText(f"즉시 미리보기 {'●' if checked else '○'}")
        if checked:
            self._quick_composite()

    def _load_saved(self):
        loaded = load_canvas_layout()
        if loaded:
            self._layout = loaded
            for cam_id, layer in self._layers.items():
                op = self._layout.get(str(cam_id), {}).get("opacity", 0.8)
                layer.set_opacity_val(op)


# ─────────────────────────────────────────────────────────────
# AdjustBar
# ─────────────────────────────────────────────────────────────
class AdjustBar(QWidget):
    changed = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self.setStyleSheet("background:#1a1a1a; border-top:1px solid #252525;")
        self._cam_id: int | None = None
        self._blocking = False
        self._build_ui()
        self.setEnabled(False)

    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(10)

        self._lbl_cam = QLabel("─")
        self._lbl_cam.setFixedWidth(120)
        self._lbl_cam.setStyleSheet("color:#00ff88; font-size:11px; font-weight:bold;")
        lay.addWidget(self._lbl_cam)

        def _sep():
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setStyleSheet("color:#2a2a2a;")
            return s

        lay.addWidget(_sep())

        def _spinbox(label, lo, hi):
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#555555; font-size:9px;")
            sb  = QSpinBox()
            sb.setRange(lo, hi)
            sb.setFixedWidth(72)
            sb.setStyleSheet(_MONO)
            return lbl, sb

        lx, self._sp_dx = _spinbox("X", -20000, 20000)
        lay.addWidget(lx); lay.addWidget(self._sp_dx)
        ly_, self._sp_dy = _spinbox("Y", -20000, 20000)
        lay.addWidget(ly_); lay.addWidget(self._sp_dy)
        self._sp_dx.valueChanged.connect(self._emit)
        self._sp_dy.valueChanged.connect(self._emit)

        lay.addWidget(_sep())

        def _slider(label, lo, hi, w=90):
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#555555; font-size:9px;")
            sl  = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setFixedWidth(w)
            val = QLabel("0")
            val.setFixedWidth(38)
            val.setStyleSheet(_MONO)
            return lbl, sl, val

        lrot, self._sl_rot, self._lbl_rot = _slider("회전°", -300, 300)
        for w in (lrot, self._sl_rot, self._lbl_rot):
            lay.addWidget(w)
        self._sl_rot.valueChanged.connect(
            lambda v: (self._lbl_rot.setText(f"{v/10:.1f}°"), self._emit()))

        lsc, self._sl_sc, self._lbl_sc = _slider("스케일", 50, 200)
        self._sl_sc.setValue(100)
        for w in (lsc, self._sl_sc, self._lbl_sc):
            lay.addWidget(w)
        self._sl_sc.valueChanged.connect(
            lambda v: (self._lbl_sc.setText(f"{v/100:.2f}×"), self._emit()))

        lbr, self._sl_br, self._lbl_br = _slider("밝기", -100, 100)
        for w in (lbr, self._sl_br, self._lbl_br):
            lay.addWidget(w)
        self._sl_br.valueChanged.connect(
            lambda v: (self._lbl_br.setText(str(v)), self._emit()))

        lay.addWidget(_sep())

        btn_reset = QPushButton("리셋")
        btn_reset.setFixedHeight(26)
        btn_reset.clicked.connect(self._reset)
        lay.addWidget(btn_reset)

        lay.addStretch()

    def load_cam(self, cam_id: int, params: dict):
        self._cam_id   = cam_id
        self._blocking = True
        role = CAM_ROLES.get(cam_id, "")
        self._lbl_cam.setText(f"CAM{cam_id}  {role}")
        self._sp_dx.setValue(int(params.get("dx", 0)))
        self._sp_dy.setValue(int(params.get("dy", 0)))
        self._sl_rot.setValue(int(params.get("rotation", 0.0) * 10))
        self._sl_sc.setValue(int(params.get("scale", 1.0) * 100))
        self._sl_br.setValue(int(params.get("brightness", 0.0)))
        self._blocking = False
        self.setEnabled(True)

    def apply_drag(self, new_dx: float, new_dy: float):
        """Called on canvas drag — update X/Y without emitting changed."""
        self._blocking = True
        self._sp_dx.setValue(int(new_dx))
        self._sp_dy.setValue(int(new_dy))
        self._blocking = False

    def current_params(self) -> dict:
        return {
            "dx":         float(self._sp_dx.value()),
            "dy":         float(self._sp_dy.value()),
            "rotation":   self._sl_rot.value() / 10.0,
            "scale":      self._sl_sc.value() / 100.0,
            "brightness": float(self._sl_br.value()),
            "opacity":    0.8,
        }

    def _emit(self):
        if self._blocking or self._cam_id is None:
            return
        self.changed.emit(self._cam_id, self.current_params())

    def _reset(self):
        if self._cam_id is None:
            return
        self._blocking = True
        self._sp_dx.setValue(0); self._sp_dy.setValue(0)
        self._sl_rot.setValue(0); self._sl_sc.setValue(100); self._sl_br.setValue(0)
        self._blocking = False
        self._emit()


# ─────────────────────────────────────────────────────────────
# ActionBar
# ─────────────────────────────────────────────────────────────
class ActionBar(QWidget):
    stitch_requested = pyqtSignal()
    save_requested   = pyqtSignal()
    render_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setStyleSheet("background:#111111; border-top:1px solid #1e1e1e;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        self._btn_stitch = QPushButton("▶  스티칭 미리보기  [F9]")
        self._btn_stitch.setObjectName("accent")
        self._btn_stitch.setFixedHeight(30)
        self._btn_stitch.clicked.connect(self.stitch_requested)
        lay.addWidget(self._btn_stitch)

        self._btn_save = QPushButton("캘리브레이션 저장")
        self._btn_save.setFixedHeight(30)
        self._btn_save.clicked.connect(self.save_requested)
        lay.addWidget(self._btn_save)

        self._btn_render = QPushButton("렌더  →  ProRes")
        self._btn_render.setFixedHeight(30)
        self._btn_render.clicked.connect(self.render_requested)
        lay.addWidget(self._btn_render)

        lay.addSpacing(16)

        self._lbl_status = QLabel("준비")
        self._lbl_status.setStyleSheet("color:#555555; font-size:10px;")
        lay.addWidget(self._lbl_status, stretch=1)

    def set_status(self, msg: str, color: str = "#888888"):
        self._lbl_status.setText(msg)
        self._lbl_status.setStyleSheet(f"color:{color}; font-size:10px;")


# ─────────────────────────────────────────────────────────────
# RenderDialog
# ─────────────────────────────────────────────────────────────
class RenderDialog(QDialog):
    def __init__(self, mov_files, homo_data, lens_params, cam_masks,
                 canvas_layout, fps: float = 24.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("렌더  →  ProRes")
        self.setMinimumWidth(540)
        self._mov_files     = mov_files
        self._homo_data     = homo_data
        self._lens_params   = lens_params
        self._cam_masks     = cam_masks
        self._canvas_layout = canvas_layout
        self._fps           = fps
        self._worker: RenderWorker | None = None
        self._frames_dir    = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # Frame range
        fr = QHBoxLayout()
        fr.addWidget(QLabel("시작 프레임"))
        self._sp_start = QSpinBox(); self._sp_start.setRange(0, 99999)
        fr.addWidget(self._sp_start)
        fr.addWidget(QLabel("끝 프레임"))
        self._sp_end = QSpinBox(); self._sp_end.setRange(0, 99999); self._sp_end.setValue(99)
        fr.addWidget(self._sp_end)
        fr.addStretch()
        lay.addLayout(fr)

        # Settings
        sr = QHBoxLayout()
        sr.addWidget(QLabel("스케일"))
        self._cb_scale = QComboBox()
        for s in ["25%", "50%", "75%", "100%"]:
            self._cb_scale.addItem(s, int(s[:-1]))
        self._cb_scale.setCurrentIndex(3)
        sr.addWidget(self._cb_scale)

        sr.addWidget(QLabel("블렌딩"))
        self._cb_blend = QComboBox()
        self._cb_blend.addItems(["feather", "hard_cut"])
        sr.addWidget(self._cb_blend)

        sr.addWidget(QLabel("포맷"))
        self._cb_fmt = QComboBox()
        self._cb_fmt.addItems(["JPEG (고화질)", "PNG (무손실)"])
        sr.addWidget(self._cb_fmt)
        sr.addStretch()
        lay.addLayout(sr)

        # Buttons
        br = QHBoxLayout()
        self._btn_start = QPushButton("렌더 시작")
        self._btn_start.setObjectName("accent")
        self._btn_start.clicked.connect(self._start)
        br.addWidget(self._btn_start)

        self._btn_abort = QPushButton("중단")
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._abort)
        br.addWidget(self._btn_abort)
        br.addStretch()
        lay.addLayout(br)

        self._progress = QProgressBar()
        lay.addWidget(self._progress)

        self._lbl_prog = QLabel("")
        self._lbl_prog.setStyleSheet("color:#888888; font-size:10px;")
        lay.addWidget(self._lbl_prog)

        self._txt_cmd = QTextEdit()
        self._txt_cmd.setReadOnly(True)
        self._txt_cmd.setFixedHeight(110)
        self._txt_cmd.setStyleSheet(
            "background:#0d0d0d; color:#00ff88;"
            "font-family:'JetBrains Mono','Consolas',monospace; font-size:10px;"
        )
        self._txt_cmd.setPlaceholderText("렌더 완료 후 ffmpeg ProRes 커맨드가 여기 표시됩니다...")
        lay.addWidget(self._txt_cmd)

    def _start(self):
        if not self._mov_files or not self._homo_data:
            self._lbl_prog.setText("오류: 소스 파일 또는 캘리브레이션이 없습니다.")
            return
        from core.video import get_video_info
        info = get_video_info(str(self._mov_files[0]))
        self._fps = info.get("fps", 24.0)

        self._frames_dir = OUTPUT_DIR / "frames"
        self._frames_dir.mkdir(exist_ok=True)

        self._worker = RenderWorker(
            mov_files   = self._mov_files,
            homo_data   = self._homo_data,
            lens_params = self._lens_params,
            cam_masks   = self._cam_masks,
            blend       = self._cb_blend.currentText(),
            scale       = self._cb_scale.currentData() / 100.0,
            start_frame = self._sp_start.value(),
            end_frame   = self._sp_end.value(),
            intensity   = 0.0,
            brightness  = 0.0,
            contrast    = 1.0,
            out_fmt     = self._cb_fmt.currentText(),
            jpeg_q      = 95,
            frames_dir  = self._frames_dir,
            rig_layout  = self._canvas_layout,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._btn_start.setEnabled(False)
        self._btn_abort.setEnabled(True)
        self._progress.setValue(0)
        self._worker.start()

    def _abort(self):
        if self._worker:
            self._worker.abort()
        self._btn_abort.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._lbl_prog.setText(msg)

    def _on_finished(self, out_dir: str):
        self._btn_start.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._progress.setValue(100)
        self._lbl_prog.setText(f"렌더 완료 → {out_dir}")
        ext         = "jpg" if "JPEG" in self._cb_fmt.currentText() else "png"
        fps         = round(self._fps, 3)
        frames_path = str(self._frames_dir / f"frame_%06d.{ext}").replace("\\", "/")
        out_path    = str(OUTPUT_DIR / "stitch.mov").replace("\\", "/")
        cmd = (
            f"ffmpeg -framerate {fps} \\\n"
            f'  -i "{frames_path}" \\\n'
            f"  -c:v prores_ks -profile:v 3 \\\n"
            f"  -pix_fmt yuv422p10le -vendor apl0 \\\n"
            f'  -an "{out_path}"'
        )
        self._txt_cmd.setPlainText(cmd)

    def _on_error(self, msg: str):
        self._btn_start.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._lbl_prog.setText(f"오류: {msg}")
        self._lbl_prog.setStyleSheet("color:#ff4444; font-size:10px;")


# ─────────────────────────────────────────────────────────────
# Workspace  (top-level container)
# ─────────────────────────────────────────────────────────────
class Workspace(QWidget):
    stitch_requested = pyqtSignal()
    layout_changed   = pyqtSignal(dict)
    save_requested   = pyqtSignal()
    render_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._wire()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1)

        self._canvas    = WorkspaceCanvas()
        self._cam_panel = CameraPanel()
        h_split.addWidget(self._canvas)
        h_split.addWidget(self._cam_panel)
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 0)
        root.addWidget(h_split, stretch=1)

        self._adj_bar    = AdjustBar()
        self._action_bar = ActionBar()
        root.addWidget(self._adj_bar)
        root.addWidget(self._action_bar)

    def _wire(self):
        self._canvas.cam_placed.connect(self._cam_panel.set_placed)
        self._canvas.camera_selected.connect(self._on_canvas_cam_selected)
        self._canvas.cam_params_changed.connect(
            lambda cam_id, dx, dy: self._adj_bar.apply_drag(dx, dy)
        )
        self._cam_panel.camera_selected.connect(self._on_panel_cam_selected)
        self._adj_bar.changed.connect(self._canvas.set_cam_params)
        self._action_bar.stitch_requested.connect(self.stitch_requested)
        self._action_bar.save_requested.connect(self.save_requested)
        self._action_bar.render_requested.connect(self.render_requested)
        self._canvas.layout_changed.connect(self.layout_changed)

    def _on_canvas_cam_selected(self, cam_id: int):
        params = self._canvas.get_cam_params(cam_id)
        self._adj_bar.load_cam(cam_id, params)
        self._cam_panel.set_selected(cam_id)

    def _on_panel_cam_selected(self, cam_id: int):
        self._cam_panel.set_selected(cam_id)
        if cam_id in self._canvas.get_placed():
            self._canvas.select_camera(cam_id)
            params = self._canvas.get_cam_params(cam_id)
            self._adj_bar.load_cam(cam_id, params)

    # ── public ────────────────────────────────────────────────
    def set_frames(self, frames: dict):
        self._canvas.set_frames(frames)
        for cam_id, rgb in frames.items():
            self._cam_panel.set_frame(cam_id, rgb)

    def set_homo_data(self, homo_data: dict | None):
        self._canvas.set_homo_data(homo_data)

    def set_stitch_result(self, rgb: np.ndarray):
        self._canvas.set_stitch_result(rgb)

    def set_status(self, msg: str, color: str = "#888888"):
        self._action_bar.set_status(msg, color)

    def get_layout(self) -> dict:
        return self._canvas.get_layout()

    def save_canvas_layout(self):
        self._canvas.save_layout()

    def get_placed_positions(self) -> dict:
        return self._canvas.get_placed_positions()

    def get_placed(self) -> set:
        return self._canvas.get_placed()

    @property
    def canvas(self) -> WorkspaceCanvas:
        return self._canvas
