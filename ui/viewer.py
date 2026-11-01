import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap

from ui.utils import numpy_to_pixmap, fit_pixmap
from ui.multicam_view import MultiCamView
from ui.canvas_view import InteractiveCanvasView


class ViewerLabel(QLabel):
    """이미지 표시 + 마스크 드로잉 지원 레이블."""
    mask_stroke = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 240)
        self.setStyleSheet("background-color: #111111; border: 1px solid #3a3a3a;")
        self._mask_mode   = False
        self._brush_size  = 30
        self._erase_mode  = False
        self._drawing     = False
        self._last_pt: QPoint | None = None
        self._base_rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    # ── public API ────────────────────────────────────────────
    def show_image(self, rgb: np.ndarray):
        self._base_rgb = rgb
        if self._mask is None or self._mask.shape[:2] != rgb.shape[:2]:
            self._mask = np.zeros(rgb.shape[:2], np.uint8)
        self._refresh()

    def set_mask(self, mask: np.ndarray | None):
        self._mask = mask
        self._refresh()

    def get_mask(self) -> np.ndarray | None:
        return self._mask

    def set_mask_mode(self, enabled: bool):
        self._mask_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_brush_size(self, size: int):
        self._brush_size = size

    def set_erase_mode(self, erase: bool):
        self._erase_mode = erase

    def clear_mask(self):
        if self._base_rgb is not None:
            self._mask = np.zeros(self._base_rgb.shape[:2], np.uint8)
            self._refresh()

    # ── mouse ─────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if self._mask_mode and ev.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last_pt = ev.position().toPoint()
            self._draw_at(self._last_pt)

    def mouseMoveEvent(self, ev):
        if self._mask_mode and self._drawing:
            pt = ev.position().toPoint()
            self._draw_line(self._last_pt, pt)
            self._last_pt = pt

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._last_pt = None
            if self._mask_mode:
                self.mask_stroke.emit()

    # ── internal ──────────────────────────────────────────────
    def _img_rect(self):
        if self._base_rgb is None:
            return 0, 0, self.width(), self.height()
        fh, fw = self._base_rgb.shape[:2]
        lw, lh = self.width(), self.height()
        scale  = min(lw / fw, lh / fh)
        iw, ih = int(fw * scale), int(fh * scale)
        return (lw - iw) // 2, (lh - ih) // 2, iw, ih

    def _display_to_frame(self, pt: QPoint):
        ix, iy, iw, ih = self._img_rect()
        if self._base_rgb is None:
            return 0, 0
        fh, fw = self._base_rgb.shape[:2]
        mx = int((pt.x() - ix) / max(iw, 1) * fw)
        my = int((pt.y() - iy) / max(ih, 1) * fh)
        return np.clip(mx, 0, fw - 1), np.clip(my, 0, fh - 1)

    def _draw_at(self, pt: QPoint):
        if self._mask is None or self._base_rgb is None:
            return
        mx, my = self._display_to_frame(pt)
        val    = 0 if self._erase_mode else 255
        cv2.circle(self._mask, (mx, my), self._brush_size, val, -1)
        self._refresh()

    def _draw_line(self, p1: QPoint, p2: QPoint):
        if self._mask is None or self._base_rgb is None:
            return
        x1, y1 = self._display_to_frame(p1)
        x2, y2 = self._display_to_frame(p2)
        val = 0 if self._erase_mode else 255
        cv2.line(self._mask, (x1, y1), (x2, y2), val, self._brush_size * 2)
        self._refresh()

    def _refresh(self):
        if self._base_rgb is None:
            return
        display = self._base_rgb.copy()
        if self._mask is not None and self._mask.max() > 0:
            red   = np.zeros_like(display)
            red[:, :, 0] = 255
            alpha = (self._mask > 127)[:, :, np.newaxis]
            display = np.where(alpha,
                               (display.astype(np.float32) * 0.4
                                + red.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8),
                               display)
        pm = numpy_to_pixmap(display)
        self.setPixmap(fit_pixmap(pm, self.width(), self.height()))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refresh()


class ViewerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Mode + toolbar bar ────────────────────────────────
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background:#1e1e1e; border-bottom:1px solid #2a2a2a;")
        tb = QHBoxLayout(bar)
        tb.setContentsMargins(6, 2, 6, 2)
        tb.setSpacing(4)

        # View mode buttons
        self._btn_single = QPushButton("단일 뷰")
        self._btn_single.setCheckable(True)
        self._btn_single.setChecked(True)
        self._btn_single.setFixedHeight(26)
        self._btn_single.clicked.connect(lambda: self._set_mode(0))
        tb.addWidget(self._btn_single)

        self._btn_multi = QPushButton("멀티캠")
        self._btn_multi.setCheckable(True)
        self._btn_multi.setFixedHeight(26)
        self._btn_multi.clicked.connect(lambda: self._set_mode(1))
        tb.addWidget(self._btn_multi)

        self._btn_canvas = QPushButton("캔버스")
        self._btn_canvas.setCheckable(True)
        self._btn_canvas.setFixedHeight(26)
        self._btn_canvas.clicked.connect(lambda: self._set_mode(2))
        tb.addWidget(self._btn_canvas)

        tb.addSpacing(12)

        self._lbl_info = QLabel("프리뷰 없음")
        self._lbl_info.setStyleSheet("color:#888888; font-size:10px;")
        tb.addWidget(self._lbl_info)
        tb.addStretch()

        self._btn_mask = QPushButton("마스크 모드")
        self._btn_mask.setCheckable(True)
        self._btn_mask.setFixedHeight(26)
        self._btn_mask.toggled.connect(self._toggle_mask_mode)
        tb.addWidget(self._btn_mask)

        self._btn_erase = QPushButton("지우개")
        self._btn_erase.setCheckable(True)
        self._btn_erase.setEnabled(False)
        self._btn_erase.setFixedHeight(26)
        self._btn_erase.toggled.connect(lambda v: self._viewer.set_erase_mode(v))
        tb.addWidget(self._btn_erase)

        self._btn_clear = QPushButton("마스크 초기화")
        self._btn_clear.setEnabled(False)
        self._btn_clear.setFixedHeight(26)
        self._btn_clear.clicked.connect(self._viewer_clear)
        tb.addWidget(self._btn_clear)

        lay.addWidget(bar)

        # ── Stacked: single viewer | multi-cam view ───────────
        self._stack = QStackedWidget()

        self._viewer = ViewerLabel()
        self._stack.addWidget(self._viewer)       # index 0

        self._multicam = MultiCamView()
        self._stack.addWidget(self._multicam)         # index 1

        self._canvas = InteractiveCanvasView()
        self._stack.addWidget(self._canvas)           # index 2

        lay.addWidget(self._stack, stretch=1)

    def _set_mode(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._btn_single.setChecked(idx == 0)
        self._btn_multi.setChecked(idx == 1)
        self._btn_canvas.setChecked(idx == 2)
        is_single = idx == 0
        for w in (self._btn_mask, self._btn_erase, self._btn_clear):
            w.setVisible(is_single)

    def _toggle_mask_mode(self, on: bool):
        self._viewer.set_mask_mode(on)
        self._btn_erase.setEnabled(on)
        self._btn_clear.setEnabled(on)
        self._btn_mask.setText("마스크 ON" if on else "마스크 모드")

    def _viewer_clear(self):
        self._viewer.clear_mask()

    # ── public ────────────────────────────────────────────────
    def show_image(self, rgb: np.ndarray, info: str = ""):
        self._viewer.show_image(rgb)
        h, w = rgb.shape[:2]
        self._lbl_info.setText(f"{w}×{h}px  {info}" if info else f"{w}×{h}px")

    def show_placeholder(self, text="스티칭 프리뷰"):
        self._viewer.setText(text)
        self._viewer.setPixmap(QPixmap())
        self._lbl_info.setText(text)

    def set_brush_size(self, size: int):
        self._viewer.set_brush_size(size)

    def get_mask(self) -> np.ndarray | None:
        return self._viewer.get_mask()

    def set_mask(self, mask):
        self._viewer.set_mask(mask)

    def set_multicam_frames(self, frames: dict):
        self._multicam.set_frames(frames)
        self._canvas.set_frames(frames)

    def set_canvas_homo_data(self, homo_data: dict | None):
        self._canvas.set_homo_data(homo_data)

    def set_canvas_stitch_result(self, rgb: np.ndarray):
        self._canvas.set_stitch_result(rgb)

    def set_canvas_status(self, msg: str):
        self._canvas.set_status(msg)

    @property
    def multicam(self) -> MultiCamView:
        return self._multicam

    @property
    def canvas_view(self) -> InteractiveCanvasView:
        return self._canvas

    @property
    def mask_stroke(self):
        return self._viewer.mask_stroke
