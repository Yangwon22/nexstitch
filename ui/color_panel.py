"""
NexStitch — Color Pipeline Panel
Per-camera and output (post-stitch) color correction UI.
"""
import copy

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QPushButton,
    QGroupBox, QScrollArea, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

_MONO   = "font-family:'JetBrains Mono','Consolas',monospace; font-size:9px; color:#00ff88;"
_CAM_C  = {1: "#ffffff", 2: "#ffffff", 3: "#ffffff", 4: "#ffffff", 5: "#ffffff"}
_GRP    = ("QGroupBox { color:#666666; font-size:9px; border:1px solid #222222;"
           " border-radius:3px; margin-top:8px; padding-top:6px; }"
           "QGroupBox::title { subcontrol-origin:margin; left:8px; }")
_CURVE_C = {"r": QColor(220, 80,  80),
            "g": QColor(80,  200, 80),
            "b": QColor(80,  130, 255)}


# ─────────────────────────────────────────────────────────────
class CurveWidget(QWidget):
    """Interactive RGB curve editor.
    Left-click on empty space → add control point.
    Left-click+drag on point  → move it.
    Right-click on point      → remove it (endpoints locked).
    """
    points_changed = pyqtSignal(list)   # [[x, y], ...] in 0-255 range

    _SZ  = 210   # widget width & height in pixels
    _PAD = 10    # inner padding for the curve drawing area
    _R   = 5     # control-point hit radius

    def __init__(self, channel: str = "r", parent=None):
        super().__init__(parent)
        self._ch     = channel
        self._points: list[list[int]] = [[0, 0], [255, 255]]
        self._drag   = None          # reference to dragged point (list object)
        self.setFixedSize(self._SZ, self._SZ)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── coordinate helpers ────────────────────────────────────
    def _to_widget(self, cx: float, cy: float):
        s = self._SZ - 2 * self._PAD
        return (self._PAD + cx * s / 255.0,
                self._SZ - self._PAD - cy * s / 255.0)

    def _to_curve(self, wx: float, wy: float):
        s = self._SZ - 2 * self._PAD
        x = int((wx - self._PAD) / s * 255.0)
        y = int((self._SZ - self._PAD - wy) / s * 255.0)
        return max(0, min(255, x)), max(0, min(255, y))

    # ── public API ────────────────────────────────────────────
    def set_points(self, pts: list):
        self._points = sorted([list(p) for p in pts], key=lambda p: p[0])
        self.update()

    def get_points(self) -> list:
        return [list(p) for p in self._points]

    def reset(self):
        self._points = [[0, 0], [255, 255]]
        self.points_changed.emit(self.get_points())
        self.update()

    # ── paint ─────────────────────────────────────────────────
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(self.rect(), QColor(12, 12, 12))

        # Grid
        p.setPen(QPen(QColor(32, 32, 32), 1))
        for i in range(1, 4):
            t   = i / 4
            s   = self._SZ - 2 * self._PAD
            gx  = int(self._PAD + t * s)
            gy  = int(self._PAD + t * s)
            p.drawLine(gx, self._PAD, gx, self._SZ - self._PAD)
            p.drawLine(self._PAD, gy, self._SZ - self._PAD, gy)

        # Diagonal identity reference
        p.setPen(QPen(QColor(42, 42, 42), 1))
        wx0, wy0 = self._to_widget(0, 0)
        wx1, wy1 = self._to_widget(255, 255)
        p.drawLine(int(wx0), int(wy0), int(wx1), int(wy1))

        # Curve
        pts_sorted = sorted(self._points, key=lambda pt: pt[0])
        if len(pts_sorted) >= 2:
            xs = [pt[0] for pt in pts_sorted]
            ys = [pt[1] for pt in pts_sorted]
            p.setPen(QPen(_CURVE_C.get(self._ch, QColor(200, 200, 200)), 1.5))
            prev_wx = prev_wy = None
            for xi in range(0, 256, 2):
                yi  = float(np.interp(xi, xs, ys))
                wx, wy = self._to_widget(xi, yi)
                if prev_wx is not None:
                    p.drawLine(int(prev_wx), int(prev_wy), int(wx), int(wy))
                prev_wx, prev_wy = wx, wy

        # Control points
        for pt in self._points:
            wx, wy = self._to_widget(pt[0], pt[1])
            is_active = (pt is self._drag)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QBrush(QColor(230, 230, 230) if is_active else QColor(150, 150, 150)))
            p.drawEllipse(QPointF(wx, wy), self._R, self._R)

    # ── mouse events ──────────────────────────────────────────
    def mousePressEvent(self, ev):
        pos = ev.position()

        if ev.button() == Qt.MouseButton.RightButton:
            # Remove nearest non-endpoint
            best, best_d = -1, float('inf')
            for i, pt in enumerate(self._points):
                if i == 0 or i == len(self._points) - 1:
                    continue
                wx, wy = self._to_widget(pt[0], pt[1])
                d = abs(pos.x() - wx) + abs(pos.y() - wy)
                if d < best_d:
                    best_d, best = d, i
            if best >= 0 and best_d < 22:
                self._points.pop(best)
                self.points_changed.emit(self.get_points())
                self.update()
            return

        # Find existing point to drag
        for pt in self._points:
            wx, wy = self._to_widget(pt[0], pt[1])
            if abs(pos.x() - wx) <= self._R + 3 and abs(pos.y() - wy) <= self._R + 3:
                self._drag = pt
                return

        # Add new point
        cx, cy  = self._to_curve(pos.x(), pos.y())
        new_pt  = [cx, cy]
        self._points.append(new_pt)
        self._points.sort(key=lambda pt: pt[0])
        self._drag = new_pt
        self.points_changed.emit(self.get_points())
        self.update()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        cx, cy = self._to_curve(ev.position().x(), ev.position().y())
        idx = next((i for i, pt in enumerate(self._points) if pt is self._drag), -1)
        if idx < 0:
            return
        if idx == 0:
            cx = 0
        elif idx == len(self._points) - 1:
            cx = 255
        else:
            cx = max(self._points[idx - 1][0] + 1,
                     min(self._points[idx + 1][0] - 1, cx))
        self._drag[0], self._drag[1] = cx, cy
        self.points_changed.emit(self.get_points())
        self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag = None


# ─────────────────────────────────────────────────────────────
class ColorPanel(QWidget):
    """Color correction panel: per-camera and output (post-stitch) controls."""

    per_cam_changed = pyqtSignal(dict)   # full color_params dict
    output_changed  = pyqtSignal(dict)   # full color_params dict
    match_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from core.color import default_color_params
        self._data:     dict = default_color_params()
        self._cur_cam:  int  = 1
        self._blocking: bool = False

        # Widget registries (populated by _build_ui)
        self._cam_btns:    dict = {}
        self._pc_sliders:  dict = {}   # key → (QSlider, scale_float, unit_str)
        self._out_sliders: dict = {}
        self._lgg:         dict = {}   # "_pc_lift_r" etc. → QDoubleSpinBox
        self._curves:      dict = {}   # "r"/"g"/"b" → CurveWidget

        self._build_ui()
        self._load_per_cam(1)
        self._load_output()

    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scroll area wraps everything (panel content is tall)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:#181818; }")

        content = QWidget()
        content.setStyleSheet("background:#181818;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        lay.addWidget(self._build_per_cam_group())
        lay.addWidget(self._build_output_group())
        lay.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    # ── Per-camera group ──────────────────────────────────────
    def _build_per_cam_group(self) -> QGroupBox:
        gb  = QGroupBox("Per Camera")
        gb.setStyleSheet(_GRP)
        lay = QVBoxLayout(gb)
        lay.setSpacing(4)

        # Camera selector + match button
        cam_row = QHBoxLayout()
        cam_row.setSpacing(2)
        for cam_id in range(1, 6):
            btn = QPushButton(str(cam_id))
            btn.setFixedSize(30, 22)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { border:1px solid #ffffff33; border-radius:2px;"
                "  color:#888888; font-size:9px; }"
                "QPushButton:checked { background:#00ff8822; border-color:#00ff88;"
                "  color:#00ff88; font-weight:bold; }"
            )
            btn.clicked.connect(lambda _, cid=cam_id: self._select_cam(cid))
            cam_row.addWidget(btn)
            self._cam_btns[cam_id] = btn

        btn_match = QPushButton("Match→CAM3")
        btn_match.setFixedHeight(22)
        btn_match.setStyleSheet(
            "QPushButton { border:1px solid #333333; border-radius:2px;"
            "  color:#888888; font-size:9px; padding:0 4px; }"
            "QPushButton:hover { color:#ffffff; border-color:#666666; }"
        )
        btn_match.clicked.connect(self.match_requested)
        cam_row.addWidget(btn_match)
        lay.addLayout(cam_row)

        # Sliders
        sl_grid = QGridLayout()
        sl_grid.setSpacing(3)
        self._pc_sliders["exposure"]    = self._add_slider(
            sl_grid, 0, "Exposure", -40,   40,    0,  10.0, "EV",
            self._on_per_cam_changed)
        self._pc_sliders["temperature"] = self._add_slider(
            sl_grid, 1, "Temp",    2000, 10000, 6500,  1.0, "K",
            self._on_per_cam_changed)
        self._pc_sliders["tint"]        = self._add_slider(
            sl_grid, 2, "Tint",    -100,   100,    0,  1.0, "",
            self._on_per_cam_changed)
        lay.addLayout(sl_grid)

        # LGG grid
        lay.addWidget(self._build_lgg_grid(per_cam=True))

        btn_reset = QPushButton("Reset Camera Color")
        btn_reset.setFixedHeight(20)
        btn_reset.setStyleSheet(
            "QPushButton { font-size:9px; color:#888888; border:1px solid #222222; }"
            "QPushButton:hover { color:#ffffff; }"
        )
        btn_reset.clicked.connect(self._reset_per_cam)
        lay.addWidget(btn_reset)

        return gb

    # ── Output group ──────────────────────────────────────────
    def _build_output_group(self) -> QGroupBox:
        gb  = QGroupBox("Output  (Post-Stitch)")
        gb.setStyleSheet(_GRP)
        lay = QVBoxLayout(gb)
        lay.setSpacing(4)

        sl_grid = QGridLayout()
        sl_grid.setSpacing(3)
        self._out_sliders["exposure"]    = self._add_slider(
            sl_grid, 0, "Exposure",    -40,   40,    0,  10.0, "EV",
            self._on_output_changed)
        self._out_sliders["contrast"]    = self._add_slider(
            sl_grid, 1, "Contrast",     50,  200,  100, 100.0, "",
            self._on_output_changed)
        self._out_sliders["saturation"]  = self._add_slider(
            sl_grid, 2, "Saturation",    0,  200,  100, 100.0, "",
            self._on_output_changed)
        self._out_sliders["temperature"] = self._add_slider(
            sl_grid, 3, "Temp",       2000, 10000, 6500,  1.0, "K",
            self._on_output_changed)
        self._out_sliders["tint"]        = self._add_slider(
            sl_grid, 4, "Tint",       -100,   100,    0,  1.0, "",
            self._on_output_changed)
        lay.addLayout(sl_grid)

        lay.addWidget(self._build_lgg_grid(per_cam=False))
        lay.addWidget(self._build_curves_section())

        btn_reset = QPushButton("Reset Output Color")
        btn_reset.setFixedHeight(20)
        btn_reset.setStyleSheet(
            "QPushButton { font-size:9px; color:#888888; border:1px solid #222222; }"
            "QPushButton:hover { color:#ffffff; }"
        )
        btn_reset.clicked.connect(self._reset_output)
        lay.addWidget(btn_reset)

        return gb

    # ── Slider helper ─────────────────────────────────────────
    def _add_slider(self, grid: QGridLayout, row: int,
                    label: str, lo: int, hi: int, default: int,
                    scale: float, unit: str, callback) -> tuple:
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#888888; font-size:9px;")
        lbl.setFixedWidth(58)
        grid.addWidget(lbl, row, 0)

        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(default)
        sl.setFixedHeight(16)
        grid.addWidget(sl, row, 1)

        val = QLabel(self._fmt(default, scale, unit))
        val.setStyleSheet(_MONO)
        val.setFixedWidth(52)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(val, row, 2)

        sl.valueChanged.connect(lambda v, vl=val, sc=scale, u=unit:
                                vl.setText(self._fmt(v, sc, u)))
        sl.valueChanged.connect(lambda _: callback())

        return sl, scale, unit

    @staticmethod
    def _fmt(raw: int, scale: float, unit: str) -> str:
        v = raw / scale
        if scale >= 100:
            return f"{v:.2f}{unit}"
        elif scale >= 10:
            return f"{v:.1f}{unit}"
        return f"{int(v)}{unit}"

    # ── LGG grid ──────────────────────────────────────────────
    def _build_lgg_grid(self, per_cam: bool) -> QWidget:
        w  = QWidget()
        gl = QGridLayout(w)
        gl.setSpacing(2)
        gl.setContentsMargins(0, 2, 0, 2)

        ch_colors = ["#cc5555", "#55cc55", "#5588cc"]
        for ci, ch in enumerate(("R", "G", "B")):
            lbl = QLabel(ch)
            lbl.setStyleSheet(f"color:{ch_colors[ci]}; font-size:9px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            gl.addWidget(lbl, 0, ci + 1)

        pfx = "_pc" if per_cam else "_out"
        cb  = self._on_per_cam_changed if per_cam else self._on_output_changed

        for ri, (name, lo, hi, default, step) in enumerate([
            ("lift",  -0.20, 0.20, 0.00, 0.010),
            ("gamma",  0.10, 3.00, 1.00, 0.050),
            ("gain",   0.10, 3.00, 1.00, 0.050),
        ]):
            lbl = QLabel(name.capitalize())
            lbl.setStyleSheet("color:#666666; font-size:9px;")
            gl.addWidget(lbl, ri + 1, 0)
            for ci, ch in enumerate(("r", "g", "b")):
                key = f"{pfx}_{name}_{ch}"
                dsb = QDoubleSpinBox()
                dsb.setRange(lo, hi)
                dsb.setValue(default)
                dsb.setSingleStep(step)
                dsb.setDecimals(2)
                dsb.setFixedHeight(20)
                dsb.setFixedWidth(62)
                dsb.setStyleSheet(_MONO)
                dsb.valueChanged.connect(lambda _, c=cb: c())
                self._lgg[key] = dsb
                gl.addWidget(dsb, ri + 1, ci + 1)

        return w

    # ── Curves section ────────────────────────────────────────
    def _build_curves_section(self) -> QGroupBox:
        gb  = QGroupBox("Curves")
        gb.setStyleSheet(_GRP)
        lay = QVBoxLayout(gb)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border:none; }"
            "QTabBar::tab { color:#888888; font-size:9px; padding:3px 10px; }"
            "QTabBar::tab:selected { color:#00ff88; }"
        )

        for ch, label in (("r", "R"), ("g", "G"), ("b", "B")):
            page = QWidget()
            pl   = QVBoxLayout(page)
            pl.setContentsMargins(4, 4, 4, 4)
            pl.setSpacing(2)

            cw = CurveWidget(channel=ch)
            cw.points_changed.connect(lambda pts, c=ch: self._on_curve_changed(c, pts))
            pl.addWidget(cw, alignment=Qt.AlignmentFlag.AlignHCenter)

            btn = QPushButton("Reset")
            btn.setFixedHeight(18)
            btn.setStyleSheet(
                "QPushButton { font-size:9px; color:#888888; border:1px solid #222222; }"
                "QPushButton:hover { color:#ffffff; }"
            )
            btn.clicked.connect(cw.reset)
            pl.addWidget(btn)

            self._curves[ch] = cw
            tabs.addTab(page, label)

        lay.addWidget(tabs)
        return gb

    # ── Data load / collect ───────────────────────────────────
    def load_params(self, color_params: dict):
        self._blocking = True
        try:
            self._data = copy.deepcopy(color_params)
            self._load_per_cam(self._cur_cam)
            self._load_output()
        finally:
            self._blocking = False

    def get_params(self) -> dict:
        return copy.deepcopy(self._data)

    def select_cam(self, cam_id: int):
        if 1 <= cam_id <= 5:
            self._select_cam(cam_id)

    def _select_cam(self, cam_id: int):
        self._cur_cam = cam_id
        for cid, btn in self._cam_btns.items():
            btn.setChecked(cid == cam_id)
        self._load_per_cam(cam_id)

    def _load_per_cam(self, cam_id: int):
        p = self._data["per_cam"].get(cam_id, {})
        self._blocking = True
        try:
            sl, sc, _ = self._pc_sliders["exposure"]
            sl.setValue(int(p.get("exposure", 0.0) * sc))
            sl, sc, _ = self._pc_sliders["temperature"]
            sl.setValue(int(p.get("temperature", 6500)))
            sl, sc, _ = self._pc_sliders["tint"]
            sl.setValue(int(p.get("tint", 0)))

            for ci, ch in enumerate(("r", "g", "b")):
                self._lgg[f"_pc_lift_{ch}"].setValue(p.get("lift",  [0, 0, 0])[ci])
                self._lgg[f"_pc_gamma_{ch}"].setValue(p.get("gamma", [1, 1, 1])[ci])
                self._lgg[f"_pc_gain_{ch}"].setValue(p.get("gain",  [1, 1, 1])[ci])
        finally:
            self._blocking = False

        # Update camera button state
        for cid, btn in self._cam_btns.items():
            btn.setChecked(cid == cam_id)

    def _load_output(self):
        p = self._data.get("output", {})
        self._blocking = True
        try:
            for key, default in [("exposure", 0.0), ("contrast", 1.0),
                                  ("saturation", 1.0), ("temperature", 6500),
                                  ("tint", 0)]:
                sl, sc, _ = self._out_sliders[key]
                sl.setValue(int(p.get(key, default) * sc))

            for ci, ch in enumerate(("r", "g", "b")):
                self._lgg[f"_out_lift_{ch}"].setValue(p.get("lift",  [0, 0, 0])[ci])
                self._lgg[f"_out_gamma_{ch}"].setValue(p.get("gamma", [1, 1, 1])[ci])
                self._lgg[f"_out_gain_{ch}"].setValue(p.get("gain",  [1, 1, 1])[ci])

            curves = p.get("curves", {})
            for ch in ("r", "g", "b"):
                self._curves[ch].set_points(curves.get(ch, [[0, 0], [255, 255]]))
        finally:
            self._blocking = False

    def _collect_per_cam(self) -> dict:
        sl, sc, _ = self._pc_sliders["exposure"]
        sl_t, _, _ = self._pc_sliders["temperature"]
        sl_tn, _, _ = self._pc_sliders["tint"]
        return {
            "exposure":    sl.value() / sc,
            "temperature": sl_t.value(),
            "tint":        sl_tn.value(),
            "lift":   [self._lgg[f"_pc_lift_{c}"].value()  for c in ("r", "g", "b")],
            "gamma":  [self._lgg[f"_pc_gamma_{c}"].value() for c in ("r", "g", "b")],
            "gain":   [self._lgg[f"_pc_gain_{c}"].value()  for c in ("r", "g", "b")],
        }

    def _collect_output(self) -> dict:
        def sv(key): return self._out_sliders[key]

        sl_e,  sc_e,  _ = sv("exposure")
        sl_c,  sc_c,  _ = sv("contrast")
        sl_s,  sc_s,  _ = sv("saturation")
        sl_t,  _,     _ = sv("temperature")
        sl_tn, _,     _ = sv("tint")
        return {
            "exposure":    sl_e.value()  / sc_e,
            "contrast":    sl_c.value()  / sc_c,
            "saturation":  sl_s.value()  / sc_s,
            "temperature": sl_t.value(),
            "tint":        sl_tn.value(),
            "lift":   [self._lgg[f"_out_lift_{c}"].value()  for c in ("r", "g", "b")],
            "gamma":  [self._lgg[f"_out_gamma_{c}"].value() for c in ("r", "g", "b")],
            "gain":   [self._lgg[f"_out_gain_{c}"].value()  for c in ("r", "g", "b")],
            "curves": {ch: self._curves[ch].get_points() for ch in ("r", "g", "b")},
        }

    # ── Signal handlers ───────────────────────────────────────
    def _on_per_cam_changed(self):
        if self._blocking:
            return
        self._data["per_cam"][self._cur_cam] = self._collect_per_cam()
        self.per_cam_changed.emit(copy.deepcopy(self._data))

    def _on_output_changed(self):
        if self._blocking:
            return
        self._data["output"] = self._collect_output()
        self.output_changed.emit(copy.deepcopy(self._data))

    def _on_curve_changed(self, channel: str, pts: list):
        if self._blocking:
            return
        self._data["output"].setdefault("curves", {})[channel] = pts
        self.output_changed.emit(copy.deepcopy(self._data))

    def _reset_per_cam(self):
        from core.color import default_per_cam
        self._data["per_cam"][self._cur_cam] = default_per_cam()
        self._load_per_cam(self._cur_cam)
        self.per_cam_changed.emit(copy.deepcopy(self._data))

    def _reset_output(self):
        from core.color import default_output
        self._data["output"] = default_output()
        self._load_output()
        self.output_changed.emit(copy.deepcopy(self._data))
