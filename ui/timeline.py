from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSlider,
                              QPushButton, QLabel, QSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont


class TimelinePanel(QWidget):
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(80)
        self._fps         = 24.0
        self._total       = 0
        self._playing     = False
        self._timer       = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        # 슬라이더
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        root.addWidget(self._slider)

        # 컨트롤
        ctrl = QHBoxLayout()

        self._btn_first = QPushButton("◀◀")
        self._btn_first.setFixedWidth(36)
        self._btn_first.clicked.connect(lambda: self.set_frame(0))
        ctrl.addWidget(self._btn_first)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.clicked.connect(lambda: self.set_frame(self.current_frame - 1))
        ctrl.addWidget(self._btn_prev)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._btn_play)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(32)
        self._btn_next.setText("▶")
        self._btn_next.clicked.connect(lambda: self.set_frame(self.current_frame + 1))
        ctrl.addWidget(self._btn_next)

        self._btn_last = QPushButton("▶▶")
        self._btn_last.setFixedWidth(36)
        self._btn_last.clicked.connect(lambda: self.set_frame(self._total - 1))
        ctrl.addWidget(self._btn_last)

        ctrl.addSpacing(12)

        lbl_f = QLabel("프레임")
        lbl_f.setStyleSheet("color:#888888; font-size:10px;")
        ctrl.addWidget(lbl_f)

        _mono = QFont("JetBrains Mono")
        _mono.setPointSize(10)

        self._spin = QSpinBox()
        self._spin.setMinimum(0)
        self._spin.setMaximum(0)
        self._spin.setFixedWidth(80)
        self._spin.setFont(_mono)
        self._spin.valueChanged.connect(self._on_spin)
        ctrl.addWidget(self._spin)

        self._lbl_total = QLabel("/ 0")
        self._lbl_total.setFont(_mono)
        self._lbl_total.setStyleSheet("color:#888888;")
        ctrl.addWidget(self._lbl_total)

        ctrl.addStretch()

        self._lbl_fps = QLabel("24.00 fps")
        self._lbl_fps.setFont(_mono)
        self._lbl_fps.setStyleSheet("color:#555555;")
        ctrl.addWidget(self._lbl_fps)

        root.addLayout(ctrl)

    # ── public ────────────────────────────────────────────────
    def set_total_frames(self, total: int, fps: float = 24.0):
        self._total = max(1, total)
        self._fps   = fps
        self._slider.setMaximum(self._total - 1)
        self._spin.setMaximum(self._total - 1)
        self._lbl_total.setText(f"/ {self._total - 1}")
        self._lbl_fps.setText(f"{fps:.2f} fps")

    def set_frame(self, idx: int):
        idx = max(0, min(idx, self._total - 1))
        self._slider.setValue(idx)

    @property
    def current_frame(self) -> int:
        return self._slider.value()

    # ── internal ──────────────────────────────────────────────
    def _on_slider(self, val: int):
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self.frame_changed.emit(val)

    def _on_spin(self, val: int):
        self._slider.setValue(val)

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            interval = max(1, int(1000 / self._fps))
            self._timer.start(interval)
            self._btn_play.setText("⏸")
        else:
            self._timer.stop()
            self._btn_play.setText("▶")

    def _on_tick(self):
        nxt = self.current_frame + 1
        if nxt >= self._total:
            nxt = 0
        self.set_frame(nxt)
