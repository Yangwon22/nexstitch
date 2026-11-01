from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from core.video import get_video_info, MpvEngine
from core.settings import INPUT_DIR, OUTPUT_DIR
from core.project import Project

from ui.source_panel  import SourcePanel
from ui.stitch_canvas import StitchCanvas
from ui.fonts         import make_ui_font, make_mono_font
from ui.workers       import MpvSeekCaptureWorker, MpvLoadWorker, MpvCaptureOnlyWorker


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self._project = project
        title = (f"NexStitch  —  {project.name}" if project
                 else "NexStitch  —  RED Komodo · Cylindrical Stitch")
        self.setWindowTitle(title)
        self.setMinimumSize(1280, 820)

        self._mov_files_map: dict[int, Path | None] = {i: None for i in range(1, 6)}
        self._cur_frame  = 0
        self._frames_rgb: dict[int, np.ndarray] = {}
        self._is_playing = False
        self._fps        = 24.0

        # MPV engine — initialised lazily on first load
        self._mpv = MpvEngine()

        # Seek-capture worker (only one active at a time)
        self._seek_worker: MpvSeekCaptureWorker | None = None
        self._load_worker:     MpvLoadWorker          | None = None
        self._playback_worker: MpvCaptureOnlyWorker   | None = None

        # Playback polling timer
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(80)          # ~12 fps UI update
        self._play_timer.timeout.connect(self._poll_playback)

        self._build_ui()
        self._build_menu()

        QTimer.singleShot(100, self._source.refresh)

    # ─────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar ────────────────────────────────────────
        title_bar = QWidget()
        title_bar.setFixedHeight(32)
        title_bar.setStyleSheet("background:#111111; border-bottom:1px solid #2a2a2a;")
        tl = QHBoxLayout(title_bar)
        tl.setContentsMargins(12, 0, 12, 0)

        lbl_t = QLabel("NexStitch")
        lbl_t.setFont(make_ui_font(14, bold=True))
        lbl_t.setStyleSheet("color:#00ff88; background:transparent; letter-spacing:2px;")
        tl.addWidget(lbl_t)

        proj_name = self._project.name if self._project else "RED Komodo · Cylindrical Panorama Stitching"
        lbl_s = QLabel(proj_name)
        lbl_s.setFont(make_ui_font(9, bold=False))
        lbl_s.setStyleSheet("color:#888888; background:transparent;")
        tl.addWidget(lbl_s)
        tl.addStretch()

        self._lbl_frame = QLabel("Frame: 0")
        self._lbl_frame.setFont(make_mono_font(9))
        self._lbl_frame.setStyleSheet("color:#666666; background:transparent;")
        tl.addWidget(self._lbl_frame)

        root.addWidget(title_bar)

        # ── Main splitter ─────────────────────────────────────
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(4)

        self._source = SourcePanel()
        self._source.camera_selected.connect(self._on_camera_selected)
        self._source.files_refreshed.connect(self._on_files_refreshed)
        h_split.addWidget(self._source)

        self._stitch_canvas = StitchCanvas()
        self._stitch_canvas.status_changed.connect(
            lambda m: self._statusbar.showMessage(m))
        h_split.addWidget(self._stitch_canvas)

        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        h_split.setSizes([240, 1080])
        root.addWidget(h_split, stretch=1)

        # ── Transport + frame navigator bar ──────────────────
        nav = QWidget()
        nav.setFixedHeight(38)
        nav.setStyleSheet("background:#181818; border-top:1px solid #222222;")
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(10, 4, 10, 4)
        nl.setSpacing(6)

        # Play / Pause button
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedSize(32, 26)
        self._btn_play.setFont(make_ui_font(12))
        self._btn_play.setStyleSheet(
            "QPushButton{background:#1a2a1a; color:#00ff88;"
            " border:1px solid #00ff88; border-radius:3px;}"
            "QPushButton:hover{background:#223322;}"
        )
        self._btn_play.clicked.connect(self._on_play_toggle)
        nl.addWidget(self._btn_play)

        from PyQt6.QtWidgets import QSlider, QSpinBox

        nl.addWidget(QLabel("Frame"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 9999)
        self._frame_spin.setFixedWidth(70)
        self._frame_spin.setFont(make_mono_font(9))
        self._frame_spin.setStyleSheet("color:#00ff88; background:#0d0d0d;")
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        nl.addWidget(self._frame_spin)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 9999)
        self._frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self._frame_slider.sliderReleased.connect(self._on_slider_released)
        self._frame_slider.valueChanged.connect(self._on_slider_value_changed)
        nl.addWidget(self._frame_slider, stretch=1)

        # Timecode display
        self._lbl_tc = QLabel("00:00:00")
        self._lbl_tc.setFont(make_mono_font(9))
        self._lbl_tc.setStyleSheet("color:#555555; background:transparent;")
        self._lbl_tc.setFixedWidth(60)
        nl.addWidget(self._lbl_tc)

        root.addWidget(nav)

        # ── Status bar ────────────────────────────────────────
        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet(
            "QStatusBar { background:#252525; color:#888888; font-size:10px;"
            " border-top:1px solid #3a3a3a; }")
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

        self._slider_dragging = False

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("File")
        act_open = QAction("Open Input Folder", self)
        act_open.triggered.connect(
            lambda: __import__("os").startfile(
                str(self._project.input_dir if self._project else INPUT_DIR)))
        file_m.addAction(act_open)

        act_out = QAction("Open Output Folder", self)
        act_out.triggered.connect(
            lambda: __import__("os").startfile(
                str(self._project.output_dir if self._project else OUTPUT_DIR)))
        file_m.addAction(act_out)

        file_m.addSeparator()
        act_r5 = QAction("Refresh Files  (F5)", self)
        act_r5.setShortcut("F5")
        act_r5.triggered.connect(self._source.refresh)
        file_m.addAction(act_r5)

        file_m.addSeparator()
        act_quit = QAction("Quit  Ctrl+Q", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_m.addAction(act_quit)

        cal_m = mb.addMenu("Calibration")
        act_save_c = QAction("Save Calibration  (S)", self)
        act_save_c.setShortcut("S")
        act_save_c.triggered.connect(self._on_save_calibration)
        cal_m.addAction(act_save_c)

        act_improve = QAction("Improve Offsets", self)
        act_improve.triggered.connect(self._stitch_canvas._improve_offsets)
        cal_m.addAction(act_improve)

        render_m = mb.addMenu("Render")
        act_render = QAction("Render…  Ctrl+R", self)
        act_render.setShortcut("Ctrl+R")
        act_render.triggered.connect(self._open_render_dialog)
        render_m.addAction(act_render)

    # ─────────────────────────────────────────────────────────
    # Slots — file loading
    # ─────────────────────────────────────────────────────────
    def _on_files_refreshed(self, mapping: dict):
        self._mov_files_map = mapping
        assigned = [p for p in mapping.values() if p is not None]
        n = len(assigned)
        src_dir = self._project.input_dir if self._project else INPUT_DIR
        self._statusbar.showMessage(f"Source Files: {n}/5  |  {src_dir}")

        if not assigned:
            return

        # Get total frames / fps from first assigned file (fast — metadata only)
        info = get_video_info(str(assigned[0]))
        total     = max(0, info.get("total", 0) - 1)
        self._fps = info.get("fps", 24.0)

        self._frame_slider.setMaximum(total)
        self._frame_spin.setMaximum(total)

        # Load MPV players in background — avoids blocking the UI event loop
        self._btn_play.setEnabled(False)
        self._statusbar.showMessage("MPV initializing…  (background loading)")

        self._load_worker = MpvLoadWorker(self._mpv, mapping)
        self._load_worker.progress.connect(self._on_mpv_progress)
        self._load_worker.loaded.connect(self._on_mpv_loaded)
        self._load_worker.error.connect(
            lambda e: self._statusbar.showMessage(f"MPV load error: {e}"))
        self._load_worker.start()

    def _on_mpv_progress(self, done: int, total: int):
        self._statusbar.showMessage(f"MPV initializing… ({done}/{total})")

    def _on_mpv_loaded(self, total: int, fps: float):
        """Called from MpvLoadWorker when all MPV instances are ready."""
        self._fps = fps
        self._btn_play.setEnabled(True)
        self._statusbar.showMessage("MPV load complete — Capturing first frame…")
        self._seek_and_capture(0)

    def _on_camera_selected(self, cam_id: int):
        pass  # future: zoom to camera on canvas

    # ─────────────────────────────────────────────────────────
    # Slots — transport controls
    # ─────────────────────────────────────────────────────────
    def _on_play_toggle(self):
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self._mpv.has_players:
            return
        self._is_playing = True
        self._btn_play.setText("⏸")
        self._mpv.play_all()
        self._play_timer.start()

    def _pause(self):
        self._is_playing = False
        self._btn_play.setText("▶")
        self._play_timer.stop()
        # Disconnect any in-progress playback capture
        if self._playback_worker and self._playback_worker.isRunning():
            self._playback_worker.frames_ready.disconnect()
            self._playback_worker = None
        self._mpv.pause_all()
        # Capture current frame and stitch
        self._seek_and_capture(self._cur_frame)

    def _poll_playback(self):
        """Called every ~80 ms during playback to update UI and canvas preview."""
        t = self._mpv.ref_time_pos()
        fps = max(self._fps, 1.0)
        idx = min(int(t * fps), self._frame_slider.maximum())

        self._cur_frame = idx
        self._lbl_frame.setText(f"Frame: {idx}")
        self._update_timecode(t)

        # Update slider + spin without triggering seek callbacks
        self._frame_slider.blockSignals(True)
        self._frame_spin.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_spin.setValue(idx)
        self._frame_slider.blockSignals(False)
        self._frame_spin.blockSignals(False)

        # Capture current frame for canvas preview (skip if previous still running)
        if (self._mpv.has_players and
                (self._playback_worker is None or
                 not self._playback_worker.isRunning())):
            w = MpvCaptureOnlyWorker(self._mpv, idx)
            w.frames_ready.connect(self._on_playback_frames)
            self._playback_worker = w
            w.start()

        # Auto-stop at end
        if idx >= self._frame_slider.maximum():
            self._pause()

    # ─────────────────────────────────────────────────────────
    # Slots — frame navigation
    # ─────────────────────────────────────────────────────────
    def _on_frame_spin_changed(self, idx: int):
        if self._slider_dragging:
            return
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)
        self._cur_frame = idx
        self._lbl_frame.setText(f"Frame: {idx}")
        self._update_timecode(idx / max(self._fps, 1.0))
        if not self._is_playing:
            self._seek_and_capture(idx)
        else:
            self._mpv.seek_frame(idx)   # re-sync during playback (non-blocking fire)

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        idx = self._frame_slider.value()
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx)
        self._frame_spin.blockSignals(False)
        self._cur_frame = idx
        self._lbl_frame.setText(f"Frame: {idx}")
        self._update_timecode(idx / max(self._fps, 1.0))
        if not self._is_playing:
            self._seek_and_capture(idx)
        else:
            self._mpv.seek_frame(idx)

    def _on_slider_value_changed(self, v: int):
        # Only update display labels while dragging (seek happens on release)
        if self._slider_dragging:
            self._lbl_frame.setText(f"Frame: {v}")
            self._update_timecode(v / max(self._fps, 1.0))

    def _update_timecode(self, secs: float):
        s = int(secs)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        self._lbl_tc.setText(f"{h:02d}:{m:02d}:{sec:02d}")

    # ─────────────────────────────────────────────────────────
    # MPV seek + capture (async, replaces old _load_frames)
    # ─────────────────────────────────────────────────────────
    def _seek_and_capture(self, frame_idx: int):
        """Start MpvSeekCaptureWorker. Aborts any running worker first."""
        if not self._mpv.has_players:
            return

        # Cancel previous worker if still running
        if self._seek_worker and self._seek_worker.isRunning():
            self._seek_worker.frames_ready.disconnect()
            self._seek_worker.error.disconnect()
            # Can't abort seek easily; just disconnect and let it finish
            self._seek_worker = None

        self._statusbar.showMessage(f"Loading frame {frame_idx}…")
        w = MpvSeekCaptureWorker(self._mpv, frame_idx)
        w.frames_ready.connect(self._on_frames_captured)
        w.error.connect(lambda e: self._statusbar.showMessage(f"Capture error: {e}"))
        self._seek_worker = w
        w.start()

    def _on_playback_frames(self, frames_rgb: dict):
        """Playback preview capture complete — fast display without stitching."""
        self._stitch_canvas.set_frames_preview(frames_rgb)

    def _on_frames_captured(self, frames_rgb: dict):
        self._frames_rgb = frames_rgb
        self._stitch_canvas.set_frames(frames_rgb)
        n = len(frames_rgb)
        self._statusbar.showMessage(
            f"Frame {self._cur_frame}  |  {n}/5 cameras  |  OpenCV")

    # ─────────────────────────────────────────────────────────
    # Calibration save (wraps canvas save + project timestamp)
    # ─────────────────────────────────────────────────────────
    def _on_save_calibration(self):
        self._stitch_canvas._save_params()
        if self._project:
            self._project.touch_modified()
            from core.recent_projects import add_recent
            add_recent(self._project)

    # ─────────────────────────────────────────────────────────
    # Render dialog
    # ─────────────────────────────────────────────────────────
    def _open_render_dialog(self):
        from ui.render_dialog import RenderDialog
        dlg = RenderDialog(
            mov_files_map = self._mov_files_map,
            fps           = self._fps,
            total_frames  = self._frame_slider.maximum() + 1,
            cam_params    = self._stitch_canvas.get_params(),
            feather_width = self._stitch_canvas.get_feather_width(),
            color_params  = self._stitch_canvas.get_color_params(),
            parent        = self,
        )
        dlg.exec()

    # ─────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────
    def closeEvent(self, ev):
        self._play_timer.stop()
        if self._playback_worker and self._playback_worker.isRunning():
            self._playback_worker.frames_ready.disconnect()
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait(2000)
        self._mpv.close()
        super().closeEvent(ev)
