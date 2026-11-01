"""
NexStitch — Render Dialog
Multi-codec cylindrical render pipeline with progress tracking.
"""
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QFileDialog, QLineEdit,
    QTextEdit, QGroupBox, QFrame,
)
from PyQt6.QtCore import Qt

from core.calibration import load_output_resolution
from core.settings import OUTPUT_DIR
from ui.workers import CylRenderWorker

_MONO = "font-family:'JetBrains Mono','Consolas',monospace; font-size:9px;"
_GRP  = ("QGroupBox { color:#666666; font-size:9px; border:1px solid #2a2a2a;"
         " border-radius:3px; margin-top:6px; padding-top:4px; }"
         "QGroupBox::title { subcontrol-origin:margin; left:8px; }")

# ── Codec presets ─────────────────────────────────────────────
# Each entry:
#   label       : displayed in the dropdown
#   ext         : default output file extension (empty string = directory/EXR)
#   is_exr      : True → save EXR sequence via cv2.imwrite, no ffmpeg
#   ffmpeg_args : list[str] appended to ffmpeg command after "-i pipe:0"
#   browse_filter : QFileDialog filter string
_CODEC_PRESETS: list[dict] = [
    {
        "label":        "ProRes 422 HQ",
        "ext":          ".mov",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "prores_ks", "-profile:v", "3",
                         "-pix_fmt", "yuv422p10le", "-vendor", "apl0", "-an"],
        "browse_filter": "QuickTime Movie (*.mov)",
    },
    {
        "label":        "ProRes 4444",
        "ext":          ".mov",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "prores_ks", "-profile:v", "4",
                         "-pix_fmt", "yuva444p10le", "-vendor", "apl0", "-an"],
        "browse_filter": "QuickTime Movie (*.mov)",
    },
    {
        "label":        "ProRes 422",
        "ext":          ".mov",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "prores_ks", "-profile:v", "2",
                         "-pix_fmt", "yuv422p10le", "-vendor", "apl0", "-an"],
        "browse_filter": "QuickTime Movie (*.mov)",
    },
    {
        "label":        "H.264",
        "ext":          ".mp4",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an"],
        "browse_filter": "MPEG-4 Video (*.mp4)",
    },
    {
        "label":        "H.265 / HEVC",
        "ext":          ".mp4",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "libx265", "-crf", "18", "-pix_fmt", "yuv420p", "-an"],
        "browse_filter": "MPEG-4 Video (*.mp4)",
    },
    {
        "label":        "DNxHD 185x",
        "ext":          ".mxf",
        "is_exr":       False,
        "ffmpeg_args":  ["-c:v", "dnxhd", "-b:v", "185M", "-pix_fmt", "yuv422p10le", "-an"],
        "browse_filter": "MXF (*.mxf)",
    },
    {
        "label":        "EXR Sequence",
        "ext":          "",          # output_path is a directory
        "is_exr":       True,
        "ffmpeg_args":  [],
        "browse_filter": "",
    },
]

_DEFAULT_STEM = "NexStitch_output"


class RenderDialog(QDialog):
    def __init__(self, mov_files_map: dict, fps: float = 24.0,
                 total_frames: int = 0, cam_params: dict = None,
                 feather_width: int = 200, color_params: dict = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("NexStitch — Render")
        self.setMinimumSize(700, 660)
        self.setStyleSheet(
            "QDialog { background:#1a1a1a; color:#ffffff; }"
            "QLabel  { color:#cccccc; font-size:10px; }"
            "QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {"
            "  background:#0d0d0d; color:#ffffff; border:1px solid #333333;"
            "  border-radius:2px; font-size:9px; }"
            "QPushButton { background:#1e1e1e; color:#cccccc;"
            "  border:1px solid #333333; border-radius:3px; padding:2px 8px; }"
            "QPushButton:hover { border-color:#555555; color:#ffffff; }"
        )

        self._mov_files_map  = mov_files_map
        self._fps            = fps
        self._total_frames   = max(total_frames, 1)
        self._cam_params     = cam_params or {}
        self._feather_width  = feather_width
        self._color_params   = color_params or {}
        self._worker: CylRenderWorker | None = None
        self._t_start: float | None = None

        self._build_ui()
        self._populate_source()
        # Sync output path label and path to the default preset (index 0)
        self._on_codec_changed(0)

    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Source files ──────────────────────────────────────
        gb_src = QGroupBox("Source  (input/)")
        gb_src.setStyleSheet(_GRP)
        src_lay = QVBoxLayout(gb_src)
        src_lay.setSpacing(2)
        self._lbl_src = QLabel()
        self._lbl_src.setStyleSheet(_MONO + " color:#888888;")
        self._lbl_src.setWordWrap(True)
        src_lay.addWidget(self._lbl_src)
        root.addWidget(gb_src)

        # ── Render settings ───────────────────────────────────
        gb_set = QGroupBox("Render Settings")
        gb_set.setStyleSheet(_GRP)
        gl = QGridLayout(gb_set)
        gl.setSpacing(8)
        gl.setColumnStretch(1, 1)
        row = 0

        # Output resolution
        gl.addWidget(self._rlbl("Output Resolution"), row, 0)
        res = load_output_resolution()
        self._out_w = res["width"]
        self._out_h = res["height"]
        self._lbl_res = QLabel(f"{self._out_w} × {self._out_h}")
        self._lbl_res.setStyleSheet(_MONO + " color:#00ff88;")
        gl.addWidget(self._lbl_res, row, 1)
        row += 1

        # Codec selector
        gl.addWidget(self._rlbl("Codec"), row, 0)
        self._cmb_codec = QComboBox()
        self._cmb_codec.setFixedHeight(24)
        for p in _CODEC_PRESETS:
            self._cmb_codec.addItem(p["label"])
        self._cmb_codec.currentIndexChanged.connect(self._on_codec_changed)
        gl.addWidget(self._cmb_codec, row, 1)
        row += 1

        # Frame range
        gl.addWidget(self._rlbl("Start Frame"), row, 0)
        self._sb_start = QSpinBox()
        self._sb_start.setRange(0, self._total_frames - 1)
        self._sb_start.setValue(0)
        self._sb_start.setFixedHeight(24)
        self._sb_start.setStyleSheet(_MONO)
        gl.addWidget(self._sb_start, row, 1)
        row += 1

        gl.addWidget(self._rlbl("End Frame"), row, 0)
        self._sb_end = QSpinBox()
        self._sb_end.setRange(0, self._total_frames - 1)
        self._sb_end.setValue(self._total_frames - 1)
        self._sb_end.setFixedHeight(24)
        self._sb_end.setStyleSheet(_MONO)
        gl.addWidget(self._sb_end, row, 1)
        row += 1

        # FPS
        gl.addWidget(self._rlbl("Frame Rate"), row, 0)
        self._dsb_fps = QDoubleSpinBox()
        self._dsb_fps.setRange(1.0, 120.0)
        self._dsb_fps.setValue(self._fps)
        self._dsb_fps.setDecimals(3)
        self._dsb_fps.setFixedHeight(24)
        self._dsb_fps.setStyleSheet(_MONO)
        gl.addWidget(self._dsb_fps, row, 1)
        row += 1

        # Feather width
        gl.addWidget(self._rlbl("Feather Width (px)"), row, 0)
        self._sb_feather = QSpinBox()
        self._sb_feather.setRange(0, 1000)
        self._sb_feather.setValue(self._feather_width)
        self._sb_feather.setFixedHeight(24)
        self._sb_feather.setStyleSheet(_MONO)
        gl.addWidget(self._sb_feather, row, 1)
        row += 1

        # Parallel workers
        gl.addWidget(self._rlbl("Parallel Workers"), row, 0)
        self._sb_workers = QSpinBox()
        self._sb_workers.setRange(1, 16)
        self._sb_workers.setValue(4)
        self._sb_workers.setFixedHeight(24)
        self._sb_workers.setStyleSheet(_MONO)
        self._sb_workers.setToolTip(
            "Frames processed in parallel.\nRTX 4090: 4–8 recommended.")
        gl.addWidget(self._sb_workers, row, 1)
        row += 1

        # Output path  (label changes with codec)
        self._lbl_out_path = self._rlbl("Output File")
        gl.addWidget(self._lbl_out_path, row, 0)
        out_row = QHBoxLayout()
        self._edit_out = QLineEdit()
        self._edit_out.setFixedHeight(24)
        self._edit_out.setStyleSheet(_MONO)
        out_row.addWidget(self._edit_out)
        self._btn_browse = QPushButton("…")
        self._btn_browse.setFixedSize(28, 24)
        self._btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._btn_browse)
        gl.addLayout(out_row, row, 1)

        root.addWidget(gb_set)

        # ── Progress ──────────────────────────────────────────
        gb_prog = QGroupBox("Progress")
        gb_prog.setStyleSheet(_GRP)
        prog_lay = QVBoxLayout(gb_prog)
        prog_lay.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border:1px solid #333; border-radius:2px;"
            "  text-align:center; font-size:9px; color:#00ff88; background:#0d0d0d; }"
            "QProgressBar::chunk { background:#00aa44; }"
        )
        prog_lay.addWidget(self._progress_bar)

        info_row = QHBoxLayout()
        self._lbl_frames = QLabel("—")
        self._lbl_frames.setStyleSheet(_MONO + " color:#888888;")
        info_row.addWidget(self._lbl_frames)
        info_row.addStretch()
        self._lbl_eta = QLabel("ETA: —")
        self._lbl_eta.setStyleSheet(_MONO + " color:#555555;")
        info_row.addWidget(self._lbl_eta)
        prog_lay.addLayout(info_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setStyleSheet(
            "QTextEdit { background:#0a0a0a; color:#555555;"
            "  border:1px solid #222; " + _MONO + " }"
        )
        prog_lay.addWidget(self._log)

        root.addWidget(gb_prog)

        # ── Buttons ───────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#2a2a2a;")
        root.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_render = QPushButton("▶  Start Render")
        self._btn_render.setFixedHeight(32)
        self._btn_render.setStyleSheet(
            "QPushButton { background:#1a3a1a; color:#00ff88;"
            "  border:1px solid #00ff88; border-radius:3px;"
            "  font-weight:bold; font-size:11px; padding:0 16px; }"
            "QPushButton:hover { background:#223322; }"
            "QPushButton:disabled { color:#333; border-color:#2a2a2a; background:#111; }"
        )
        self._btn_render.clicked.connect(self._start_render)
        btn_row.addWidget(self._btn_render)

        self._btn_cancel = QPushButton("⏹  Cancel")
        self._btn_cancel.setFixedHeight(32)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setStyleSheet(
            "QPushButton { background:#3a1a1a; color:#ff6666;"
            "  border:1px solid #884444; border-radius:3px; padding:0 12px; }"
            "QPushButton:hover { background:#442222; }"
            "QPushButton:disabled { color:#333; border-color:#2a2a2a; background:#111; }"
        )
        self._btn_cancel.clicked.connect(self._cancel_render)
        btn_row.addWidget(self._btn_cancel)

        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)

        root.addLayout(btn_row)

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _rlbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet("color:#777777; font-size:9px;")
        l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return l

    def _populate_source(self):
        lines = []
        for cam_id in range(1, 6):
            p = self._mov_files_map.get(cam_id)
            lines.append(
                f"CAM{cam_id}  {Path(str(p)).name}" if p
                else f"CAM{cam_id}  ⚠  Not assigned"
            )
        self._lbl_src.setText("\n".join(lines))

    # ── Codec change handler ──────────────────────────────────
    def _on_codec_changed(self, idx: int):
        preset = _CODEC_PRESETS[idx]
        is_exr = preset["is_exr"]
        ext    = preset["ext"]

        if is_exr:
            self._lbl_out_path.setText("Output Folder")
            new_path = str(OUTPUT_DIR / "frames")
        else:
            self._lbl_out_path.setText(f"Output File ({ext})")
            # Preserve stem, swap extension
            cur = Path(self._edit_out.text()) if self._edit_out.text() else None
            stem = cur.stem if cur and cur.suffix else _DEFAULT_STEM
            new_path = str(OUTPUT_DIR / (stem + ext))

        self._edit_out.setText(new_path)

    # ── Browse handler ────────────────────────────────────────
    def _browse_output(self):
        preset = _CODEC_PRESETS[self._cmb_codec.currentIndex()]
        if preset["is_exr"]:
            path = QFileDialog.getExistingDirectory(
                self, "Output Folder", str(OUTPUT_DIR)
            )
            if path:
                self._edit_out.setText(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Output File",
                self._edit_out.text() or str(OUTPUT_DIR),
                preset["browse_filter"],
            )
            if path:
                self._edit_out.setText(path)

    # ── Start render ──────────────────────────────────────────
    def _start_render(self):
        mov_paths: dict[int, str] = {
            cam_id: str(p)
            for cam_id, p in self._mov_files_map.items() if p
        }
        if not mov_paths:
            self._log_msg("✕  No source files assigned")
            return
        if len(mov_paths) < 5:
            self._log_msg(f"⚠  {len(mov_paths)}/5 cameras — rendering with available cameras")

        start = self._sb_start.value()
        end   = self._sb_end.value()
        if end < start:
            self._log_msg("✕  End frame must be ≥ Start frame")
            return

        preset      = _CODEC_PRESETS[self._cmb_codec.currentIndex()]
        output_path = self._edit_out.text().strip() or str(OUTPUT_DIR / _DEFAULT_STEM)
        focal       = self._cam_params.get("focal", 1100.0)

        # Refresh resolution in case toolbar setting changed
        res = load_output_resolution()
        self._out_w = res["width"]
        self._out_h = res["height"]
        self._lbl_res.setText(f"{self._out_w} × {self._out_h}")

        self._log_msg("─" * 50)
        self._log_msg(f"Codec:  {preset['label']}")
        self._log_msg(f"Frames: {start}–{end}  ({end - start + 1} total)")
        self._log_msg(f"Output: {self._out_w}×{self._out_h}  FPS: {self._dsb_fps.value():.3f}")
        self._log_msg(f"Feather: {self._sb_feather.value()}px  Workers: {self._sb_workers.value()}")
        self._log_msg(f"→ {output_path}")

        self._btn_render.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress_bar.setValue(0)
        self._lbl_frames.setText("Starting…")
        self._t_start = time.perf_counter()

        self._worker = CylRenderWorker(
            mov_paths     = mov_paths,
            cam_params    = self._cam_params,
            focal         = focal,
            feather_width = self._sb_feather.value(),
            out_w         = self._out_w,
            out_h         = self._out_h,
            start_frame   = start,
            end_frame     = end,
            fps           = self._dsb_fps.value(),
            ffmpeg_args   = preset["ffmpeg_args"],
            is_exr        = preset["is_exr"],
            output_path   = output_path,
            max_workers   = self._sb_workers.value(),
            color_params  = self._color_params,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.log.connect(self._log_msg)
        self._worker.start()

    def _cancel_render(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._log_msg("Cancelling…")
            self._btn_cancel.setEnabled(False)

    # ── Signals ───────────────────────────────────────────────
    def _on_progress(self, current: int, total: int, eta_secs: float):
        pct = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self._lbl_frames.setText(f"Frame  {current} / {total}")
        self._lbl_eta.setText(f"ETA: {self._fmt_time(eta_secs)}")

    def _on_finished(self, output_path: str):
        elapsed = time.perf_counter() - self._t_start if self._t_start else 0.0
        self._progress_bar.setValue(100)
        self._lbl_frames.setText("Complete")
        self._lbl_eta.setText(f"Total: {self._fmt_time(elapsed)}")
        self._log_msg(f"✓  Done  ({self._fmt_time(elapsed)})")
        self._log_msg(f"   → {output_path}")
        self._btn_render.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _on_error(self, msg: str):
        self._log_msg(f"✕  {msg}")
        self._btn_render.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _log_msg(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _fmt_time(secs: float) -> str:
        if secs <= 0:
            return "—"
        s = int(secs)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m {sec:02d}s"
        if m > 0:
            return f"{m}m {sec:02d}s"
        return f"{sec}s"

    def closeEvent(self, ev):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(3000)
        super().closeEvent(ev)
