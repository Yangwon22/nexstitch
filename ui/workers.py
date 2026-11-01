import threading
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from core.video import grab_frame, get_video_info
from core.calibration import undistort, load_lens, match_sift, find_H, compute_H_abs, compute_canvas
from core.stitch import stitch_one
from core.settings import ADJ_PAIRS, STITCH_ORDER, REF_CAM


class MpvLoadWorker(QThread):
    """Runs MPV load_mapping in background in parallel — prevents main thread blocking."""
    loaded   = pyqtSignal(int, float)   # (total_frames, fps)
    progress = pyqtSignal(int, int)     # (done, total)
    error    = pyqtSignal(str)

    def __init__(self, engine, mapping: dict, parent=None):
        super().__init__(parent)
        self._engine  = engine
        self._mapping = mapping

    def run(self):
        try:
            print("[MpvLoadWorker] load_mapping start")

            def on_progress(done: int, total: int):
                print(f"[MpvLoadWorker] Progress: {done}/{total}")
                self.progress.emit(done, total)

            total, fps = self._engine.load_mapping(self._mapping, progress_cb=on_progress)
            print(f"[MpvLoadWorker] Done — total={total} fps={fps}")
            self.loaded.emit(total, fps)
        except Exception as e:
            import traceback
            print(f"[MpvLoadWorker] Exception:\n{traceback.format_exc()}")
            self.error.emit(str(e))


class MpvCaptureOnlyWorker(QThread):
    """Playback preview capture — no MPV seek, no undistort (speed priority)."""
    frames_ready = pyqtSignal(dict)   # {cam_id: rgb_ndarray}

    def __init__(self, engine, frame_idx: int, parent=None):
        super().__init__(parent)
        self._engine    = engine
        self._frame_idx = frame_idx

    def run(self):
        try:
            self._engine.set_capture_frame(self._frame_idx)
            raw = self._engine.capture_all()
            if raw:
                self.frames_ready.emit(raw)
        except Exception:
            pass  # ignore capture failures during playback


class MpvSeekCaptureWorker(QThread):
    """MPV seek → capture → undistort, runs off main thread to keep UI responsive."""
    frames_ready = pyqtSignal(dict)   # {cam_id: rgb_ndarray}
    error        = pyqtSignal(str)

    def __init__(self, engine, frame_idx: int, parent=None):
        super().__init__(parent)
        self._engine    = engine
        self._frame_idx = frame_idx

    def run(self):
        try:
            self._engine.seek_frame(self._frame_idx)
            raw = self._engine.capture_all()
            if not raw:
                self.error.emit("MPV: No frames captured")
                return
            lens_p = load_lens()
            result = {}
            for cam_id, rgb in raw.items():
                result[cam_id] = undistort(rgb, lens_p[f"cam{cam_id}"])
            self.frames_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CylStitchWorker(QThread):
    """Cylindrical projection stitching worker — runs at preview scale."""
    result_ready = pyqtSignal(object, object, float, float)  # rgb, cam_masks, cx_p, cy_p
    error        = pyqtSignal(str)

    def __init__(self, frames: dict, cam_params: dict, focal: float,
                 feather_width: int = 200, scale: float = 0.25,
                 color_params: dict = None, parent=None):
        super().__init__(parent)
        self.frames        = frames
        self.cam_params    = cam_params
        self.focal         = focal
        self.feather_width = feather_width
        self.scale         = scale
        self.color_params  = color_params
        self._abort_ev     = threading.Event()

    def request_abort(self):
        self._abort_ev.set()

    def run(self):
        if self._abort_ev.is_set():
            return
        try:
            import time
            from core.stitch import stitch_cylindrical

            # Apply per-camera color correction before stitching
            frames = dict(self.frames)
            if self.color_params:
                from core.color import apply_per_cam_color
                per_cam = self.color_params.get("per_cam", {})
                for cam_id in list(frames.keys()):
                    p = per_cam.get(cam_id) or per_cam.get(str(cam_id))
                    if p:
                        frames[cam_id] = apply_per_cam_color(frames[cam_id], p)

            t0  = time.perf_counter()
            rgb, cam_masks, cx_p, cy_p = stitch_cylindrical(
                frames, self.cam_params, self.focal,
                self.feather_width, self.scale
            )
            dt = time.perf_counter() - t0
            print(f"[Stitch] scale={self.scale:.2f}  {rgb.shape[1]}×{rgb.shape[0]}px  {dt*1000:.0f}ms"
                  + ("  (aborted)" if self._abort_ev.is_set() else ""))
            if not self._abort_ev.is_set():
                self.result_ready.emit(rgb, cam_masks, cx_p, cy_p)
        except Exception as e:
            if not self._abort_ev.is_set():
                self.error.emit(str(e))


class FrameLoader(QThread):
    """Async single-frame loader."""
    frame_ready = pyqtSignal(int, object)   # (cam_id, rgb_ndarray|None)

    def __init__(self, cam_id: int, path: str, idx: int, lens_p: dict, parent=None):
        super().__init__(parent)
        self.cam_id  = cam_id
        self.path    = path
        self.idx     = idx
        self.lens_p  = lens_p

    def run(self):
        rgb = grab_frame(self.path, self.idx)
        if rgb is not None and self.lens_p:
            rgb = undistort(rgb, self.lens_p)
        self.frame_ready.emit(self.cam_id, rgb)


class StitchWorker(QThread):
    """Stitch current frame."""
    result_ready = pyqtSignal(object)   # RGB ndarray
    error        = pyqtSignal(str)

    def __init__(self, frames: dict, homo_data: dict,
                 blend: str, intensity: float, brightness: float, contrast: float,
                 cam_masks: dict | None = None, rig_layout: dict | None = None,
                 canvas_layout: dict | None = None, parent=None):
        super().__init__(parent)
        self.frames        = frames
        self.homo_data     = homo_data
        self.blend         = blend
        self.intensity     = intensity
        self.brightness    = brightness
        self.contrast      = contrast
        self.cam_masks     = cam_masks
        # canvas_layout takes priority over rig_layout when provided
        self.rig_layout    = canvas_layout if canvas_layout is not None else rig_layout

    def run(self):
        try:
            ref_bgr = cv2.cvtColor(self.frames[REF_CAM], cv2.COLOR_RGB2BGR)
            result  = stitch_one(
                self.frames, self.homo_data["H_abs"], self.homo_data["canvas"],
                self.blend, ref_bgr,
                self.intensity, self.brightness, self.contrast,
                self.cam_masks, self.rig_layout,
            )
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CylRenderWorker(QThread):
    """
    Full cylindrical render pipeline.

    Two output modes selected by `is_exr`:

    pipe mode (is_exr=False)  — default
        ffmpeg reads raw BGR24 from stdin and encodes to the requested codec.
        `ffmpeg_args` carries the output-side codec flags verbatim.
        `output_path` is the destination video file.

    EXR mode (is_exr=True)
        Each stitched frame is saved as a float32 OpenEXR file directly via
        cv2.imwrite (requires OpenCV built with OpenEXR support).
        `output_path` is the destination directory.

    Pipeline architecture (both modes):
        reader thread  → sequential VideoCapture reads → ThreadPoolExecutor submit
        worker threads → undistort + stitch + resize   (parallel)
        writer (run()) → ordered collection → ffmpeg stdin  OR  imwrite
    """
    progress = pyqtSignal(int, int, float)   # (current, total, eta_secs)
    finished = pyqtSignal(str)               # output path (file or directory)
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, mov_paths: dict, cam_params: dict, focal: float,
                 feather_width: int, out_w: int, out_h: int,
                 start_frame: int, end_frame: int,
                 fps: float = 24.0,
                 ffmpeg_args: list = None,
                 is_exr: bool = False,
                 output_path: str = "",
                 max_workers: int = 4,
                 color_params: dict = None,
                 parent=None):
        super().__init__(parent)
        self._mov_paths     = mov_paths        # {cam_id: path_str}
        self._cam_params    = cam_params
        self._focal         = focal
        self._feather_width = feather_width
        self._out_w         = out_w
        self._out_h         = out_h
        self._start_frame   = start_frame
        self._end_frame     = end_frame
        self._fps           = fps
        self._ffmpeg_args   = ffmpeg_args or []
        self._is_exr        = is_exr
        self._output_path   = output_path
        self._max_workers   = max_workers
        self._color_params  = color_params or {}
        self._abort_ev      = threading.Event()

    def abort(self):
        self._abort_ev.set()

    # ── main render loop ──────────────────────────────────────
    def run(self):
        import subprocess
        import queue as _queue
        from core.settings import BASE_DIR

        proc = None
        caps: dict[int, cv2.VideoCapture] = {}

        try:
            lens_p = load_lens()

            # ── Open VideoCaptures ────────────────────────────
            for cam_id, path in self._mov_paths.items():
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    caps[cam_id] = cap
                else:
                    self.log.emit(f"⚠ CAM{cam_id}: cannot open {path}")
            if not caps:
                self.error.emit("No video files could be opened")
                return

            sample_cap = next(iter(caps.values()))
            img_w = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log.emit(f"Source: {img_w}×{img_h}  |  Output: {self._out_w}×{self._out_h}")

            # ── Precompute remap maps (once) ──────────────────
            self.log.emit("Precomputing cylindrical remap maps…")
            from core.stitch import precompute_remap_maps, stitch_frame_with_maps
            remap_maps, pano_w, pano_h, _cx, _cy = precompute_remap_maps(
                self._cam_params, self._focal, img_w, img_h,
            )
            self.log.emit(f"Natural pano: {pano_w}×{pano_h}  →  {self._out_w}×{self._out_h}")

            # ── Mode: start ffmpeg OR prepare EXR dir ─────────
            exr_dir: Path | None = None
            stderr_thread = None

            if self._is_exr:
                exr_dir = Path(self._output_path)
                exr_dir.mkdir(parents=True, exist_ok=True)
                self.log.emit(f"EXR sequence → {exr_dir}")
            else:
                ffmpeg_bin = BASE_DIR / "bin" / "ffmpeg.exe"
                ffmpeg_cmd = str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"

                cmd = [
                    ffmpeg_cmd, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{self._out_w}x{self._out_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(self._fps),
                    "-i", "pipe:0",
                    *self._ffmpeg_args,
                    self._output_path,
                ]
                self.log.emit("ffmpeg: " + " ".join(cmd))

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                except FileNotFoundError:
                    self.error.emit(
                        "ffmpeg not found.\n"
                        "Place ffmpeg.exe in bin/ or add it to PATH."
                    )
                    return

                # Drain stderr in background to prevent deadlock
                def _drain_stderr():
                    for raw in proc.stderr:
                        line = raw.decode("utf-8", errors="replace").rstrip()
                        if line and ("frame=" in line or "error" in line.lower()
                                     or "warning" in line.lower()):
                            self.log.emit(line)
                stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
                stderr_thread.start()

            # ── Seek to start frame ───────────────────────────
            for cap in caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

            total   = self._end_frame - self._start_frame + 1
            t_start = time.perf_counter()

            # ── Pre-extract color params for use in closure ────
            _color_per_cam = self._color_params.get("per_cam", {})
            _color_output  = self._color_params.get("output",  {})

            # ── Per-frame processing (returns uint8 BGR ndarray) ──
            def process_frame(frames_bgr: dict) -> np.ndarray:
                """Undistort → per-cam color → stitch → output color → resize."""
                from core.color import apply_per_cam_color, apply_output_color as _aoc
                undistorted: dict[int, np.ndarray] = {}
                for cam_id, bgr in frames_bgr.items():
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    p   = lens_p.get(f"cam{cam_id}")
                    if p:
                        rgb = undistort(rgb, p)
                    # Per-camera color correction
                    pc = _color_per_cam.get(cam_id) or _color_per_cam.get(str(cam_id))
                    if pc:
                        rgb = apply_per_cam_color(rgb, pc)
                    undistorted[cam_id] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                result_bgr = stitch_frame_with_maps(
                    undistorted, remap_maps,
                    self._cam_params, self._focal,
                    img_w, img_h, pano_w, pano_h,
                    self._feather_width,
                )

                if result_bgr.shape[1] != self._out_w or result_bgr.shape[0] != self._out_h:
                    result_bgr = cv2.resize(
                        result_bgr, (self._out_w, self._out_h),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                # Output color correction (post-stitch)
                if _color_output:
                    rgb_out = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                    rgb_out = _aoc(rgb_out, _color_output)
                    result_bgr = cv2.cvtColor(rgb_out, cv2.COLOR_BGR2RGB)

                return result_bgr

            # ── Pipeline: reader thread + ordered writer ──────
            write_queue: _queue.Queue = _queue.Queue(maxsize=self._max_workers * 2)

            def reader():
                for fi in range(total):
                    if self._abort_ev.is_set():
                        break
                    frames_bgr: dict[int, np.ndarray] = {}
                    for cam_id, cap in caps.items():
                        ok, bgr = cap.read()
                        if ok:
                            frames_bgr[cam_id] = bgr
                    if len(frames_bgr) == len(caps):
                        fut = executor.submit(process_frame, frames_bgr)
                    else:
                        self.log.emit(
                            f"Frame {self._start_frame + fi}: "
                            f"{len(frames_bgr)}/{len(caps)} cameras — skipped"
                        )
                        fut = None
                    write_queue.put((fi, fut))
                write_queue.put(None)   # sentinel

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                read_thread = threading.Thread(target=reader, daemon=True)
                read_thread.start()

                fi_write = 0
                while True:
                    item = write_queue.get()
                    if item is None:
                        break

                    _fi, fut = item
                    if fut is not None:
                        try:
                            result_bgr = fut.result()
                            if self._is_exr:
                                # float32 [0,1] for EXR (linear scene-referred)
                                frame_idx = self._start_frame + _fi
                                exr_path  = exr_dir / f"frame_{frame_idx:06d}.exr"
                                result_f32 = result_bgr.astype(np.float32) / 255.0
                                ok = cv2.imwrite(str(exr_path), result_f32)
                                if not ok:
                                    self.log.emit(
                                        f"⚠ EXR write failed for {exr_path.name} "
                                        f"(OpenCV OpenEXR support required)"
                                    )
                            else:
                                proc.stdin.write(result_bgr.tobytes())
                        except Exception as exc:
                            self.log.emit(f"Frame {self._start_frame + _fi} error: {exc}")

                    fi_write += 1
                    elapsed = time.perf_counter() - t_start
                    eta     = (elapsed / fi_write) * (total - fi_write) \
                              if fi_write < total else 0.0
                    self.progress.emit(fi_write, total, eta)

                    if self._abort_ev.is_set():
                        break

                read_thread.join()

            # ── Finish ────────────────────────────────────────
            if self._is_exr:
                if self._abort_ev.is_set():
                    self.log.emit("Render aborted by user")
                    return
                self.log.emit(f"✓ EXR sequence → {self._output_path}")
                self.finished.emit(self._output_path)
            else:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

                if self._abort_ev.is_set():
                    proc.terminate()
                    self.log.emit("Render aborted by user")
                    return

                proc.wait()
                if stderr_thread:
                    stderr_thread.join(timeout=5)

                if proc.returncode != 0:
                    self.log.emit(f"⚠ ffmpeg exited with code {proc.returncode}")
                    self.error.emit(f"ffmpeg failed (code {proc.returncode})")
                    return

                self.log.emit(f"✓ Encoded → {self._output_path}")
                self.finished.emit(self._output_path)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

        finally:
            for cap in caps.values():
                cap.release()
            if proc and proc.returncode is None:
                try:
                    proc.stdin.close()
                    proc.terminate()
                except OSError:
                    pass


class RigCalibWorker(QThread):
    """Rig calibration: SIFT matching + homography computation."""
    progress = pyqtSignal(int, str)      # (percent, message)
    finished = pyqtSignal(dict)          # homography data dict
    error    = pyqtSignal(str)

    def __init__(self, frame_data: dict, overlap: dict, ref_w: int, ref_h: int, parent=None):
        super().__init__(parent)
        self.frame_data = frame_data
        self.overlap    = overlap
        self.ref_w      = ref_w
        self.ref_h      = ref_h

    def run(self):
        try:
            H_adj  = {}
            n_pairs = len(ADJ_PAIRS)
            for step, (cam_a, cam_b) in enumerate(ADJ_PAIRS):
                pct = int(step / n_pairs * 80)
                self.progress.emit(pct, f"CAM{cam_a} ↔ CAM{cam_b} Matching...")
                img_a = self.frame_data[cam_a]
                img_b = self.frame_data[cam_b]
                ha, wa = img_a.shape[:2]
                hb, wb = img_b.shape[:2]
                ov = self.overlap.get((cam_a, cam_b), 30) / 100.0

                mask_a = np.zeros((ha, wa), np.uint8)
                mask_b = np.zeros((hb, wb), np.uint8)
                mask_a[:, int(wa * (1 - ov)):] = 255
                mask_b[:, :int(wb * ov)]        = 255

                src, dst = match_sift(img_a, img_b, mask_a, mask_b)
                n_m = len(src) if src is not None else 0
                if n_m < 10:
                    src, dst = match_sift(img_a, img_b)

                H = find_H(src, dst)
                if H is None:
                    self.error.emit(f"CAM{cam_a}↔CAM{cam_b}: Matching failed ({len(src) if src is not None else 0})")
                    return
                H_adj[(cam_a, cam_b)] = H

            self.progress.emit(85, "Computing absolute homography...")
            H_abs  = compute_H_abs(H_adj)
            canvas = compute_canvas(H_abs, self.ref_w, self.ref_h)

            if canvas["out_w"] > 30000 or canvas["out_h"] > 15000:
                self.error.emit(f"Canvas size exceeded ({canvas['out_w']}×{canvas['out_h']}px)")
                return

            self.progress.emit(100, "Done")
            self.finished.emit({
                "H_abs":  {str(k): v.tolist() for k, v in H_abs.items()},
                "canvas": canvas,
            })
        except Exception as e:
            self.error.emit(str(e))


class RenderWorker(QThread):
    """Full range rendering."""
    progress = pyqtSignal(int, str)   # (percent, message)
    finished = pyqtSignal(str)        # output dir path
    error    = pyqtSignal(str)

    def __init__(self, mov_files: list, homo_data: dict, lens_params: dict,
                 cam_masks: dict, blend: str, scale: float,
                 start_frame: int, end_frame: int,
                 intensity: float, brightness: float, contrast: float,
                 out_fmt: str, jpeg_q: int, frames_dir,
                 rig_layout: dict | None = None, parent=None):
        super().__init__(parent)
        self.mov_files    = mov_files
        self.homo_data    = homo_data
        self.lens_params  = lens_params
        self.cam_masks    = cam_masks
        self.blend        = blend
        self.scale        = scale
        self.start_frame  = start_frame
        self.end_frame    = end_frame
        self.intensity    = intensity
        self.brightness   = brightness
        self.contrast     = contrast
        self.out_fmt      = out_fmt
        self.jpeg_q       = jpeg_q
        self.frames_dir   = frames_dir
        self.rig_layout   = rig_layout
        self._abort       = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            canvas = self.homo_data["canvas"]
            sc     = self.scale
            S      = np.diag([sc, sc, 1.0])
            H_abs_scaled = {k: (S @ np.array(v, np.float64)).tolist()
                            for k, v in self.homo_data["H_abs"].items()}
            render_canvas = dict(
                x_min=canvas["x_min"], y_min=canvas["y_min"],
                out_w=int(canvas["out_w"] * sc), out_h=int(canvas["out_h"] * sc),
            )
            caps  = [cv2.VideoCapture(str(f)) for f in self.mov_files]
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            total   = self.end_frame - self.start_frame + 1
            written = 0
            ext     = "jpg" if "JPEG" in self.out_fmt else "png"

            for fi in range(total):
                if self._abort:
                    break
                fidx      = self.start_frame + fi
                frames_rgb = {}
                for cam_id in range(1, 6):
                    ok, bgr = caps[cam_id - 1].read()
                    if ok:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        frames_rgb[cam_id] = undistort(rgb, self.lens_params[f"cam{cam_id}"])

                if len(frames_rgb) < 5:
                    continue

                ref_bgr    = cv2.cvtColor(frames_rgb[REF_CAM], cv2.COLOR_RGB2BGR)
                result_rgb = stitch_one(
                    frames_rgb, H_abs_scaled, render_canvas,
                    self.blend, ref_bgr,
                    self.intensity, self.brightness, self.contrast,
                    self.cam_masks, self.rig_layout,
                )
                out_path = self.frames_dir / f"frame_{fidx:06d}.{ext}"
                result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB)
                if ext == "jpg":
                    cv2.imwrite(str(out_path), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
                else:
                    cv2.imwrite(str(out_path), result_bgr)
                written += 1
                self.progress.emit(int((fi + 1) / total * 100),
                                   f"Frame {fidx} / {self.end_frame}  ({fi+1}/{total})")

            for cap in caps:
                cap.release()
            self.finished.emit(str(self.frames_dir))
        except Exception as e:
            self.error.emit(str(e))
