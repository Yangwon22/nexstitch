import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from pathlib import Path

from .settings import INPUT_DIR

# ── MPV binary path (must be set before `import mpv`) ─────────
_BIN_DIR = Path(__file__).parent.parent / "bin"
os.environ["PATH"] = str(_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
# Python 3.8+ on Windows requires add_dll_directory for ctypes DLL lookup
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_BIN_DIR))


# ─────────────────────────────────────────────────────────────
# Generic helpers (unchanged)
# ─────────────────────────────────────────────────────────────
def get_mov_files() -> list[Path]:
    seen, unique = set(), []
    for f in sorted(INPUT_DIR.glob("*.mov")) + sorted(INPUT_DIR.glob("*.MOV")):
        if f.name.lower() not in seen:
            seen.add(f.name.lower())
            unique.append(f)
    return unique


def get_video_info(path) -> dict:
    """Returns total/fps/w/h via OpenCV (metadata only — fast)."""
    cap = cv2.VideoCapture(str(path))
    info = dict(
        total=max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        fps=cap.get(cv2.CAP_PROP_FPS) or 24.0,
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def grab_frame(path, idx: int) -> np.ndarray | None:
    """OpenCV single-frame grab — kept for RenderWorker batch export."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



# ─────────────────────────────────────────────────────────────
# MpvPlayer — single camera instance
# ─────────────────────────────────────────────────────────────
class MpvPlayer:
    """One MPV process per camera.
    MPV handles seek / play / pause / time_pos.
    Frame capture uses OpenCV grab_frame() for reliability (vo=null 호환).
    """

    _SEEK_TIMEOUT = 3.0   # seconds

    def __init__(self, cam_id: int):
        self.cam_id       = cam_id
        self._player      = None
        self._path        = ""    # stored for OpenCV capture
        self._fps         = 24.0
        self._last_frame  = 0     # last seeked frame index, used by capture_rgb
        self._loaded      = False
        self._lock        = threading.Lock()

    _LOAD_TIMEOUT = 5.0   # seconds before giving up on wait_until_playing

    # ── lifecycle ────────────────────────────────────────────
    def _make_player(self):
        import mpv
        print(f"[CAM{self.cam_id}] mpv.MPV() 생성 중...")
        player = mpv.MPV(
            vo="null",
            ao="null",
            keep_open="yes",
            pause=True,
            loglevel="error",
        )
        print(f"[CAM{self.cam_id}] mpv.MPV() 생성 완료")
        return player

    def load(self, path: str, fps: float) -> bool:
        with self._lock:
            print(f"[CAM{self.cam_id}] load() 시작: {path}")
            try:
                if self._player is None:
                    self._player = self._make_player()
                self._fps    = max(fps, 1.0)
                self._loaded = False

                self._path = path
                self._last_frame = 0
                print(f"[CAM{self.cam_id}] play() 호출")
                self._player.play(path)

                # wait_until_playing() has no timeout — replace with polling loop
                print(f"[CAM{self.cam_id}] 재생 대기 중 (최대 {self._LOAD_TIMEOUT}s)...")
                deadline = time.monotonic() + self._LOAD_TIMEOUT
                while time.monotonic() < deadline:
                    try:
                        if self._player.core_idle is False:
                            break
                    except Exception:
                        pass
                    time.sleep(0.05)
                else:
                    print(f"[CAM{self.cam_id}] 경고: 타임아웃 — 로드 미완료, 계속 진행")

                self._player.pause = True
                self._loaded = True
                print(f"[CAM{self.cam_id}] load() 완료")
                return True
            except Exception as e:
                print(f"[CAM{self.cam_id}] load() 오류: {e}")
                return False

    def close(self):
        with self._lock:
            if self._player:
                try:
                    self._player.terminate()
                except Exception:
                    pass
                self._player = None
            self._loaded = False

    # ── capture position (no MPV seek) ───────────────────────
    def set_capture_frame(self, frame_idx: int):
        """재생 중 OpenCV 캡처 위치만 갱신 — MPV 시크 없음."""
        with self._lock:
            if self._loaded:
                self._last_frame = frame_idx

    # ── seek ─────────────────────────────────────────────────
    def seek_frame(self, frame_idx: int):
        with self._lock:
            if not self._loaded or self._player is None:
                return
            self._last_frame = frame_idx   # record for OpenCV capture
            t = frame_idx / self._fps
            try:
                self._player.seek(t, reference="absolute", precision="exact")
                deadline = time.monotonic() + self._SEEK_TIMEOUT
                while time.monotonic() < deadline:
                    try:
                        if not self._player.seeking:
                            break
                    except Exception:
                        break
                    time.sleep(0.01)
            except Exception as e:
                print(f"[MpvPlayer CAM{self.cam_id}] seek error: {e}")

    # ── capture ───────────────────────────────────────────────
    def capture_rgb(self) -> np.ndarray | None:
        """OpenCV로 현재 프레임 읽기 (vo=null 환경에서 screenshot_raw 불가 대체)."""
        with self._lock:
            if not self._loaded or not self._path:
                return None
            try:
                rgb = grab_frame(self._path, self._last_frame)
                if rgb is None:
                    print(f"[CAM{self.cam_id}] grab_frame 반환 None (frame={self._last_frame})")
                return rgb
            except Exception as e:
                print(f"[CAM{self.cam_id}] capture_rgb 오류: {e}")
                return None

    # ── playback ─────────────────────────────────────────────
    def play(self):
        with self._lock:
            if self._player and self._loaded:
                self._player.pause = False

    def pause(self):
        with self._lock:
            if self._player:
                self._player.pause = True

    @property
    def time_pos(self) -> float:
        if self._player:
            try:
                v = self._player.time_pos
                return float(v) if v is not None else 0.0
            except Exception:
                pass
        return 0.0

    @property
    def is_paused(self) -> bool:
        if self._player:
            try:
                return bool(self._player.pause)
            except Exception:
                pass
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────
# MpvEngine — 5-camera synchronised manager
# ─────────────────────────────────────────────────────────────
class MpvEngine:
    """Manages up to 5 MpvPlayer instances with frame-accurate sync."""

    def __init__(self):
        self._players: dict[int, MpvPlayer] = {}
        self._fps    = 24.0
        self._total  = 0
        self._paths: dict[int, str] = {}

    # ── loading ───────────────────────────────────────────────
    def load_mapping(self, mapping: dict, progress_cb=None) -> tuple[int, float]:
        """
        mapping: {cam_id: Path|None}
        Loads (or reloads) changed files in parallel via ThreadPoolExecutor.
        progress_cb(done: int, total: int) called after each camera finishes.
        Returns (total_frames, fps) from the first loaded file.
        """
        # Phase 1: metadata + player creation (sequential, fast)
        to_load: list[tuple[int, str, float]] = []
        for cam_id in range(1, 6):
            path = mapping.get(cam_id)
            if path is None:
                continue
            path_str = str(path)
            if self._paths.get(cam_id) == path_str:
                continue  # unchanged
            info = get_video_info(path_str)
            self._fps   = info["fps"]
            self._total = info["total"]
            if cam_id not in self._players:
                self._players[cam_id] = MpvPlayer(cam_id)
            to_load.append((cam_id, path_str, self._fps))

        if not to_load:
            return self._total, self._fps

        # Phase 2: MPV init + file open in parallel
        done_count = 0
        done_lock  = threading.Lock()
        total_count = len(to_load)

        def _load_one(cam_id: int, path_str: str, fps: float):
            nonlocal done_count
            self._players[cam_id].load(path_str, fps)
            self._paths[cam_id] = path_str
            with done_lock:
                done_count += 1
                n = done_count
            if progress_cb:
                progress_cb(n, total_count)

        with ThreadPoolExecutor(max_workers=min(5, total_count)) as pool:
            futures = [pool.submit(_load_one, cam_id, path_str, fps)
                       for cam_id, path_str, fps in to_load]
            for f in as_completed(futures):
                f.result()  # re-raise any exception from worker threads

        return self._total, self._fps

    # ── capture position (playback only, no MPV seek) ────────
    def set_capture_frame(self, frame_idx: int):
        """모든 플레이어의 캡처 위치 갱신 (MPV 시크 없음 — 재생 중 미리보기용)."""
        for p in self._players.values():
            p.set_capture_frame(frame_idx)

    # ── seek (parallel across cameras) ───────────────────────
    def seek_frame(self, frame_idx: int):
        """Seek all loaded players in parallel threads."""
        threads = [
            threading.Thread(target=p.seek_frame, args=(frame_idx,), daemon=True)
            for p in self._players.values()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ── capture ───────────────────────────────────────────────
    def capture_all(self) -> dict[int, np.ndarray]:
        """Capture current decoded frame from each player → RGB ndarrays."""
        result: dict[int, np.ndarray] = {}
        for cam_id, player in self._players.items():
            rgb = player.capture_rgb()
            if rgb is not None:
                result[cam_id] = rgb
        return result

    # ── playback ─────────────────────────────────────────────
    def play_all(self):
        for p in self._players.values():
            p.play()

    def pause_all(self):
        for p in self._players.values():
            p.pause()

    def ref_time_pos(self) -> float:
        """Position of the first loaded player (seconds)."""
        for p in self._players.values():
            if p.is_loaded:
                return p.time_pos
        return 0.0

    # ── properties ───────────────────────────────────────────
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def has_players(self) -> bool:
        return bool(self._players)

    def close(self):
        for p in self._players.values():
            p.close()
        self._players.clear()
        self._paths.clear()
