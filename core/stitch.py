"""
NexStitch  ─  Cylindrical Projection Stitching
Mistika VR 방식: Yaw/Pitch/Roll + 원통 투영

GPU acceleration
────────────────
When OpenCV is built with CUDA support the CUDA path is selected automatically
(cv2.cuda.getCudaEnabledDeviceCount() > 0).  CPU fallback is always available.

Standard pip opencv-contrib-python does NOT include CUDA.
To enable GPU, build OpenCV from source with -DWITH_CUDA=ON or install a
CUDA-enabled wheel, e.g.:
    pip install opencv-contrib-python-headless   ← still no CUDA (pip official)
    # Build from source:
    # https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html

Map caching
───────────
Remap maps and feather-weight arrays are expensive to compute but change only
when camera parameters change.  They are cached at module level (preview path)
or precomputed once before the render loop (render path), so per-frame cost is
limited to the GPU/CPU warp + weighted accumulation.
"""
import time
import numpy as np
import cv2
from math import sin, cos, radians, pi


# ── CUDA detection ────────────────────────────────────────────
_CUDA_AVAIL:    bool = False
_CUDA_DEV_NAME: str  = ""

try:
    _n = cv2.cuda.getCudaEnabledDeviceCount()
    if _n > 0:
        _CUDA_AVAIL = True
        try:
            _CUDA_DEV_NAME = cv2.cuda.DeviceInfo(0).name()
        except Exception:
            _CUDA_DEV_NAME = "GPU #0"
        print(f"[Stitch] CUDA enabled — {_CUDA_DEV_NAME}  ({_n} device(s))")
    else:
        print("[Stitch] CUDA: no CUDA device found, CPU path active")
except Exception as _e:
    print(f"[Stitch] CUDA unavailable ({_e}), CPU path active")


# ── GPU helpers ───────────────────────────────────────────────
def _upload_gpu(arr: np.ndarray):
    """Upload ndarray to a new GpuMat.  Returns None on any error."""
    if not _CUDA_AVAIL:
        return None
    try:
        g = cv2.cuda_GpuMat()
        g.upload(arr)
        return g
    except Exception:
        return None


def _remap_frame(src: np.ndarray,
                 map_x: np.ndarray, map_y: np.ndarray,
                 g_mx=None, g_my=None) -> np.ndarray:
    """
    Warp `src` with precomputed remap maps.

    CUDA path  — when _CUDA_AVAIL:
        Uploads `src` to GPU each call (fast: ~0.5 ms at 1080p).
        Uses pre-uploaded g_mx/g_my when provided (zero-copy from cache),
        otherwise uploads maps fresh (safe for multi-threaded render pool).

    CPU path   — always-available fallback.
    """
    if _CUDA_AVAIL:
        try:
            g_src = cv2.cuda_GpuMat()
            g_src.upload(src)
            # Fresh upload when not pre-cached (render pool: thread-safe)
            gx = g_mx if g_mx is not None else _upload_gpu(map_x)
            gy = g_my if g_my is not None else _upload_gpu(map_y)
            if gx is not None and gy is not None:
                g_out = cv2.cuda.remap(g_src, gx, gy,
                                       interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
                return g_out.download()
        except Exception:
            pass   # fall through to CPU

    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Rotation matrix ───────────────────────────────────────────
def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """YPR rotation (degrees). Yaw=Y, Pitch=X, Roll=Z."""
    y = radians(yaw);  p = radians(pitch);  r = radians(roll)
    Ry = np.array([[ cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]], np.float64)
    Rp = np.array([[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]], np.float64)
    Rr = np.array([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]], np.float64)
    return Ry @ Rp @ Rr


# ── Panorama dimensions ───────────────────────────────────────
def compute_pano_dims(cam_params: dict, focal: float,
                      img_w: int, img_h: int) -> tuple[int, int, float, float]:
    """Compute panorama (w, h, cx, cy) from camera params."""
    import math
    yaws    = [v.get("yaw",   0.0) for v in cam_params.values()]
    pitches = [v.get("pitch", 0.0) for v in cam_params.values()]
    fov_h   = 2.0 * math.atan(img_w / (2.0 * focal))
    fov_v   = 2.0 * math.atan(img_h / (2.0 * focal))
    yaw_min_rad   = radians(min(yaws))    - fov_h / 2.0 - 0.05
    yaw_max_rad   = radians(max(yaws))    + fov_h / 2.0 + 0.05
    pitch_min_rad = radians(min(pitches)) - fov_v / 2.0 - 0.05
    pitch_max_rad = radians(max(pitches)) + fov_v / 2.0 + 0.05
    pano_w = max(4, int(focal * (yaw_max_rad   - yaw_min_rad)))
    pano_h = max(4, int(focal * (pitch_max_rad - pitch_min_rad)))
    cx_p   = float(-yaw_min_rad   * focal)
    cy_p   = float(-pitch_min_rad * focal)
    return pano_w, pano_h, cx_p, cy_p


# ── Per-camera remap map ──────────────────────────────────────
def cylindrical_map(yaw: float, pitch: float, roll: float,
                    focal: float, img_w: int, img_h: int,
                    pano_w: int, pano_h: int,
                    cx_p: float, cy_p: float) -> tuple[np.ndarray, np.ndarray]:
    """
    For each panorama pixel (X,Y) compute the source camera pixel (map_x, map_y).
    Returns float32 arrays for cv2.remap / cv2.cuda.remap.
    """
    R = rotation_matrix(yaw, pitch, roll)
    cx_cam, cy_cam = img_w / 2.0, img_h / 2.0

    Xp = np.arange(pano_w, dtype=np.float64)
    Yp = np.arange(pano_h, dtype=np.float64)
    Xp, Yp = np.meshgrid(Xp, Yp)           # (pano_h, pano_w)

    theta = (Xp - cx_p) / focal            # horizontal angle (rad)
    phi   = (Yp - cy_p) / focal            # vertical angle   (rad, linear approx)

    X3 = np.sin(theta)
    Y3 = phi
    Z3 = np.cos(theta)

    d   = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=0)   # (3, N)
    cd  = R.T @ d                                                    # (3, N)

    cx_r = cd[0].reshape(pano_h, pano_w)
    cy_r = cd[1].reshape(pano_h, pano_w)
    cz_r = cd[2].reshape(pano_h, pano_w)

    with np.errstate(invalid='ignore', divide='ignore'):
        x_img = np.where(cz_r > 1e-3, focal * cx_r / cz_r + cx_cam, -1.0)
        y_img = np.where(cz_r > 1e-3, focal * cy_r / cz_r + cy_cam, -1.0)

    return x_img.astype(np.float32), y_img.astype(np.float32)


# ── Feather blend ─────────────────────────────────────────────
def feather_blend(warped_list: list, mask_list: list,
                  feather_width: int = 200) -> np.ndarray:
    """Standard feather blend — computes distanceTransform per call."""
    h, w = warped_list[0].shape[:2]
    acc  = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w),    np.float32)
    fw   = max(1.0, float(feather_width))
    for img, mask in zip(warped_list, mask_list):
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        w_   = np.minimum(dist / fw, 1.0)
        acc  += img.astype(np.float32) * w_[:, :, np.newaxis]
        wsum += w_
    wsum = np.where(wsum < 1e-6, 1.0, wsum)
    return (acc / wsum[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


def feather_blend_preweighted(warped_list: list, dist_list: list,
                               feather_width: int = 200) -> np.ndarray:
    """
    Feather blend with pre-computed distance arrays.

    Identical result to feather_blend() but avoids calling distanceTransform
    per frame — the caller supplies float32 dist arrays computed once when
    the remap maps were built.  This removes ~5–20 ms per camera from the
    per-frame hot path.
    """
    h, w = warped_list[0].shape[:2]
    acc  = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w),    np.float32)
    fw   = max(1.0, float(feather_width))
    for img, dist in zip(warped_list, dist_list):
        w_   = np.minimum(dist / fw, 1.0)
        acc  += img.astype(np.float32) * w_[:, :, np.newaxis]
        wsum += w_
    wsum = np.where(wsum < 1e-6, 1.0, wsum)
    return (acc / wsum[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


# ── Preview map cache ─────────────────────────────────────────
# Remap maps + distance transforms are expensive to compute but change only
# when camera parameters change.  The preview cache avoids recomputing them
# on every frame seek when the user has not adjusted any parameter.
#
# Structure:  _pc_maps[cam_id] = (mx, my, dist, valid, g_mx, g_my)
#   mx, my  : float32 remap maps
#   dist    : float32 distance-transform of the valid region
#   valid   : uint8 mask (0/255) of pixels with valid source coords
#   g_mx,g_my : pre-uploaded GpuMat (None when CUDA unavailable)
_pc_key:  object = None
_pc_maps: dict   = {}
_pc_dims: tuple  = ()   # (pano_w, pano_h, cx_p, cy_p)


def _make_params_key(active_p: dict, focal_s: float,
                     img_w: int, img_h: int) -> tuple:
    """Build a hashable fingerprint for the map cache."""
    parts: list = [round(focal_s, 4), img_w, img_h]
    for key in sorted(active_p):
        p = active_p[key]
        parts += [round(p.get("yaw",   0.0), 4),
                  round(p.get("pitch", 0.0), 4),
                  round(p.get("roll",  0.0), 4),
                  round(p.get("k1",    0.0), 6),
                  round(p.get("k2",    0.0), 6),
                  bool(p.get("enabled", True))]
    return tuple(parts)


def _build_maps_for_params(active_p: dict, focal_s: float,
                            img_w: int, img_h: int
                            ) -> tuple[dict, int, int, float, float]:
    """
    Compute remap maps, validity masks, distance transforms, and GPU uploads
    for all cameras in active_p.  Returns (maps_dict, pano_w, pano_h, cx_p, cy_p).

    maps_dict[cam_id] = (mx, my, dist, valid, g_mx, g_my)
    """
    pano_w, pano_h, cx_p, cy_p = compute_pano_dims(active_p, focal_s, img_w, img_h)
    maps: dict = {}
    for key in sorted(active_p):
        cam_id = int(key[3:])
        p      = active_p[key]
        mx, my = cylindrical_map(
            p.get("yaw", 0.0), p.get("pitch", 0.0), p.get("roll", 0.0),
            focal_s, img_w, img_h, pano_w, pano_h, cx_p, cy_p,
        )
        valid = ((mx >= 0) & (mx < img_w) &
                 (my >= 0) & (my < img_h)).astype(np.uint8) * 255
        dist  = cv2.distanceTransform(valid, cv2.DIST_L2, 5)   # float32
        g_mx  = _upload_gpu(mx)
        g_my  = _upload_gpu(my)
        maps[cam_id] = (mx, my, dist, valid, g_mx, g_my)
    return maps, pano_w, pano_h, cx_p, cy_p


# ── Overlay rendering ─────────────────────────────────────────
_CAM_COLORS_RGB = {
    1: (0, 200, 255),
    2: (100, 255, 100),
    3: (255, 160, 0),
    4: (255, 80,  80),
    5: (200, 100, 255),
}


def render_overlays(cam_masks: dict, pano_w: int, pano_h: int,
                    show_bounds: bool, show_feather: bool, show_grid: bool,
                    focal: float, cx_p: float, cy_p: float) -> np.ndarray:
    """Returns RGBA (H,W,4) overlay for the panorama."""
    ov = np.zeros((pano_h, pano_w, 4), np.uint8)

    if show_feather and len(cam_masks) >= 2:
        total = np.zeros((pano_h, pano_w), np.uint8)
        for mask in cam_masks.values():
            if mask.shape == (pano_h, pano_w):
                total += (mask > 0).astype(np.uint8)
        multi = total >= 2
        ov[multi, 1] = 180
        ov[multi, 3] = 90

    if show_bounds:
        for cam_id, mask in cam_masks.items():
            if mask.shape != (pano_h, pano_w):
                continue
            col = _CAM_COLORS_RGB.get(cam_id, (200, 200, 200))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp = np.zeros((pano_h, pano_w, 3), np.uint8)
            cv2.drawContours(tmp, contours, -1, col, 2)
            drawn = tmp.max(axis=2) > 0
            ov[drawn, :3] = tmp[drawn]
            ov[drawn, 3]  = 220

    if show_grid:
        for deg in range(-180, 181, 45):
            x = int(cx_p + radians(deg) * focal)
            if 0 <= x < pano_w:
                ov[:, x, :3] = 180;  ov[:, x, 3] = 70
        for deg in range(-90, 91, 30):
            y = int(cy_p + radians(deg) * focal)
            if 0 <= y < pano_h:
                ov[y, :, :3] = 180;  ov[y, :, 3] = 70

    return ov


# ── Main preview stitch ───────────────────────────────────────
def stitch_cylindrical(frames: dict, cam_params: dict, focal: float,
                       feather_width: int = 200,
                       scale: float = 1.0) -> tuple[np.ndarray, dict, float, float]:
    """
    Cylindrical panorama stitch with map caching and optional CUDA acceleration.

    frames     : {cam_id: RGB ndarray}
    cam_params : {"cam1": {yaw, pitch, roll, k1, k2, enabled}, ...}
    scale      : downscale factor (e.g. 0.25 for preview, 1.0 for full)

    Returns: (result_rgb, cam_masks, cx_pano, cy_pano)

    Map cache
    ─────────
    Remap maps and distance transforms are recomputed only when the
    effective parameters (focal*scale, resolution, YPR, k1/k2) change.
    On a cache hit the warp step goes straight to GPU/CPU remap + blend.
    """
    global _pc_key, _pc_maps, _pc_dims

    if not frames:
        return np.zeros((1080, 1920, 3), np.uint8), {}, 960.0, 540.0

    sample         = next(iter(frames.values()))
    img_h_f, img_w_f = sample.shape[:2]

    focal_s = focal * scale
    img_w   = max(4, int(img_w_f * scale))
    img_h   = max(4, int(img_h_f * scale))

    active = {cam_id: rgb for cam_id, rgb in frames.items()
              if cam_params.get(f"cam{cam_id}", {}).get("enabled", True)}
    if not active:
        return np.zeros((img_h, img_w, 3), np.uint8), {}, img_w / 2.0, img_h / 2.0

    active_p = {f"cam{k}": cam_params[f"cam{k}"] for k in active
                if f"cam{k}" in cam_params}

    # ── Map cache check ──────────────────────────────────────
    cache_key = _make_params_key(active_p, focal_s, img_w, img_h)
    t_map = 0.0
    if cache_key != _pc_key:
        t0 = time.perf_counter()
        _pc_maps, pano_w, pano_h, cx_p, cy_p = _build_maps_for_params(
            active_p, focal_s, img_w, img_h)
        _pc_dims = (pano_w, pano_h, cx_p, cy_p)
        _pc_key  = cache_key
        t_map    = time.perf_counter() - t0

    pano_w, pano_h, cx_p, cy_p = _pc_dims

    # ── Per-frame: resize → undistort → remap → blend ────────
    t0 = time.perf_counter()
    warped_list: list = []
    dist_list:   list = []
    cam_masks:   dict = {}

    for cam_id in sorted(active.keys()):
        if cam_id not in _pc_maps:
            continue
        p  = cam_params[f"cam{cam_id}"]
        mx, my, dist, valid, g_mx, g_my = _pc_maps[cam_id]

        rgb = active[cam_id]
        if scale != 1.0:
            rgb = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)

        k1, k2 = p.get("k1", 0.0), p.get("k2", 0.0)
        if k1 != 0.0 or k2 != 0.0:
            K        = np.array([[focal_s, 0, img_w / 2.0],
                                  [0, focal_s, img_h / 2.0],
                                  [0, 0, 1.0]], np.float64)
            dist_coef = np.array([k1, k2, 0.0, 0.0], np.float64)
            bgr_      = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb       = cv2.cvtColor(cv2.undistort(bgr_, K, dist_coef),
                                     cv2.COLOR_BGR2RGB)

        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        warped  = _remap_frame(img_bgr, mx, my, g_mx, g_my)
        warped_list.append(warped)
        dist_list.append(dist)
        cam_masks[cam_id] = valid

    if not warped_list:
        return np.zeros((pano_h, pano_w, 3), np.uint8), {}, cx_p, cy_p

    result_bgr = feather_blend_preweighted(warped_list, dist_list, feather_width)
    t_warp     = time.perf_counter() - t0

    mode     = "GPU" if _CUDA_AVAIL else "CPU"
    map_tag  = f"map:{t_map*1000:.0f}ms" if t_map > 0 else "map:cached"
    print(f"[Stitch] {mode}  {pano_w}×{pano_h}px  scale={scale:.2f}  "
          f"{map_tag}  warp+blend:{t_warp*1000:.0f}ms  "
          f"total:{(t_map + t_warp)*1000:.0f}ms")

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), cam_masks, cx_p, cy_p


# ── Improve Yaw offsets ───────────────────────────────────────
def improve_yaw(frames: dict, cam_params: dict, focal: float,
                adj_pairs: list) -> dict:
    """SIFT-based Yaw micro-adjustment for adjacent camera pairs."""
    SCALE   = 0.25
    focal_s = focal * SCALE
    sample  = next(iter(frames.values()))
    img_h_f, img_w_f = sample.shape[:2]
    img_w   = max(4, int(img_w_f * SCALE))
    img_h   = max(4, int(img_h_f * SCALE))

    active_p = {k: v for k, v in cam_params.items()
                if any(f"cam{cid}" == k for cid, _ in adj_pairs)
                or any(f"cam{cid}" == k for _, cid in adj_pairs)}

    pano_w, pano_h, cx_p, cy_p = compute_pano_dims(
        active_p or cam_params, focal_s, img_w, img_h)

    warped_map: dict = {}
    mask_map:   dict = {}
    for cam_id, rgb in frames.items():
        key = f"cam{cam_id}"
        if key not in cam_params:
            continue
        p   = cam_params[key]
        img = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)

        k1, k2 = p.get("k1", 0.0), p.get("k2", 0.0)
        if k1 != 0.0 or k2 != 0.0:
            K    = np.array([[focal_s, 0, img_w/2],
                             [0, focal_s, img_h/2],
                             [0, 0, 1]], np.float64)
            dist = np.array([k1, k2, 0, 0], np.float64)
            bgr_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img  = cv2.cvtColor(cv2.undistort(bgr_, K, dist), cv2.COLOR_BGR2RGB)

        bgr    = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mx, my = cylindrical_map(p.get("yaw", 0), p.get("pitch", 0),
                                  p.get("roll", 0),
                                  focal_s, img_w, img_h, pano_w, pano_h,
                                  cx_p, cy_p)
        # Use CPU remap for improve_yaw (small scale, no hot path)
        warped = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid  = ((mx >= 0) & (mx < img_w) &
                  (my >= 0) & (my < img_h)).astype(np.uint8) * 255
        warped_map[cam_id] = warped
        mask_map[cam_id]   = valid

    updated = {k: dict(v) for k, v in cam_params.items()}
    det     = cv2.SIFT_create(nfeatures=400)
    bfm     = cv2.BFMatcher(cv2.NORM_L2)

    for cam_a, cam_b in adj_pairs:
        if cam_a not in warped_map or cam_b not in warped_map:
            continue
        overlap = (mask_map[cam_a] > 0) & (mask_map[cam_b] > 0)
        if overlap.sum() < 200:
            continue
        ol_u8 = overlap.astype(np.uint8) * 255
        ga    = cv2.cvtColor(warped_map[cam_a], cv2.COLOR_BGR2GRAY)
        gb    = cv2.cvtColor(warped_map[cam_b], cv2.COLOR_BGR2GRAY)
        kpa, da = det.detectAndCompute(ga, ol_u8)
        kpb, db = det.detectAndCompute(gb, ol_u8)
        if da is None or db is None or len(da) < 5 or len(db) < 5:
            continue
        ms   = bfm.knnMatch(da, db, k=2)
        good = [m for m, n in ms if m.distance < 0.75 * n.distance]
        if len(good) < 5:
            continue
        dx_vals = [kpb[m.trainIdx].pt[0] - kpa[m.queryIdx].pt[0] for m in good]
        dx_med  = float(np.median(dx_vals))
        dyaw    = -dx_med / focal_s * (180.0 / pi)
        key_b   = f"cam{cam_b}"
        updated[key_b]["yaw"] = updated[key_b].get("yaw", 0.0) + dyaw

    return updated


# ── Render helpers (precomputed maps for fast batch render) ────
def precompute_remap_maps(cam_params: dict, focal: float,
                           img_w: int, img_h: int
                           ) -> tuple[dict, int, int, float, float]:
    """
    Precompute cylindrical remap maps for all active cameras.
    Call once before the render loop; pass the result to stitch_frame_with_maps.

    Returns (maps, pano_w, pano_h, cx_p, cy_p) where:
        maps[cam_id] = (mx, my, dist, valid, g_mx, g_my)
        mx, my   : float32 remap maps
        dist     : float32 distance-transform (for feather weights, no recompute per frame)
        valid    : uint8 mask of valid source pixels
        g_mx/g_my: pre-uploaded GpuMat (None if CUDA unavailable)
    """
    active_p = {k: v for k, v in cam_params.items()
                if k.startswith("cam") and v.get("enabled", True)}
    if not active_p:
        return {}, 1920, 1080, 960.0, 540.0
    maps, pano_w, pano_h, cx_p, cy_p = _build_maps_for_params(
        active_p, focal, img_w, img_h)
    return maps, pano_w, pano_h, cx_p, cy_p


def stitch_frame_with_maps(frames_bgr: dict, remap_maps: dict,
                            cam_params: dict, focal: float,
                            img_w: int, img_h: int,
                            pano_w: int, pano_h: int,
                            feather_width: int = 200) -> np.ndarray:
    """
    Stitch one pre-undistorted frame using precomputed remap maps.

    frames_bgr  : {cam_id: bgr_ndarray}  at img_w × img_h
    remap_maps  : output of precompute_remap_maps()
    Returns BGR ndarray (pano_h, pano_w, 3).

    GPU path: each call uploads the source frame to GPU (~0.5 ms at 1080p).
    Maps are re-uploaded fresh per call (thread-safe for ThreadPoolExecutor).
    """
    warped_list: list = []
    dist_list:   list = []

    for cam_id in sorted(remap_maps.keys()):
        if cam_id not in frames_bgr:
            continue
        mx, my, dist, valid, _g_mx, _g_my = remap_maps[cam_id]

        bgr = frames_bgr[cam_id]
        if bgr.shape[1] != img_w or bgr.shape[0] != img_h:
            bgr = cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_AREA)

        p = cam_params.get(f"cam{cam_id}", {})
        k1, k2 = p.get("k1", 0.0), p.get("k2", 0.0)
        if k1 != 0.0 or k2 != 0.0:
            K        = np.array([[focal, 0, img_w / 2.0],
                                  [0, focal, img_h / 2.0],
                                  [0, 0, 1.0]], np.float64)
            dist_coef = np.array([k1, k2, 0.0, 0.0], np.float64)
            bgr       = cv2.undistort(bgr, K, dist_coef)

        # Re-upload maps fresh each call — thread-safe for render pool
        warped = _remap_frame(bgr, mx, my)   # g_mx/g_my = None → upload in fn
        warped_list.append(warped)
        dist_list.append(dist)

    if not warped_list:
        return np.zeros((pano_h, pano_w, 3), np.uint8)

    return feather_blend_preweighted(warped_list, dist_list, feather_width)


# ── Legacy helpers (homography-based pipeline) ────────────────
from .settings import STITCH_ORDER, REF_CAM


def color_match(src: np.ndarray, ref: np.ndarray,
                intensity: float, brightness: float, contrast: float) -> np.ndarray:
    s, r = src.astype(np.float32), ref.astype(np.float32)
    out  = s.copy()
    for c in range(3):
        sm, ss = s[:, :, c].mean(), max(s[:, :, c].std(), 1e-6)
        rm, rs = r[:, :, c].mean(), max(r[:, :, c].std(), 1e-6)
        matched = (s[:, :, c] - sm) * (rs / ss) + rm
        out[:, :, c] = s[:, :, c] * (1 - intensity) + matched * intensity
    return (out * contrast + brightness).clip(0, 255).astype(np.uint8)


def blend_feather(wlist: list, mlist: list) -> np.ndarray:
    h, w = wlist[0].shape[:2]
    acc  = np.zeros((h, w, 3), np.float64)
    wsum = np.zeros((h, w),    np.float64)
    for img, mask in zip(wlist, mlist):
        d     = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
        acc  += img.astype(np.float64) * d[:, :, np.newaxis]
        wsum += d
    wsum = np.where(wsum == 0, 1, wsum)
    return (acc / wsum[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


def blend_hard(wlist: list, mlist: list) -> np.ndarray:
    h, w = wlist[0].shape[:2]
    out  = np.zeros((h, w, 3), np.uint8)
    done = np.zeros((h, w),    bool)
    for img, mask in zip(wlist, mlist):
        px = (mask > 0) & ~done
        out[px] = img[px]
        done |= (mask > 0)
    return out


def stitch_one(frames: dict, H_abs: dict, canvas: dict,
               blend: str, ref_bgr: np.ndarray,
               intensity: float, brightness: float, contrast: float,
               cam_masks: dict | None = None,
               rig_layout: dict | None = None) -> np.ndarray:
    T  = np.array([[1, 0, -canvas["x_min"]], [0, 1, -canvas["y_min"]], [0, 0, 1]], np.float64)
    ow, oh = canvas["out_w"], canvas["out_h"]
    wlist, mlist = [], []
    for cam_id in STITCH_ORDER:
        if cam_id not in frames:
            continue
        bgr = cv2.cvtColor(frames[cam_id], cv2.COLOR_RGB2BGR)
        if cam_id != REF_CAM and intensity > 0:
            bgr = color_match(bgr, ref_bgr, intensity, brightness, contrast)
        h0, w0 = bgr.shape[:2]
        adj    = (rig_layout or {}).get(str(cam_id), {})
        cam_br = adj.get("brightness", 0.0)
        cam_ct = adj.get("contrast",   1.0)
        if cam_br != 0.0 or cam_ct != 1.0:
            bgr = (bgr.astype(np.float32) * cam_ct + cam_br).clip(0, 255).astype(np.uint8)
        stitch_mask = np.ones((h0, w0), np.uint8) * 255
        if cam_masks and cam_id in cam_masks:
            cm = cam_masks[cam_id]
            if cm.shape != (h0, w0):
                cm = cv2.resize(cm, (w0, h0), interpolation=cv2.INTER_NEAREST)
            stitch_mask = np.where(cm > 127, 0, stitch_mask).astype(np.uint8)
        rotation = adj.get("rotation", 0.0)
        scale    = adj.get("scale",    1.0)
        if rotation != 0.0 or abs(scale - 1.0) > 0.001:
            M           = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), rotation, scale)
            bgr         = cv2.warpAffine(bgr,         M, (w0, h0))
            stitch_mask = cv2.warpAffine(stitch_mask, M, (w0, h0))
        Hf = T @ np.array(H_abs[str(cam_id)], dtype=np.float64)
        dx = adj.get("dx", 0)
        dy = adj.get("dy", 0)
        if dx != 0 or dy != 0:
            T_adj = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
            Hf    = T_adj @ Hf
        wlist.append(cv2.warpPerspective(bgr,         Hf, (ow, oh)))
        mlist.append(cv2.warpPerspective(stitch_mask, Hf, (ow, oh)))
    result = blend_feather(wlist, mlist) if blend == "feather" else blend_hard(wlist, mlist)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
