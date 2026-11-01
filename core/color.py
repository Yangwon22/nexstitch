"""
NexStitch — Color Pipeline
Per-camera and output color math.
"""
import copy
import numpy as np

# ── Default parameter structures ──────────────────────────────
_DEFAULT_PER_CAM: dict = {
    "exposure":    0.0,
    "temperature": 6500,
    "tint":        0,
    "lift":        [0.0, 0.0, 0.0],
    "gamma":       [1.0, 1.0, 1.0],
    "gain":        [1.0, 1.0, 1.0],
}

_DEFAULT_OUTPUT: dict = {
    "exposure":    0.0,
    "contrast":    1.0,
    "saturation":  1.0,
    "temperature": 6500,
    "tint":        0,
    "lift":        [0.0, 0.0, 0.0],
    "gamma":       [1.0, 1.0, 1.0],
    "gain":        [1.0, 1.0, 1.0],
    "curves": {
        "r": [[0, 0], [255, 255]],
        "g": [[0, 0], [255, 255]],
        "b": [[0, 0], [255, 255]],
    },
}


def default_per_cam() -> dict:
    return copy.deepcopy(_DEFAULT_PER_CAM)


def default_output() -> dict:
    return copy.deepcopy(_DEFAULT_OUTPUT)


def default_color_params() -> dict:
    return {
        "per_cam": {i: default_per_cam() for i in range(1, 6)},
        "output":  default_output(),
    }


# ── Color math helpers ────────────────────────────────────────
def _temp_tint_mult(temp_k: float, tint: float) -> np.ndarray:
    """Returns [r_mult, g_mult, b_mult] for temperature/tint adjustment.

    temp_k relative to 6500K neutral:
      below 6500K = warm  → boost R, reduce B
      above 6500K = cool  → reduce R, boost B
    tint: positive = magenta (reduce G), negative = green (boost G)
    """
    t = (temp_k - 6500.0) / 4000.0        # ≈ -1 (warm) to +1 (cool)
    r = 1.0 - 0.40 * t
    g = 1.0 - 0.04 * abs(t)
    b = 1.0 + 0.55 * t
    tn = tint / 100.0
    g *= max(0.01, 1.0 - 0.30 * tn)       # magenta tint reduces green
    r *= 1.0 + 0.08 * tn
    b *= 1.0 + 0.08 * tn
    return np.array([r, g, b], dtype=np.float32)


def _apply_lgg(img: np.ndarray, lift: list, gamma: list, gain: list) -> None:
    """Apply Lift/Gamma/Gain per RGB channel on float32 [0,1] image (in-place)."""
    for ch in range(3):
        if lift[ch] == 0.0 and gamma[ch] == 1.0 and gain[ch] == 1.0:
            continue
        c = img[:, :, ch]
        c = np.clip(gain[ch] * c + lift[ch], 0.0, 1.0)
        if gamma[ch] != 1.0 and gamma[ch] > 0.0:
            c = np.power(c, 1.0 / gamma[ch])
        img[:, :, ch] = c


def _build_lut(control_points: list) -> np.ndarray:
    """Build a 256-entry uint8 LUT from [[x, y], ...] control points."""
    pts = sorted(control_points, key=lambda p: p[0])
    xs  = np.array([p[0] for p in pts], dtype=np.float64)
    ys  = np.array([p[1] for p in pts], dtype=np.float64)
    lut = np.interp(np.arange(256, dtype=np.float64), xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


_IDENTITY_CURVES = {"r": [[0, 0], [255, 255]],
                    "g": [[0, 0], [255, 255]],
                    "b": [[0, 0], [255, 255]]}


# ── Public color application functions ────────────────────────
def apply_per_cam_color(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Apply per-camera color correction.

    Input / output: RGB uint8 H×W×3 ndarray.
    """
    img = rgb.astype(np.float32) / 255.0

    ev = float(params.get("exposure", 0.0))
    if ev:
        img *= 2.0 ** ev

    temp = float(params.get("temperature", 6500))
    tint = float(params.get("tint", 0))
    if temp != 6500 or tint:
        img *= _temp_tint_mult(temp, tint)

    _apply_lgg(
        img,
        params.get("lift",  [0.0, 0.0, 0.0]),
        params.get("gamma", [1.0, 1.0, 1.0]),
        params.get("gain",  [1.0, 1.0, 1.0]),
    )

    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_output_color(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Apply output (post-stitch global) color correction.

    Input / output: RGB uint8 H×W×3 ndarray.
    """
    img = rgb.astype(np.float32) / 255.0

    ev = float(params.get("exposure", 0.0))
    if ev:
        img *= 2.0 ** ev

    temp = float(params.get("temperature", 6500))
    tint = float(params.get("tint", 0))
    if temp != 6500 or tint:
        img *= _temp_tint_mult(temp, tint)

    _apply_lgg(
        img,
        params.get("lift",  [0.0, 0.0, 0.0]),
        params.get("gamma", [1.0, 1.0, 1.0]),
        params.get("gain",  [1.0, 1.0, 1.0]),
    )

    contrast = float(params.get("contrast", 1.0))
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5

    saturation = float(params.get("saturation", 1.0))
    if saturation != 1.0:
        luma = (0.299 * img[:, :, 0] +
                0.587 * img[:, :, 1] +
                0.114 * img[:, :, 2])[:, :, np.newaxis]
        img = luma + saturation * (img - luma)

    img = np.clip(img, 0.0, 1.0)

    curves = params.get("curves", _IDENTITY_CURVES)
    for ch, key in enumerate(("r", "g", "b")):
        pts = curves.get(key, [[0, 0], [255, 255]])
        if pts != [[0, 0], [255, 255]]:
            lut = _build_lut(pts)
            idx = (img[:, :, ch] * 255.0).clip(0, 255).astype(np.uint8)
            img[:, :, ch] = lut[idx].astype(np.float32) / 255.0

    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def match_colors_to_ref(frames_rgb: dict, ref_cam_id: int = 3) -> dict:
    """Auto-match per-camera color to reference camera by equalising RGB means.

    Returns {cam_id: per_cam_params} for all non-reference cameras.
    Adjusts only the Gain channels; all other params are reset to defaults.
    """
    ref = frames_rgb.get(ref_cam_id)
    if ref is None:
        return {}

    ref_mean = (np.mean(ref.reshape(-1, 3).astype(np.float64), axis=0) / 255.0
                + 1e-6)

    result: dict = {}
    for cam_id, frame in frames_rgb.items():
        if cam_id == ref_cam_id:
            continue
        cam_mean = (np.mean(frame.reshape(-1, 3).astype(np.float64), axis=0) / 255.0
                    + 1e-6)
        gain = np.clip(ref_mean / cam_mean, 0.1, 3.0).tolist()
        p = default_per_cam()
        p["gain"] = gain
        result[cam_id] = p

    return result
