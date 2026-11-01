import cv2
import numpy as np
import json
import copy
from .settings import (
    LENS_FILE, HOMO_FILE, RIG_LAYOUT_FILE, CANVAS_LAYOUT_FILE, CAL_DIR,
    RED_KOMODO_PRESET, STITCH_ORDER, REF_CAM, REF_IDX, ADJ_PAIRS,
    CYL_PARAMS_FILE, PRESETS_DIR, DEFAULT_CYL_PARAMS, CAM_MAPPING_FILE,
    OUTPUT_RES_FILE, COLOR_FILE,
)


# ── 렌즈 ────────────────────────────────────────────────────
def load_lens() -> dict:
    if LENS_FILE.exists():
        try:
            data = json.loads(LENS_FILE.read_text("utf-8"))
            return {f"cam{i}": {**RED_KOMODO_PRESET, **data.get(f"cam{i}", {})}
                    for i in range(1, 6)}
        except Exception:
            pass
    return {f"cam{i}": RED_KOMODO_PRESET.copy() for i in range(1, 6)}


def save_lens(params: dict) -> None:
    LENS_FILE.write_text(json.dumps(params, ensure_ascii=False, indent=2), "utf-8")


def make_K(p: dict, w: int, h: int) -> np.ndarray:
    return np.array([
        [p["fx_ratio"] * w, 0,                p["cx_ratio"] * w],
        [0,                 p["fy_ratio"] * h, p["cy_ratio"] * h],
        [0,                 0,                1],
    ], dtype=np.float64)


def make_D(p: dict) -> np.ndarray:
    return np.array([p["k1"], p["k2"], p["p1"], p["p2"], p["k3"]], dtype=np.float64)


def undistort(rgb: np.ndarray, p: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.undistort(bgr, make_K(p, w, h), make_D(p))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ── 호모그래피 ───────────────────────────────────────────────
def load_homo() -> dict | None:
    if HOMO_FILE.exists():
        try:
            return json.loads(HOMO_FILE.read_text("utf-8"))
        except Exception:
            pass
    return None


def save_homo(d: dict) -> None:
    HOMO_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")


def match_sift(img_a, img_b, mask_a=None, mask_b=None):
    det = cv2.SIFT_create()
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    ga  = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb  = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    kpa, da = det.detectAndCompute(ga, mask_a)
    kpb, db = det.detectAndCompute(gb, mask_b)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None, None
    ms   = bfm.knnMatch(da, db, k=2)
    good = [m for m, n in ms if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, None
    src = np.float32([kpa[m.queryIdx].pt for m in good])
    dst = np.float32([kpb[m.trainIdx].pt for m in good])
    return src, dst


def find_H(src, dst):
    if src is None or len(src) < 10:
        return None
    H, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    return H


def compute_H_abs(H_adj: dict) -> dict:
    """H_adj: {(cam_a, cam_b): ndarray}  →  H_abs: {cam_id: ndarray}"""
    H_abs = {REF_CAM: np.eye(3, dtype=np.float64)}
    for i in range(REF_IDX - 1, -1, -1):
        cam, nxt = STITCH_ORDER[i], STITCH_ORDER[i + 1]
        H_abs[cam] = H_abs[nxt] @ H_adj[(cam, nxt)]
    for i in range(REF_IDX + 1, 5):
        cam, prev = STITCH_ORDER[i], STITCH_ORDER[i - 1]
        H_abs[cam] = H_abs[prev] @ np.linalg.inv(H_adj[(prev, cam)])
    return H_abs


def compute_canvas(H_abs: dict, w: int, h: int) -> dict:
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners = np.concatenate([cv2.perspectiveTransform(pts, H) for H in H_abs.values()])
    x_min = int(np.floor(corners[:, :, 0].min()))
    y_min = int(np.floor(corners[:, :, 1].min()))
    x_max = int(np.ceil (corners[:, :, 0].max()))
    y_max = int(np.ceil (corners[:, :, 1].max()))
    return dict(x_min=x_min, y_min=y_min, out_w=x_max - x_min, out_h=y_max - y_min)


# ── 리그 레이아웃 ─────────────────────────────────────────────
_DEFAULT_CAM_ADJ = {"dx": 0, "dy": 0, "rotation": 0.0, "brightness": 0.0, "contrast": 1.0}


def load_rig_layout() -> dict | None:
    if RIG_LAYOUT_FILE.exists():
        try:
            return json.loads(RIG_LAYOUT_FILE.read_text("utf-8"))
        except Exception:
            pass
    return None


def save_rig_layout(layout: dict) -> None:
    RIG_LAYOUT_FILE.write_text(json.dumps(layout, ensure_ascii=False, indent=2), "utf-8")


# ── 캔버스 레이아웃 ───────────────────────────────────────────
def load_canvas_layout() -> dict | None:
    if CANVAS_LAYOUT_FILE.exists():
        try:
            return json.loads(CANVAS_LAYOUT_FILE.read_text("utf-8"))
        except Exception:
            pass
    return None


def save_canvas_layout(layout: dict) -> None:
    CANVAS_LAYOUT_FILE.write_text(json.dumps(layout, ensure_ascii=False, indent=2), "utf-8")


# ── 마스크 ───────────────────────────────────────────────────
def load_cam_masks() -> dict:
    masks = {}
    for cam_id in range(1, 6):
        p = CAL_DIR / f"mask_CAM{cam_id}.png"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks[cam_id] = m
    return masks


# ── 원통 투영 파라미터 ─────────────────────────────────────────
def load_cyl_params() -> dict:
    if CYL_PARAMS_FILE.exists():
        try:
            data = json.loads(CYL_PARAMS_FILE.read_text("utf-8"))
            # Merge with defaults so new keys are always present
            out = copy.deepcopy(DEFAULT_CYL_PARAMS)
            out["focal"] = data.get("focal", DEFAULT_CYL_PARAMS["focal"])
            for key in [f"cam{i}" for i in range(1, 6)]:
                if key in data:
                    out[key].update(data[key])
            return out
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_CYL_PARAMS)


def save_cyl_params(params: dict) -> None:
    CYL_PARAMS_FILE.write_text(json.dumps(params, ensure_ascii=False, indent=2), "utf-8")


# ── 프리셋 ────────────────────────────────────────────────────
def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def load_preset(name: str) -> dict | None:
    p = PRESETS_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return None


def save_preset(name: str, params: dict) -> None:
    safe = "".join(c if c.isalnum() or c in "_- " else "_" for c in name).strip()
    if not safe:
        safe = "preset"
    (PRESETS_DIR / f"{safe}.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), "utf-8"
    )


def delete_preset(name: str) -> None:
    p = PRESETS_DIR / f"{name}.json"
    if p.exists():
        p.unlink()


# ── 카메라 소스 매핑 ──────────────────────────────────────────
def load_cam_mapping() -> dict[int, str | None]:
    """Returns {1: path_str|None, ..., 5: path_str|None}"""
    result: dict[int, str | None] = {i: None for i in range(1, 6)}
    if CAM_MAPPING_FILE.exists():
        try:
            data = json.loads(CAM_MAPPING_FILE.read_text("utf-8"))
            for k, v in data.items():
                try:
                    cam_id = int(k)
                    if 1 <= cam_id <= 5:
                        result[cam_id] = v if v else None
                except ValueError:
                    pass
        except Exception:
            pass
    return result


def save_cam_mapping(mapping: dict[int, str | None]) -> None:
    """mapping: {cam_id: path_str|None}"""
    data = {str(k): str(v) if v else None for k, v in mapping.items()}
    CAM_MAPPING_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ── 출력 해상도 ───────────────────────────────────────────────
_DEFAULT_OUTPUT_RES = {"width": 7680, "height": 2160}


def load_output_resolution() -> dict:
    if OUTPUT_RES_FILE.exists():
        try:
            data = json.loads(OUTPUT_RES_FILE.read_text("utf-8"))
            w, h = int(data.get("width", 0)), int(data.get("height", 0))
            if w > 0 and h > 0:
                return {"width": w, "height": h}
        except Exception:
            pass
    return _DEFAULT_OUTPUT_RES.copy()


def save_output_resolution(width: int, height: int) -> None:
    OUTPUT_RES_FILE.write_text(
        json.dumps({"width": width, "height": height}, indent=2), "utf-8"
    )


# ── 컬러 파라미터 ──────────────────────────────────────────────
def load_color_params() -> dict:
    from .color import default_color_params
    if COLOR_FILE.exists():
        try:
            data = json.loads(COLOR_FILE.read_text("utf-8"))
            defaults = default_color_params()
            # Merge: keep defaults for any missing keys
            result = {"per_cam": {}, "output": {**defaults["output"]}}
            result["output"].update(data.get("output", {}))
            saved_pc = data.get("per_cam", {})
            for cam_id in range(1, 6):
                key = str(cam_id)
                cam_defaults = copy.deepcopy(defaults["per_cam"][cam_id])
                if key in saved_pc:
                    cam_defaults.update(saved_pc[key])
                result["per_cam"][cam_id] = cam_defaults
            return result
        except Exception:
            pass
    from .color import default_color_params
    return default_color_params()


def save_color_params(params: dict) -> None:
    # Convert int keys to str for JSON serialisation
    serialisable = {
        "per_cam": {str(k): v for k, v in params.get("per_cam", {}).items()},
        "output":  params.get("output", {}),
    }
    COLOR_FILE.write_text(
        json.dumps(serialisable, ensure_ascii=False, indent=2), "utf-8"
    )
