import streamlit as st
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

Image.MAX_IMAGE_PIXELS = None   # PIL 픽셀 한계 해제 (대형 파노라마 대응)

PREVIEW_MAX_W = 1920            # 미리보기 최대 폭 (px)
CANVAS_W      = 900             # 마스크 캔버스 고정 폭 (px)

def preview(img_rgb: np.ndarray) -> np.ndarray:
    """st.image() 표시 전 1920px 이하로 리사이즈. 원본은 변경하지 않음."""
    h, w = img_rgb.shape[:2]
    if w <= PREVIEW_MAX_W:
        return img_rgb
    scale = PREVIEW_MAX_W / w
    return cv2.resize(img_rgb, (PREVIEW_MAX_W, int(h * scale)), interpolation=cv2.INTER_AREA)

# ── 경로 / 상수 ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CAL_DIR    = BASE_DIR / "calibration"
LENS_FILE  = CAL_DIR / "lens.json"
HOMO_FILE  = CAL_DIR / "homography.json"

for d in [INPUT_DIR, OUTPUT_DIR, CAL_DIR]:
    d.mkdir(exist_ok=True)

CAM_ROLES    = {1: "정면", 2: "좌1", 3: "좌2", 4: "우1", 5: "우2"}
STITCH_ORDER = [3, 2, 1, 4, 5]   # 좌→우 (1-based)
REF_CAM      = 1
REF_IDX      = STITCH_ORDER.index(REF_CAM)   # = 2
ADJ_PAIRS    = [(STITCH_ORDER[i], STITCH_ORDER[i + 1]) for i in range(4)]
# [(3,2), (2,1), (1,4), (4,5)]

# RED Komodo 21mm / Super35 추정 프리셋
RED_KOMODO_PRESET = dict(
    k1=-0.10, k2=0.04, k3=0.00,
    p1=1e-4,  p2=-1e-4,
    fx_ratio=0.780, fy_ratio=0.780,
    cx_ratio=0.500, cy_ratio=0.500,
)

# ── 파일 I/O ──────────────────────────────────────────────────
def load_lens() -> dict:
    if LENS_FILE.exists():
        try:
            data = json.loads(LENS_FILE.read_text("utf-8"))
            return {f"cam{i}": {**RED_KOMODO_PRESET, **data.get(f"cam{i}", {})}
                    for i in range(1, 6)}
        except Exception:
            pass
    return {f"cam{i}": RED_KOMODO_PRESET.copy() for i in range(1, 6)}

def save_lens(p: dict) -> None:
    LENS_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=2), "utf-8")

def load_homo() -> dict | None:
    if HOMO_FILE.exists():
        try:
            return json.loads(HOMO_FILE.read_text("utf-8"))
        except Exception:
            pass
    return None

def save_homo(d: dict) -> None:
    HOMO_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")

def load_cam_masks() -> dict:
    """저장된 카메라 마스크 로드. {cam_id: grayscale ndarray (흰색=제거)}"""
    masks = {}
    for cam_id in range(1, 6):
        p = CAL_DIR / f"mask_CAM{cam_id}.png"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks[cam_id] = m
    return masks

def mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """마스크 영역을 빨간색 반투명으로 오버레이한 RGB 이미지 반환"""
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame_rgb.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[:, :, 0] = 255.0
    alpha = (mask_gray > 127)[:, :, np.newaxis]
    out = np.where(alpha, out * 0.4 + red * 0.6, out)
    return out.clip(0, 255).astype(np.uint8)

# ── 영상 유틸 ─────────────────────────────────────────────────
def get_mov_files() -> list[Path]:
    seen, unique = set(), []
    for f in sorted(INPUT_DIR.glob("*.mov")) + sorted(INPUT_DIR.glob("*.MOV")):
        if f.name.lower() not in seen:
            seen.add(f.name.lower())
            unique.append(f)
    return unique

@st.cache_data(show_spinner=False)
def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = dict(
        total=max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        fps=cap.get(cv2.CAP_PROP_FPS) or 24.0,
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info

@st.cache_data(show_spinner=False)
def grab_frame(path: str, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

# ── 렌즈 보정 ─────────────────────────────────────────────────
def make_K(p: dict, w: int, h: int) -> np.ndarray:
    return np.array([
        [p["fx_ratio"] * w, 0,               p["cx_ratio"] * w],
        [0,                 p["fy_ratio"] * h, p["cy_ratio"] * h],
        [0,                 0,               1],
    ], dtype=np.float64)

def make_D(p: dict) -> np.ndarray:
    return np.array([p["k1"], p["k2"], p["p1"], p["p2"], p["k3"]], dtype=np.float64)

def undistort(rgb: np.ndarray, p: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.undistort(bgr, make_K(p, w, h), make_D(p))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ── 피처 매칭 / 호모그래피 ────────────────────────────────────
def match_sift(img_a: np.ndarray, img_b: np.ndarray,
               mask_a=None, mask_b=None) -> tuple[np.ndarray | None, np.ndarray | None]:
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

def find_H(src: np.ndarray | None, dst: np.ndarray | None) -> np.ndarray | None:
    if src is None or len(src) < 10:
        return None
    H, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    return H

def compute_canvas(H_abs: dict, w: int, h: int) -> dict:
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    corners = np.concatenate([cv2.perspectiveTransform(pts, H) for H in H_abs.values()])
    x_min = int(np.floor(corners[:,:,0].min()))
    y_min = int(np.floor(corners[:,:,1].min()))
    x_max = int(np.ceil (corners[:,:,0].max()))
    y_max = int(np.ceil (corners[:,:,1].max()))
    return dict(x_min=x_min, y_min=y_min, out_w=x_max-x_min, out_h=y_max-y_min)

# ── 컬러 매칭 ─────────────────────────────────────────────────
def color_match(src: np.ndarray, ref: np.ndarray,
                intensity: float, brightness: float, contrast: float) -> np.ndarray:
    s = src.astype(np.float32)
    r = ref.astype(np.float32)
    out = s.copy()
    for c in range(3):
        sm, ss = s[:,:,c].mean(), max(s[:,:,c].std(), 1e-6)
        rm, rs = r[:,:,c].mean(), max(r[:,:,c].std(), 1e-6)
        matched = (s[:,:,c] - sm) * (rs / ss) + rm
        out[:,:,c] = s[:,:,c] * (1 - intensity) + matched * intensity
    out = out * contrast + brightness
    return out.clip(0, 255).astype(np.uint8)

# ── 블렌딩 ───────────────────────────────────────────────────
def blend_feather(wlist: list, mlist: list) -> np.ndarray:
    h, w = wlist[0].shape[:2]
    acc  = np.zeros((h, w, 3), np.float64)
    wsum = np.zeros((h, w),    np.float64)
    for img, mask in zip(wlist, mlist):
        d     = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
        acc  += img.astype(np.float64) * d[:,:,np.newaxis]
        wsum += d
    wsum = np.where(wsum == 0, 1, wsum)
    return (acc / wsum[:,:,np.newaxis]).clip(0, 255).astype(np.uint8)

def blend_hard(wlist: list, mlist: list) -> np.ndarray:
    h, w = wlist[0].shape[:2]
    out  = np.zeros((h, w, 3), np.uint8)
    done = np.zeros((h, w),    bool)
    for img, mask in zip(wlist, mlist):
        px = (mask > 0) & ~done
        out[px] = img[px]
        done |= (mask > 0)
    return out

# ── 프레임 스티치 ─────────────────────────────────────────────
def stitch_one(frames: dict, H_abs: dict, canvas: dict,
               blend: str, ref_bgr: np.ndarray,
               intensity: float, brightness: float, contrast: float,
               cam_masks: dict | None = None) -> np.ndarray:
    T  = np.array([[1,0,-canvas["x_min"]],[0,1,-canvas["y_min"]],[0,0,1]], np.float64)
    ow, oh = canvas["out_w"], canvas["out_h"]
    wlist, mlist = [], []
    for cam_id in STITCH_ORDER:
        if cam_id not in frames:
            continue
        bgr = cv2.cvtColor(frames[cam_id], cv2.COLOR_RGB2BGR)
        if cam_id != REF_CAM and intensity > 0:
            bgr = color_match(bgr, ref_bgr, intensity, brightness, contrast)
        h0, w0 = bgr.shape[:2]

        # 스티치 마스크 생성 (기본: 전체 유효)
        stitch_mask = np.ones((h0, w0), np.uint8) * 255
        if cam_masks and cam_id in cam_masks:
            cm = cam_masks[cam_id]
            if cm.shape != (h0, w0):
                cm = cv2.resize(cm, (w0, h0), interpolation=cv2.INTER_NEAREST)
            # 마스크 흰색 영역(제거) → stitch_mask 0 으로
            stitch_mask = np.where(cm > 127, 0, stitch_mask).astype(np.uint8)

        Hf = T @ np.array(H_abs[str(cam_id)], dtype=np.float64)
        wlist.append(cv2.warpPerspective(bgr, Hf, (ow, oh)))
        mlist.append(cv2.warpPerspective(stitch_mask, Hf, (ow, oh)))
    result = blend_feather(wlist, mlist) if blend == "feather" else blend_hard(wlist, mlist)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="NexStitch", layout="wide")

c1, c2 = st.columns([4, 1])
with c1:
    st.title("NexStitch")
    st.caption("RED Komodo ×5 | 21mm | Mistika Workflow")
with c2:
    mov_files = get_mov_files()
    n = len(mov_files)
    st.metric("소스 파일", f"{n} / 5")
    if n == 5:
        st.success("준비 완료", icon="✅")
    else:
        st.warning(f"{n}개 감지됨")

with st.expander("소스 파일 목록", expanded=(n != 5)):
    if not mov_files:
        st.info("input/ 폴더에 .mov 파일을 넣어주세요.")
    else:
        for i, f in enumerate(mov_files):
            role = CAM_ROLES.get(i + 1, "?")
            st.caption(f"CAM{i+1} ({role}) — {f.name}  |  {f.stat().st_size/1e6:.1f} MB")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔭 렌즈 캘리브레이션",
    "🎨 마스크 편집",
    "📐 리그 캘리브레이션",
    "🎬 스티칭 렌더",
])

# ════════════════════════════════════════════════════════════
# 탭 1: 렌즈 캘리브레이션
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("렌즈 캘리브레이션")
    st.caption("카메라별 왜곡 파라미터를 수동으로 입력합니다. 체커보드 없이 파라미터 직접 조정 방식입니다.")

    lens_params = load_lens()

    btn_col1, btn_col2 = st.columns([2, 2])
    with btn_col1:
        if st.button("RED Komodo 21mm 프리셋 전체 적용"):
            lens_params = {f"cam{i}": RED_KOMODO_PRESET.copy() for i in range(1, 6)}
            save_lens(lens_params)
            st.rerun()
    with btn_col2:
        if st.button("렌즈 파라미터 저장", type="primary"):
            save_lens(lens_params)
            st.success(f"저장 완료 → {LENS_FILE.name}")

    st.divider()

    for cam_id in range(1, 6):
        key  = f"cam{cam_id}"
        role = CAM_ROLES[cam_id]
        p    = lens_params[key]

        with st.expander(f"CAM {cam_id} — {role}", expanded=(cam_id == 1)):
            # 프리셋 개별 적용
            if st.button(f"프리셋 적용", key=f"preset_{cam_id}"):
                p = RED_KOMODO_PRESET.copy()
                lens_params[key] = p

            st.markdown("**방사형 왜곡 (Radial)**")
            rc = st.columns(3)
            p["k1"] = rc[0].number_input("k1", value=float(p["k1"]), format="%.5f",
                                          min_value=-2.0, max_value=2.0, step=0.001,
                                          key=f"k1_{cam_id}",
                                          help="음수: 배럴 왜곡 / 양수: 핀쿠션 왜곡")
            p["k2"] = rc[1].number_input("k2", value=float(p["k2"]), format="%.5f",
                                          min_value=-2.0, max_value=2.0, step=0.001,
                                          key=f"k2_{cam_id}")
            p["k3"] = rc[2].number_input("k3", value=float(p["k3"]), format="%.5f",
                                          min_value=-2.0, max_value=2.0, step=0.001,
                                          key=f"k3_{cam_id}")

            st.markdown("**접선형 왜곡 (Tangential)**")
            tc = st.columns(2)
            p["p1"] = tc[0].number_input("p1", value=float(p["p1"]), format="%.6f",
                                          min_value=-0.5, max_value=0.5, step=0.0001,
                                          key=f"p1_{cam_id}")
            p["p2"] = tc[1].number_input("p2", value=float(p["p2"]), format="%.6f",
                                          min_value=-0.5, max_value=0.5, step=0.0001,
                                          key=f"p2_{cam_id}")

            st.markdown("**카메라 내부 파라미터**")
            fc = st.columns(4)
            p["fx_ratio"] = fc[0].number_input("fx (폭 비율)", value=float(p["fx_ratio"]),
                                                format="%.4f", min_value=0.1, max_value=3.0,
                                                step=0.001, key=f"fx_{cam_id}",
                                                help="fx = fx_ratio × 이미지 폭")
            p["fy_ratio"] = fc[1].number_input("fy (높이 비율)", value=float(p["fy_ratio"]),
                                                format="%.4f", min_value=0.1, max_value=3.0,
                                                step=0.001, key=f"fy_{cam_id}")
            p["cx_ratio"] = fc[2].number_input("cx (주점 x)", value=float(p["cx_ratio"]),
                                                format="%.4f", min_value=0.0, max_value=1.0,
                                                step=0.001, key=f"cx_{cam_id}",
                                                help="cx = cx_ratio × 이미지 폭 (보통 0.5)")
            p["cy_ratio"] = fc[3].number_input("cy (주점 y)", value=float(p["cy_ratio"]),
                                                format="%.4f", min_value=0.0, max_value=1.0,
                                                step=0.001, key=f"cy_{cam_id}")

            lens_params[key] = p

            # 보정 전/후 미리보기
            if n >= cam_id:
                file_path = str(mov_files[cam_id - 1])
                raw = grab_frame(file_path, 0)
                if raw is not None:
                    corrected = undistort(raw, p)
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.caption("원본")
                        st.image(preview(raw), width="stretch")
                    with pc2:
                        st.caption("왜곡 보정 후")
                        st.image(preview(corrected), width="stretch")
            else:
                st.info(f"CAM{cam_id} 소스 파일 없음 — input/ 폴더에 파일을 추가하세요.")

# ════════════════════════════════════════════════════════════
# 탭 2: 마스크 편집
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("마스크 편집")
    st.caption("카메라별 제거할 영역을 직접 그립니다.  흰색 = 제거 영역 / 검정 = 유지 영역")

    if n == 0:
        st.warning("소스 파일이 필요합니다. input/ 폴더에 .mov 파일을 추가하세요.")
    else:
        # 카메라 선택
        cam_opts = [f"CAM{i}  ({CAM_ROLES[i]})" for i in range(1, min(n, 5) + 1)]
        cam_sel_str = st.selectbox("카메라 선택", cam_opts, key="mask_cam_sel")
        mask_cam = int(cam_sel_str.split()[0].replace("CAM", ""))

        fpath_m = str(mov_files[mask_cam - 1])
        info_m  = get_video_info(fpath_m)

        # 프레임 슬라이더
        fidx_m = st.slider(
            "참고 프레임", 0, info_m["total"] - 1, 0,
            key=f"mask_fidx_{mask_cam}"
        )

        raw_m = grab_frame(fpath_m, fidx_m)
        if raw_m is None:
            st.error("프레임 추출 실패")
        else:
            lens_m  = load_lens()
            frame_m = undistort(raw_m, lens_m[f"cam{mask_cam}"])
            h_f, w_f = frame_m.shape[:2]

            # 캔버스 표시 크기
            canvas_h_m = int(h_f * CANVAS_W / w_f)
            frame_disp = cv2.resize(frame_m, (CANVAS_W, canvas_h_m), interpolation=cv2.INTER_AREA)

            # 드로잉 설정
            mc1, mc2 = st.columns(2)
            draw_mode   = mc1.selectbox("드로잉 모드", ["freedraw", "polygon"],
                                        key="mask_mode",
                                        help="freedraw: 자유 브러시 / polygon: 다각형")
            brush_size  = mc2.slider("브러시 크기", 5, 120, 30, key="mask_brush")

            # 기존 마스크가 있으면 캔버스 배경으로 로드
            mask_path_m = CAL_DIR / f"mask_CAM{mask_cam}.png"
            bg_pil = None
            if mask_path_m.exists():
                existing_m = cv2.imread(str(mask_path_m), cv2.IMREAD_GRAYSCALE)
                if existing_m is not None:
                    existing_disp = cv2.resize(existing_m, (CANVAS_W, canvas_h_m),
                                               interpolation=cv2.INTER_NEAREST)
                    bg_pil = Image.fromarray(existing_disp)

            # 참고용 프레임 미리보기
            st.caption("참고 프레임 (캔버스에서 그릴 영역 확인용)")
            st.image(preview(frame_disp), width="stretch")

            st.caption("마스크 캔버스 — 흰색으로 제거할 영역을 그리세요")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.85)",
                stroke_width=brush_size,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=bg_pil,
                update_streamlit=True,
                height=canvas_h_m,
                width=CANVAS_W,
                drawing_mode=draw_mode,
                key=f"mask_canvas_{mask_cam}",
            )

            # 마스크 추출 및 오버레이 미리보기
            if canvas_result.image_data is not None:
                drawn_rgb  = canvas_result.image_data[:, :, :3]
                mask_disp  = np.where(drawn_rgb.mean(axis=2) > 127, 255, 0).astype(np.uint8)

                if mask_disp.max() > 0:
                    overlay_img = mask_overlay(frame_disp, mask_disp)
                    st.caption("마스크 오버레이 미리보기 (빨간색 = 제거 영역)")
                    st.image(preview(overlay_img), width="stretch")

                col_save_m, col_reset_m = st.columns(2)
                with col_save_m:
                    if st.button(f"마스크 저장 (CAM{mask_cam})", type="primary",
                                 key=f"mask_save_{mask_cam}"):
                        # 원본 해상도로 업스케일하여 저장
                        full_mask = cv2.resize(mask_disp, (w_f, h_f),
                                               interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(str(mask_path_m), full_mask)
                        st.success(f"저장 완료 → {mask_path_m.name}  ({w_f}×{h_f}px)")
                with col_reset_m:
                    if st.button(f"마스크 초기화 (CAM{mask_cam})",
                                 key=f"mask_reset_{mask_cam}"):
                        if mask_path_m.exists():
                            mask_path_m.unlink()
                        st.success("마스크 초기화 완료")
                        st.rerun()

        st.divider()
        st.subheader("저장된 마스크 현황")
        status_cols = st.columns(5)
        for ci in range(1, 6):
            mp = CAL_DIR / f"mask_CAM{ci}.png"
            with status_cols[ci - 1]:
                if mp.exists():
                    m_stat = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    pct = (m_stat > 127).sum() / m_stat.size * 100 if m_stat is not None else 0
                    st.success(f"CAM{ci}", icon="✅")
                    st.caption(f"마스크 {pct:.1f}%")
                else:
                    st.info(f"CAM{ci}")
                    st.caption("없음")

# ════════════════════════════════════════════════════════════
# 탭 3: 리그 캘리브레이션
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("리그 캘리브레이션")

    # 배치 다이어그램
    st.caption("스티칭 순서 (왼쪽 → 오른쪽)")
    diag_cols = st.columns(5)
    for i, cam_id in enumerate(STITCH_ORDER):
        is_ref = cam_id == REF_CAM
        diag_cols[i].metric(
            f"{'★ ' if is_ref else ''}CAM{cam_id}",
            CAM_ROLES[cam_id],
            "기준" if is_ref else f"← →"[int(i > REF_IDX)] if i != REF_IDX else None,
        )

    st.divider()

    # 오버랩 슬라이더
    st.subheader("인접 카메라 오버랩 비율")
    ov_cols = st.columns(4)
    overlap = {}
    for i, (a, b) in enumerate(ADJ_PAIRS):
        overlap[(a, b)] = ov_cols[i].slider(
            f"CAM{a}↔CAM{b}", 5, 60, 30, 5,
            key=f"ov_{a}_{b}",
            help="두 카메라 간 예상 겹침 영역 (%)"
        )

    st.divider()

    if n < 5:
        st.warning("소스 파일이 5개 필요합니다.")
    else:
        st.subheader("캘리브레이션 프레임 선택")
        lens_data = load_lens()

        # 프레임 슬라이더 + 미리보기
        frame_sel = {}
        frame_data = {}
        fr_cols = st.columns(5)

        for i, cam_id in enumerate(range(1, 6)):
            fpath = str(mov_files[cam_id - 1])
            info  = get_video_info(fpath)
            role  = CAM_ROLES[cam_id]

            with fr_cols[i]:
                st.caption(f"CAM{cam_id} {role}")
                fidx = st.slider(
                    "프레임", 0, info["total"] - 1,
                    value=(info["total"] - 1) // 2,
                    key=f"fsel_{cam_id}"
                )
                frame_sel[cam_id] = fidx
                raw = grab_frame(fpath, fidx)
                if raw is not None:
                    corr = undistort(raw, lens_data[f"cam{cam_id}"])
                    frame_data[cam_id] = corr
                    st.image(preview(corr), width="stretch")
                    st.caption(f"{info['w']}×{info['h']}  |  {info['total']}프레임")
                else:
                    st.error("프레임 추출 실패")

        st.divider()

        # 호모그래피 계산 버튼
        homo_exists = HOMO_FILE.exists()
        comp_col, stat_col = st.columns([1, 3])
        with comp_col:
            compute_btn = st.button(
                "호모그래피 계산 및 저장",
                type="primary",
                disabled=(len(frame_data) < 5)
            )
        with stat_col:
            if homo_exists:
                st.success(f"저장된 캘리브레이션 있음: {HOMO_FILE.name}", icon="✅")

        if compute_btn and len(frame_data) == 5:
            errors = []
            H_adj  = {}

            progress = st.progress(0, text="피처 매칭 중...")

            for step, (cam_a, cam_b) in enumerate(ADJ_PAIRS):
                progress.progress((step) / 4, text=f"CAM{cam_a} ↔ CAM{cam_b} 매칭 중...")
                img_a = frame_data[cam_a]
                img_b = frame_data[cam_b]
                ha, wa = img_a.shape[:2]
                hb, wb = img_b.shape[:2]
                ov = overlap[(cam_a, cam_b)] / 100.0

                # 오버랩 마스크: cam_a 오른쪽, cam_b 왼쪽
                mask_a = np.zeros((ha, wa), np.uint8)
                mask_b = np.zeros((hb, wb), np.uint8)
                mask_a[:, int(wa * (1 - ov)):] = 255
                mask_b[:, :int(wb * ov)]        = 255

                src, dst = match_sift(img_a, img_b, mask_a, mask_b)
                n_masked = len(src) if src is not None else 0

                if n_masked < 10:
                    # 마스크 없이 재시도
                    src, dst = match_sift(img_a, img_b)

                H = find_H(src, dst)
                n_pts = len(src) if src is not None else 0

                if H is None:
                    errors.append(f"CAM{cam_a}↔CAM{cam_b}: 매칭 실패 ({n_pts}개 매칭점). "
                                  f"오버랩 비율을 조정하거나 다른 프레임을 선택하세요.")
                else:
                    H_adj[(cam_a, cam_b)] = H
                    st.caption(f"CAM{cam_a}↔CAM{cam_b}: {n_pts}개 매칭점 ({n_masked}개 마스크 영역)")

            progress.progress(1.0, text="호모그래피 계산 중...")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # 절대 호모그래피 계산 (모든 카메라 → CAM1 공간)
                H_abs = {REF_CAM: np.eye(3, dtype=np.float64)}

                # REF 왼쪽 (i < REF_IDX), 오른쪽에서 왼쪽 방향으로 체인
                for i in range(REF_IDX - 1, -1, -1):
                    cam  = STITCH_ORDER[i]
                    nxt  = STITCH_ORDER[i + 1]
                    # H_adj[(cam, nxt)] maps cam→nxt
                    # H_abs[cam] = H_abs[nxt] @ H_adj[(cam, nxt)]
                    H_abs[cam] = H_abs[nxt] @ H_adj[(cam, nxt)]

                # REF 오른쪽 (i > REF_IDX), 왼쪽에서 오른쪽 방향으로 체인
                for i in range(REF_IDX + 1, 5):
                    cam  = STITCH_ORDER[i]
                    prev = STITCH_ORDER[i - 1]
                    # H_adj[(prev, cam)] maps prev→cam
                    # inv 를 취하면 cam→prev
                    # H_abs[cam] = H_abs[prev] @ inv(H_adj[(prev, cam)])
                    H_abs[cam] = H_abs[prev] @ np.linalg.inv(H_adj[(prev, cam)])

                # 캔버스 계산 (CAM1 해상도 기준)
                ref_info = get_video_info(str(mov_files[0]))
                canvas = compute_canvas(H_abs, ref_info["w"], ref_info["h"])

                if canvas["out_w"] > 30000 or canvas["out_h"] > 15000:
                    st.error(f"캔버스 크기 초과 ({canvas['out_w']}×{canvas['out_h']}px). "
                             f"호모그래피 추정 오류일 수 있습니다. 카메라 순서/프레임을 확인하세요.")
                else:
                    # JSON 저장 (numpy → list)
                    save_homo({
                        "H_abs":  {str(k): v.tolist() for k, v in H_abs.items()},
                        "canvas": canvas,
                    })

                    # 스티치 미리보기
                    progress.progress(1.0, text="스티치 미리보기 렌더링 중...")
                    ref_bgr = cv2.cvtColor(frame_data[REF_CAM], cv2.COLOR_RGB2BGR)
                    preview_rgb = stitch_one(
                        frame_data,
                        {str(k): v.tolist() for k, v in H_abs.items()},
                        canvas, "feather", ref_bgr,
                        intensity=0.0, brightness=0.0, contrast=1.0,
                    )
                    st.success(f"캘리브레이션 저장 완료 → {HOMO_FILE.name}  "
                               f"| 캔버스: {canvas['out_w']}×{canvas['out_h']}px")
                    st.subheader("스티칭 미리보기")
                    st.image(preview(preview_rgb), width="stretch")

# ════════════════════════════════════════════════════════════
# 탭 4: 스티칭 렌더
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("스티칭 렌더")

    # 캘리브레이션 로드
    lens_render = load_lens()
    homo_data   = load_homo()

    cal_ok    = homo_data is not None
    cam_masks = load_cam_masks()

    col_l, col_m, col_r = st.columns(3)
    with col_l:
        if LENS_FILE.exists():
            st.success("렌즈 캘리브레이션 로드됨", icon="✅")
        else:
            st.warning("렌즈 캘리브레이션 없음 — 탭 1에서 저장하세요.")
    with col_m:
        if cal_ok:
            canvas = homo_data["canvas"]
            st.success(f"리그 캘리브레이션 로드됨  |  "
                       f"캔버스 {canvas['out_w']}×{canvas['out_h']}px", icon="✅")
        else:
            st.warning("리그 캘리브레이션 없음 — 탭 3에서 계산하세요.")
    with col_r:
        if cam_masks:
            mask_summary = ", ".join(f"CAM{k}" for k in sorted(cam_masks))
            st.info(f"마스크 로드됨: {mask_summary}", icon="🎨")
        else:
            st.caption("마스크 없음 (탭 2에서 선택 적용)")

    st.divider()

    if not cal_ok or n < 5:
        st.info("렌더 실행을 위해 소스 파일 5개와 리그 캘리브레이션(탭 3)이 필요합니다.")
        st.stop()

    # ── 렌더 설정 ─────────────────────────────────────────────
    st.subheader("렌더 설정")

    rs1, rs2 = st.columns(2)
    with rs1:
        blend_mode = st.selectbox(
            "블렌딩 방식",
            ["feather", "hard_cut"],
            help="feather: 부드러운 경계 / hard_cut: 날카로운 경계"
        )
        output_scale = st.select_slider(
            "출력 해상도 스케일",
            options=[25, 50, 75, 100],
            value=100,
            format_func=lambda x: f"{x}%"
        )

    with rs2:
        out_fmt = st.selectbox("출력 포맷", ["JPEG (고화질)", "PNG (무손실)"])
        jpeg_q  = st.slider("JPEG 품질", 70, 100, 95, disabled=("PNG" in out_fmt))

    st.subheader("컬러 매칭 (기준: 정면 CAM1)")
    cc1, cc2, cc3 = st.columns(3)
    color_intensity  = cc1.slider("컬러 매칭 강도", 0.0, 1.0, 0.5, 0.05)
    brightness_offset = cc2.slider("밝기 보정", -50.0, 50.0, 0.0, 1.0)
    contrast_scale    = cc3.slider("대비 스케일", 0.5, 2.0, 1.0, 0.05)

    st.divider()
    st.subheader("렌더 범위")

    # 비디오 정보 수집
    video_infos = [get_video_info(str(f)) for f in mov_files]
    min_frames  = min(v["total"] for v in video_infos)
    fps         = video_infos[0]["fps"]

    rc1, rc2, rc3 = st.columns(3)
    start_frame = rc1.number_input("시작 프레임", 0, min_frames - 1, 0, 1)
    end_frame   = rc2.number_input("끝 프레임",   0, min_frames - 1, min(min_frames - 1, 99), 1)
    rc3.metric("렌더 프레임 수", max(0, int(end_frame) - int(start_frame) + 1))
    rc3.caption(f"전체 최소 프레임: {min_frames}  |  {fps:.2f} fps")

    if start_frame > end_frame:
        st.error("시작 프레임이 끝 프레임보다 큽니다.")
        st.stop()

    total_render = int(end_frame) - int(start_frame) + 1

    # 출력 폴더
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    st.divider()

    if st.button(f"렌더 시작  ({total_render} 프레임)", type="primary"):
        H_abs_data = homo_data["H_abs"]
        canvas     = homo_data["canvas"]

        # 출력 스케일 적용
        scale = output_scale / 100.0
        render_canvas = dict(
            x_min  = canvas["x_min"],
            y_min  = canvas["y_min"],
            out_w  = int(canvas["out_w"] * scale),
            out_h  = int(canvas["out_h"] * scale),
        )
        # 스케일 행렬을 H_abs에 합산
        S = np.diag([scale, scale, 1.0])
        H_abs_scaled = {k: (S @ np.array(v, np.float64)).tolist()
                        for k, v in H_abs_data.items()}

        # 비디오 캡처 오픈
        caps = [cv2.VideoCapture(str(f)) for f in mov_files]
        for i, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        ext           = "jpg" if "JPEG" in out_fmt else "png"
        written       = 0

        for fi in range(total_render):
            frame_idx = int(start_frame) + fi
            frames_rgb = {}

            for cam_id in range(1, 6):
                ok, bgr = caps[cam_id - 1].read()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames_rgb[cam_id] = undistort(rgb, lens_render[f"cam{cam_id}"])

            if len(frames_rgb) < 5:
                st.warning(f"프레임 {frame_idx}: 일부 카메라 읽기 실패, 건너뜀")
                continue

            ref_bgr = cv2.cvtColor(frames_rgb[REF_CAM], cv2.COLOR_RGB2BGR)

            result_rgb = stitch_one(
                frames_rgb, H_abs_scaled, render_canvas,
                blend_mode, ref_bgr,
                color_intensity, brightness_offset, contrast_scale,
                cam_masks=cam_masks,
            )

            out_path = frames_dir / f"frame_{frame_idx:06d}.{ext}"
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            if ext == "jpg":
                cv2.imwrite(str(out_path), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
            else:
                cv2.imwrite(str(out_path), result_bgr)

            written += 1
            pct = (fi + 1) / total_render
            progress_bar.progress(pct)
            status_text.text(f"프레임 {frame_idx} / {int(end_frame)}  ({fi+1}/{total_render})")

        for cap in caps:
            cap.release()

        progress_bar.progress(1.0)
        status_text.empty()

        st.success(f"렌더 완료: {written}프레임 → {frames_dir}")
        st.info(
            f"ffmpeg로 영상 인코딩:\n"
            f"```\nffmpeg -framerate {fps:.2f} -i {frames_dir}/frame_%06d.{ext} "
            f"-c:v prores_ks -profile:v 3 output/result.mov\n```"
        )

        # 마지막 프레임 미리보기
        if written > 0:
            last = frames_dir / f"frame_{int(end_frame):06d}.{ext}"
            if last.exists():
                st.subheader("마지막 렌더 프레임")
                last_bgr = cv2.imread(str(last))
                if last_bgr is not None:
                    st.image(preview(cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)), width="stretch")
