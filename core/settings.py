from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CAL_DIR    = BASE_DIR / "calibration"
LENS_FILE        = CAL_DIR / "lens.json"
HOMO_FILE        = CAL_DIR / "homography.json"
RIG_LAYOUT_FILE    = CAL_DIR / "rig_layout.json"
CANVAS_LAYOUT_FILE = CAL_DIR / "canvas_layout.json"
CYL_PARAMS_FILE    = CAL_DIR / "cyl_params.json"
PRESETS_DIR        = CAL_DIR / "presets"
CAM_MAPPING_FILE   = CAL_DIR / "cam_mapping.json"
OUTPUT_RES_FILE    = CAL_DIR / "output_resolution.json"
COLOR_FILE         = CAL_DIR / "color.json"

for d in [INPUT_DIR, OUTPUT_DIR, CAL_DIR, PRESETS_DIR]:
    d.mkdir(exist_ok=True)

CAM_ROLES    = {1: "좌2", 2: "좌1", 3: "정면", 4: "우1", 5: "우2"}
STITCH_ORDER = [1, 2, 3, 4, 5]
REF_CAM      = 3
REF_IDX      = STITCH_ORDER.index(REF_CAM)
ADJ_PAIRS    = [(STITCH_ORDER[i], STITCH_ORDER[i + 1]) for i in range(4)]

RED_KOMODO_PRESET = dict(
    k1=-0.10, k2=0.04, k3=0.00,
    p1=1e-4,  p2=-1e-4,
    fx_ratio=0.780, fy_ratio=0.780,
    cx_ratio=0.500, cy_ratio=0.500,
)

DEFAULT_FOCAL = 1100.0

# Default Yaw: CAM1=좌2(-90), CAM2=좌1(-45), CAM3=정면(0), CAM4=우1(+45), CAM5=우2(+90)
DEFAULT_CYL_PARAMS: dict = {
    "focal": DEFAULT_FOCAL,
    "cam1": {"yaw": -90.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0, "k2": 0.0, "enabled": True},
    "cam2": {"yaw": -45.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0, "k2": 0.0, "enabled": True},
    "cam3": {"yaw":   0.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0, "k2": 0.0, "enabled": True},
    "cam4": {"yaw":  45.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0, "k2": 0.0, "enabled": True},
    "cam5": {"yaw":  90.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0, "k2": 0.0, "enabled": True},
}
