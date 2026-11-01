"""NexStitch — Project model.

Each project is a directory containing:
  project.nxs      – JSON metadata
  input/           – source video files
  proxy/           – proxy media cache
  output/          – rendered panoramas
  calibration/     – lens / stitch calibration data
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

NXS_FILENAME = "project.nxs"
CAMERA_RIGS  = ["RED Komodo 5-Camera"]


class Project:
    """In-memory representation of an open .nxs project."""

    def __init__(self, root: Path, data: dict):
        self.root  = root
        self._data = data

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def nxs_path(self) -> Path:
        return self.root / NXS_FILENAME

    @property
    def name(self) -> str:
        return self._data.get("name", self.root.name)

    @property
    def camera_rig(self) -> str:
        return self._data.get("camera_rig", CAMERA_RIGS[0])

    @property
    def input_dir(self) -> Path:
        return self.root / self._data.get("input", "input")

    @property
    def output_dir(self) -> Path:
        return self.root / self._data.get("output", "output")

    @property
    def proxy_dir(self) -> Path:
        return self.root / "proxy"

    @property
    def cal_dir(self) -> Path:
        return self.root / self._data.get("calibration", "calibration")

    @property
    def modified(self) -> str:
        return self._data.get("modified", "")

    @property
    def created(self) -> str:
        return self._data.get("created", "")

    @property
    def output_resolution(self) -> list:
        return self._data.get("output_resolution", [7680, 2160])

    # ── I/O ──────────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, parent_dir: Path, name: str,
               camera_rig: str = CAMERA_RIGS[0]) -> "Project":
        """Create a new project folder structure and write project.nxs."""
        root = parent_dir / name
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("input", "proxy", "output",
                    "calibration", "calibration/presets"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        now  = datetime.now().isoformat(timespec="seconds")
        data = {
            "name":              name,
            "created":           now,
            "modified":          now,
            "camera_rig":        camera_rig,
            "output_resolution": [7680, 2160],
            "calibration":       "calibration",
            "input":             "input",
            "output":            "output",
        }
        (root / NXS_FILENAME).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), "utf-8"
        )
        return cls(root, data)

    @classmethod
    def load(cls, nxs_path: Path) -> "Project":
        data = json.loads(nxs_path.read_text("utf-8"))
        return cls(nxs_path.parent, data)

    def touch_modified(self):
        """Stamp the current time into modified and re-save project.nxs."""
        self._data["modified"] = datetime.now().isoformat(timespec="seconds")
        self.nxs_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), "utf-8"
        )

    # ── dynamic path patching ─────────────────────────────────────────────────

    def apply_to_settings(self):
        """Redirect core.settings (and all modules that imported from it)
        to point at this project's directory structure.

        Must be called BEFORE importing MainWindow / SourcePanel / etc.,
        so that their `from core.settings import X` picks up the new paths.
        """
        import core.settings as s

        s.INPUT_DIR  = self.input_dir
        s.OUTPUT_DIR = self.output_dir
        s.CAL_DIR    = self.cal_dir

        # Re-derive all file-level paths
        s.LENS_FILE          = s.CAL_DIR / "lens.json"
        s.HOMO_FILE          = s.CAL_DIR / "homography.json"
        s.RIG_LAYOUT_FILE    = s.CAL_DIR / "rig_layout.json"
        s.CANVAS_LAYOUT_FILE = s.CAL_DIR / "canvas_layout.json"
        s.CYL_PARAMS_FILE    = s.CAL_DIR / "cyl_params.json"
        s.PRESETS_DIR        = s.CAL_DIR / "presets"
        s.CAM_MAPPING_FILE   = s.CAL_DIR / "cam_mapping.json"
        s.OUTPUT_RES_FILE    = s.CAL_DIR / "output_resolution.json"
        s.COLOR_FILE         = s.CAL_DIR / "color.json"

        # Ensure all directories exist
        for d in (self.input_dir, self.output_dir, self.cal_dir,
                  self.proxy_dir, s.PRESETS_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # Patch core.calibration (imports names directly from settings)
        import core.calibration as cal
        cal.CAL_DIR          = s.CAL_DIR
        cal.LENS_FILE        = s.LENS_FILE
        cal.HOMO_FILE        = s.HOMO_FILE
        cal.RIG_LAYOUT_FILE  = s.RIG_LAYOUT_FILE
        cal.CANVAS_LAYOUT_FILE = s.CANVAS_LAYOUT_FILE
        cal.CYL_PARAMS_FILE  = s.CYL_PARAMS_FILE
        cal.PRESETS_DIR      = s.PRESETS_DIR
        cal.CAM_MAPPING_FILE = s.CAM_MAPPING_FILE
        cal.OUTPUT_RES_FILE  = s.OUTPUT_RES_FILE
        cal.COLOR_FILE       = s.COLOR_FILE

        # Patch core.video (imports INPUT_DIR directly from settings)
        import core.video as vid
        vid.INPUT_DIR = self.input_dir
