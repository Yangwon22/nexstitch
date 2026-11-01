import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSlider, QComboBox,
    QGroupBox, QScrollArea, QProgressBar, QSpinBox, QCheckBox,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from PyQt6.QtGui import QPixmap

from core.calibration import (
    load_lens, save_lens, load_homo, save_homo, undistort,
    match_sift, find_H, compute_H_abs, compute_canvas, load_cam_masks,
)
from core.settings import (
    CAM_ROLES, ADJ_PAIRS, STITCH_ORDER, RED_KOMODO_PRESET,
    CAL_DIR, OUTPUT_DIR,
)
from core.video import get_video_info, grab_frame
from ui.utils import numpy_to_pixmap, fit_pixmap
from ui.workers import RigCalibWorker, RenderWorker


# ── 렌즈 캘리브레이션 탭 ─────────────────────────────────────
class LensTab(QWidget):
    calibration_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mov_files = []
        self._params    = load_lens()
        self._spinboxes: dict[str, dict[str, QDoubleSpinBox]] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # 상단 버튼
        btn_row = QHBoxLayout()
        btn_preset = QPushButton("RED Komodo 21mm 프리셋 전체 적용")
        btn_preset.clicked.connect(self._apply_preset_all)
        btn_row.addWidget(btn_preset)
        btn_save = QPushButton("렌즈 파라미터 저장")
        btn_save.setObjectName("accent")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#00ff88; font-size:10px;")
        root.addWidget(self._status_lbl)

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(6)

        for cam_id in range(1, 6):
            box = QGroupBox(f"CAM{cam_id}  {CAM_ROLES[cam_id]}")
            gl  = QGridLayout(box)
            key = f"cam{cam_id}"
            p   = self._params[key]
            self._spinboxes[key] = {}

            fields = [
                ("k1", "k1 (방사형)", -2.0, 2.0, 0.001),
                ("k2", "k2 (방사형)", -2.0, 2.0, 0.001),
                ("k3", "k3 (방사형)", -2.0, 2.0, 0.001),
                ("p1", "p1 (접선형)", -0.5, 0.5, 0.0001),
                ("p2", "p2 (접선형)", -0.5, 0.5, 0.0001),
                ("fx_ratio", "fx 비율",  0.1, 3.0, 0.001),
                ("fy_ratio", "fy 비율",  0.1, 3.0, 0.001),
                ("cx_ratio", "cx 주점",  0.0, 1.0, 0.001),
                ("cy_ratio", "cy 주점",  0.0, 1.0, 0.001),
            ]
            for col, (fname, flabel, fmin, fmax, fstep) in enumerate(fields):
                gl.addWidget(QLabel(flabel), 0, col)
                sb = QDoubleSpinBox()
                sb.setRange(fmin, fmax)
                sb.setSingleStep(fstep)
                sb.setDecimals(5)
                sb.setValue(float(p[fname]))
                sb.setFixedWidth(90)
                gl.addWidget(sb, 1, col)
                self._spinboxes[key][fname] = sb

            # 프리셋 버튼
            btn_p = QPushButton("프리셋")
            btn_p.setFixedWidth(70)
            btn_p.clicked.connect(lambda _, k=key: self._apply_preset(k))
            gl.addWidget(btn_p, 1, len(fields))

            inner_lay.addWidget(box)

        inner_lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

    def _apply_preset(self, key: str):
        for fname, sb in self._spinboxes[key].items():
            sb.setValue(float(RED_KOMODO_PRESET[fname]))

    def _apply_preset_all(self):
        for key in self._spinboxes:
            self._apply_preset(key)

    def _collect_params(self) -> dict:
        params = {}
        for key, fields in self._spinboxes.items():
            params[key] = {fname: sb.value() for fname, sb in fields.items()}
        return params

    def _save(self):
        params = self._collect_params()
        save_lens(params)
        self._params = params
        self._status_lbl.setText(f"저장 완료 → lens.json")
        self.calibration_saved.emit()

    def set_mov_files(self, files):
        self._mov_files = files

    def get_params(self) -> dict:
        return self._collect_params()


# ── 마스크 탭 ────────────────────────────────────────────────
class MaskTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mov_files = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(QLabel("마스크 편집은 중앙 뷰어의 '마스크 모드' 버튼을 사용하세요."))

        # 카메라 선택
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("카메라"))
        self._cam_combo = QComboBox()
        for i in range(1, 6):
            self._cam_combo.addItem(f"CAM{i}  {CAM_ROLES[i]}", i)
        row1.addWidget(self._cam_combo)

        row1.addWidget(QLabel("브러시 크기"))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(5, 150)
        self._brush_slider.setValue(30)
        self._brush_slider.setFixedWidth(120)
        row1.addWidget(self._brush_slider)
        self._brush_lbl = QLabel("30")
        self._brush_lbl.setFixedWidth(28)
        self._brush_slider.valueChanged.connect(lambda v: self._brush_lbl.setText(str(v)))
        row1.addWidget(self._brush_lbl)
        row1.addStretch()
        root.addLayout(row1)

        # 버튼 행
        btn_row = QHBoxLayout()
        self._btn_save  = QPushButton("마스크 저장")
        self._btn_save.setObjectName("accent")
        self._btn_save.clicked.connect(self._save_mask)
        btn_row.addWidget(self._btn_save)

        self._btn_load  = QPushButton("마스크 불러오기")
        self._btn_load.clicked.connect(self._load_mask)
        btn_row.addWidget(self._btn_load)

        self._btn_clear = QPushButton("초기화")
        self._btn_clear.setObjectName("danger")
        self._btn_clear.clicked.connect(self._clear_mask)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # 저장 현황
        self._status_grid = QGridLayout()
        for i, cam_id in enumerate(range(1, 6)):
            lbl = QLabel(f"CAM{cam_id}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#888888; font-size:10px;")
            self._status_grid.addWidget(lbl, 0, i)
        self._mask_status_lbls: list[QLabel] = []
        for i in range(5):
            lbl = QLabel("없음")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#555555; font-size:10px;")
            self._status_grid.addWidget(lbl, 1, i)
            self._mask_status_lbls.append(lbl)
        root.addLayout(self._status_grid)
        root.addStretch()

        self._viewer_ref = None  # set by MainWindow
        self._refresh_status()

    def set_viewer(self, viewer):
        self._viewer_ref = viewer
        self._brush_slider.valueChanged.connect(viewer.set_brush_size)

    def set_mov_files(self, files):
        self._mov_files = files

    @property
    def selected_cam(self) -> int:
        return self._cam_combo.currentData()

    def _save_mask(self):
        if self._viewer_ref is None:
            return
        mask = self._viewer_ref.get_mask()
        if mask is None:
            return
        cam_id = self.selected_cam
        path   = CAL_DIR / f"mask_CAM{cam_id}.png"
        cv2.imwrite(str(path), mask)
        self._refresh_status()

    def _load_mask(self):
        if self._viewer_ref is None:
            return
        cam_id = self.selected_cam
        path   = CAL_DIR / f"mask_CAM{cam_id}.png"
        if path.exists():
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                self._viewer_ref.set_mask(m)

    def _clear_mask(self):
        if self._viewer_ref:
            self._viewer_ref._viewer.clear_mask()
        cam_id = self.selected_cam
        path   = CAL_DIR / f"mask_CAM{cam_id}.png"
        if path.exists():
            path.unlink()
        self._refresh_status()

    def _refresh_status(self):
        for i, cam_id in enumerate(range(1, 6)):
            p   = CAL_DIR / f"mask_CAM{cam_id}.png"
            lbl = self._mask_status_lbls[i]
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                pct = (m > 127).sum() / m.size * 100 if m is not None else 0
                lbl.setText(f"{pct:.1f}%")
                lbl.setStyleSheet("color:#00ff88; font-size:10px;")
            else:
                lbl.setText("없음")
                lbl.setStyleSheet("color:#555555; font-size:10px;")


# ── 리그 캘리브레이션 탭 ─────────────────────────────────────
class RigTab(QWidget):
    calibration_saved = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mov_files  = []
        self._lens_tab   = None   # reference to LensTab
        self._worker: RigCalibWorker | None = None
        self._frame_data: dict = {}
        self._overlap_spins: dict = {}
        self._frame_spins:   dict = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # 배치 다이어그램
        diag_row = QHBoxLayout()
        for i, cam_id in enumerate(STITCH_ORDER):
            lbl = QLabel(f"CAM{cam_id}\n{CAM_ROLES[cam_id]}" + ("\n★ 기준" if cam_id == 1 else ""))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "background:#2d2d2d; border:1px solid #00ff88; border-radius:3px; "
                "color:#00ff88; font-size:10px; padding:4px;" if cam_id == 1 else
                "background:#2d2d2d; border:1px solid #3a3a3a; border-radius:3px; "
                "color:#aaaaaa; font-size:10px; padding:4px;"
            )
            diag_row.addWidget(lbl)
        root.addLayout(diag_row)

        # 오버랩 슬라이더
        ov_box = QGroupBox("인접 카메라 오버랩 (%)")
        ov_gl  = QGridLayout(ov_box)
        for col, (a, b) in enumerate(ADJ_PAIRS):
            ov_gl.addWidget(QLabel(f"CAM{a}↔{b}"), 0, col)
            sp = QSpinBox()
            sp.setRange(5, 60)
            sp.setValue(30)
            sp.setSuffix("%")
            ov_gl.addWidget(sp, 1, col)
            self._overlap_spins[(a, b)] = sp
        root.addWidget(ov_box)

        # 프레임 선택
        fr_box = QGroupBox("캘리브레이션 프레임 선택")
        fr_gl  = QGridLayout(fr_box)
        for col, cam_id in enumerate(range(1, 6)):
            fr_gl.addWidget(QLabel(f"CAM{cam_id}"), 0, col)
            sp = QSpinBox()
            sp.setRange(0, 9999)
            sp.setValue(0)
            fr_gl.addWidget(sp, 1, col)
            self._frame_spins[cam_id] = sp
        root.addWidget(fr_box)

        # 계산 버튼 + 진행바
        btn_row = QHBoxLayout()
        self._btn_compute = QPushButton("호모그래피 계산 및 저장")
        self._btn_compute.setObjectName("accent")
        self._btn_compute.clicked.connect(self._compute)
        btn_row.addWidget(self._btn_compute)

        self._homo_status = QLabel("")
        self._homo_status.setStyleSheet("color:#888888; font-size:10px;")
        btn_row.addWidget(self._homo_status, stretch=1)
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._err_lbl = QLabel("")
        self._err_lbl.setStyleSheet("color:#ff4444; font-size:10px;")
        self._err_lbl.setWordWrap(True)
        root.addWidget(self._err_lbl)

        root.addStretch()

        # 저장된 캘리브레이션 표시
        if load_homo():
            d = load_homo()
            c = d["canvas"]
            self._homo_status.setText(
                f"로드됨: {c['out_w']}×{c['out_h']}px"
            )
            self._homo_status.setStyleSheet("color:#00ff88; font-size:10px;")

    def set_mov_files(self, files):
        self._mov_files = files
        for cam_id in range(1, 6):
            if cam_id - 1 < len(files):
                info = get_video_info(str(files[cam_id - 1]))
                self._frame_spins[cam_id].setMaximum(info["total"] - 1)
                self._frame_spins[cam_id].setValue((info["total"] - 1) // 2)

    def set_lens_tab(self, lens_tab):
        self._lens_tab = lens_tab

    def _compute(self):
        if len(self._mov_files) < 5:
            self._err_lbl.setText("소스 파일 5개가 필요합니다.")
            return

        lens_p = self._lens_tab.get_params() if self._lens_tab else load_lens()
        self._frame_data = {}
        self._err_lbl.setText("프레임 로딩 중...")

        for cam_id in range(1, 6):
            fidx = self._frame_spins[cam_id].value()
            rgb  = grab_frame(str(self._mov_files[cam_id - 1]), fidx)
            if rgb is None:
                self._err_lbl.setText(f"CAM{cam_id} 프레임 추출 실패")
                return
            self._frame_data[cam_id] = undistort(rgb, lens_p[f"cam{cam_id}"])

        self._err_lbl.setText("")
        ref_info = get_video_info(str(self._mov_files[0]))
        overlap  = {k: sp.value() for k, sp in self._overlap_spins.items()}

        self._worker = RigCalibWorker(
            self._frame_data, overlap, ref_info["w"], ref_info["h"]
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_compute.setEnabled(False)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._homo_status.setText(msg)
        self._homo_status.setStyleSheet("color:#aaaaaa; font-size:10px;")

    def _on_finished(self, data: dict):
        save_homo(data)
        self._btn_compute.setEnabled(True)
        self._progress.setVisible(False)
        c = data["canvas"]
        self._homo_status.setText(f"저장 완료: {c['out_w']}×{c['out_h']}px")
        self._homo_status.setStyleSheet("color:#00ff88; font-size:10px;")
        self.calibration_saved.emit(data)

    def _on_error(self, msg: str):
        self._btn_compute.setEnabled(True)
        self._progress.setVisible(False)
        self._err_lbl.setText(msg)


# ── 렌더 탭 ─────────────────────────────────────────────────
class RenderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mov_files  = []
        self._worker: RenderWorker | None = None
        self._lens_tab   = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # 캘리브레이션 상태
        self._cal_lbl = QLabel("")
        root.addWidget(self._cal_lbl)
        self._refresh_cal_status()

        # 렌더 설정
        settings_box = QGroupBox("렌더 설정")
        sg = QGridLayout(settings_box)

        sg.addWidget(QLabel("블렌딩"), 0, 0)
        self._blend_combo = QComboBox()
        self._blend_combo.addItems(["feather", "hard_cut"])
        sg.addWidget(self._blend_combo, 0, 1)

        sg.addWidget(QLabel("해상도 스케일"), 0, 2)
        self._scale_combo = QComboBox()
        for s in ["25%", "50%", "75%", "100%"]:
            self._scale_combo.addItem(s, int(s[:-1]))
        self._scale_combo.setCurrentIndex(3)
        sg.addWidget(self._scale_combo, 0, 3)

        sg.addWidget(QLabel("출력 포맷"), 1, 0)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["JPEG (고화질)", "PNG (무손실)"])
        sg.addWidget(self._fmt_combo, 1, 1)

        sg.addWidget(QLabel("JPEG 품질"), 1, 2)
        self._jpeg_spin = QSpinBox()
        self._jpeg_spin.setRange(70, 100)
        self._jpeg_spin.setValue(95)
        sg.addWidget(self._jpeg_spin, 1, 3)

        root.addWidget(settings_box)

        # 컬러 매칭
        color_box = QGroupBox("컬러 매칭 (기준: CAM1)")
        cg = QGridLayout(color_box)

        cg.addWidget(QLabel("컬러 강도"), 0, 0)
        self._color_sl = QSlider(Qt.Orientation.Horizontal)
        self._color_sl.setRange(0, 100)
        self._color_sl.setValue(50)
        cg.addWidget(self._color_sl, 0, 1)

        cg.addWidget(QLabel("밝기 보정"), 1, 0)
        self._bright_sp = QDoubleSpinBox()
        self._bright_sp.setRange(-50, 50)
        self._bright_sp.setValue(0)
        cg.addWidget(self._bright_sp, 1, 1)

        cg.addWidget(QLabel("대비 스케일"), 1, 2)
        self._contrast_sp = QDoubleSpinBox()
        self._contrast_sp.setRange(0.5, 2.0)
        self._contrast_sp.setSingleStep(0.05)
        self._contrast_sp.setValue(1.0)
        cg.addWidget(self._contrast_sp, 1, 3)

        root.addWidget(color_box)

        # 렌더 범위
        range_box = QGroupBox("렌더 범위")
        rg = QHBoxLayout(range_box)
        rg.addWidget(QLabel("시작 프레임"))
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, 99999)
        rg.addWidget(self._start_spin)
        rg.addWidget(QLabel("끝 프레임"))
        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, 99999)
        self._end_spin.setValue(99)
        rg.addWidget(self._end_spin)
        rg.addStretch()
        root.addWidget(range_box)

        # 버튼 + 진행바
        btn_row = QHBoxLayout()
        self._btn_render = QPushButton("렌더 시작")
        self._btn_render.setObjectName("accent")
        self._btn_render.clicked.connect(self._start_render)
        btn_row.addWidget(self._btn_render)

        self._btn_abort = QPushButton("중단")
        self._btn_abort.setObjectName("danger")
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._abort_render)
        btn_row.addWidget(self._btn_abort)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#aaaaaa; font-size:10px;")
        root.addWidget(self._status_lbl)

        root.addStretch()

    def _refresh_cal_status(self):
        from core.calibration import load_homo
        from core.settings import LENS_FILE
        parts = []
        parts.append("✅ 렌즈" if LENS_FILE.exists() else "❌ 렌즈")
        hd = load_homo()
        if hd:
            c = hd["canvas"]
            parts.append(f"✅ 리그 ({c['out_w']}×{c['out_h']}px)")
        else:
            parts.append("❌ 리그")
        self._cal_lbl.setText("  ".join(parts))
        self._cal_lbl.setStyleSheet("font-size:10px; color:#aaaaaa;")

    def set_mov_files(self, files):
        self._mov_files = files
        if files:
            infos = [get_video_info(str(f)) for f in files]
            min_f = min(v["total"] for v in infos)
            self._start_spin.setMaximum(min_f - 1)
            self._end_spin.setMaximum(min_f - 1)
            self._end_spin.setValue(min(99, min_f - 1))

    def set_lens_tab(self, lens_tab):
        self._lens_tab = lens_tab

    def _start_render(self):
        self._refresh_cal_status()
        homo_data = load_homo()
        if not homo_data:
            self._status_lbl.setText("리그 캘리브레이션이 필요합니다.")
            return
        if len(self._mov_files) < 5:
            self._status_lbl.setText("소스 파일 5개가 필요합니다.")
            return

        lens_p   = self._lens_tab.get_params() if self._lens_tab else load_lens()
        masks    = load_cam_masks()
        frames_dir = OUTPUT_DIR / "frames"
        frames_dir.mkdir(exist_ok=True)

        self._worker = RenderWorker(
            mov_files   = self._mov_files,
            homo_data   = homo_data,
            lens_params = lens_p,
            cam_masks   = masks,
            blend       = self._blend_combo.currentText(),
            scale       = self._scale_combo.currentData() / 100.0,
            start_frame = self._start_spin.value(),
            end_frame   = self._end_spin.value(),
            intensity   = self._color_sl.value() / 100.0,
            brightness  = self._bright_sp.value(),
            contrast    = self._contrast_sp.value(),
            out_fmt     = self._fmt_combo.currentText(),
            jpeg_q      = self._jpeg_spin.value(),
            frames_dir  = frames_dir,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._btn_render.setEnabled(False)
        self._btn_abort.setEnabled(True)
        self._progress.setValue(0)
        self._worker.start()

    def _abort_render(self):
        if self._worker:
            self._worker.abort()
        self._btn_abort.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_finished(self, out_dir: str):
        self._btn_render.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._progress.setValue(100)
        self._status_lbl.setText(f"렌더 완료 → {out_dir}")
        self._status_lbl.setStyleSheet("color:#00ff88; font-size:10px;")

    def _on_error(self, msg: str):
        self._btn_render.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._status_lbl.setText(f"오류: {msg}")
        self._status_lbl.setStyleSheet("color:#ff4444; font-size:10px;")


# ── 통합 파라미터 탭위젯 ─────────────────────────────────────
class ParamTabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(280)

        self.lens_tab   = LensTab()
        self.mask_tab   = MaskTab()
        self.rig_tab    = RigTab()
        self.render_tab = RenderTab()

        self.addTab(self.lens_tab,   "🔭 렌즈 캘리브레이션")
        self.addTab(self.mask_tab,   "🎨 마스크")
        self.addTab(self.rig_tab,    "📐 리그 캘리브레이션")
        self.addTab(self.render_tab, "🎬 렌더")

        self.rig_tab.set_lens_tab(self.lens_tab)
        self.render_tab.set_lens_tab(self.lens_tab)

    def set_mov_files(self, files):
        self.lens_tab.set_mov_files(files)
        self.mask_tab.set_mov_files(files)
        self.rig_tab.set_mov_files(files)
        self.render_tab.set_mov_files(files)
