import shutil
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QFrame, QFileDialog, QSizePolicy, QApplication,
    QMenu, QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QMimeData, QUrl, QPoint, QRect
from PyQt6.QtGui import QDrag, QColor, QPixmap, QPainter, QPen, QFont

from core.settings import CAM_ROLES, INPUT_DIR
from core.video import get_mov_files, get_video_info
from core.calibration import load_cam_mapping, save_cam_mapping


# ── Draggable file list ────────────────────────────────────────────────────────
class _DraggableFileList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        mime = QMimeData()
        mime.setText(str(path))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


# ── CamSlot ────────────────────────────────────────────────────────────────────
_CAM_COLORS = {1: "#ffffff", 2: "#ffffff", 3: "#ffffff",
               4: "#ffffff", 5: "#ffffff"}


class CamSlot(QFrame):
    mapping_changed = pyqtSignal(int, object)   # (cam_id, Path|None)

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id   = cam_id
        self._path: Path | None = None
        self._press_pos: QPoint | None = None
        self.setAcceptDrops(True)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui()
        self._update_style(False)

    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        color = _CAM_COLORS.get(self.cam_id, "#888888")
        role  = CAM_ROLES.get(self.cam_id, "")

        lbl_id = QLabel(f"CAM{self.cam_id}")
        lbl_id.setFixedWidth(44)
        lbl_id.setStyleSheet("color:#ffffff; font-weight:bold; font-size:11px;")
        lay.addWidget(lbl_id)

        lbl_role = QLabel(role)
        lbl_role.setFixedWidth(30)
        lbl_role.setStyleSheet("color:#666666; font-size:9px;")
        lay.addWidget(lbl_role)

        self._lbl_file = QLabel("— None —")
        self._lbl_file.setStyleSheet("color:#555555; font-size:9px;")
        self._lbl_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._lbl_file.setWordWrap(False)
        lay.addWidget(self._lbl_file, stretch=1)

        self._btn_clear = QPushButton("✕")
        self._btn_clear.setFixedSize(18, 18)
        self._btn_clear.setStyleSheet(
            "QPushButton{color:#ff4444;background:transparent;border:none;font-size:11px;}"
            "QPushButton:hover{color:#ff8888;}")
        self._btn_clear.setVisible(False)
        self._btn_clear.clicked.connect(self._clear)
        lay.addWidget(self._btn_clear)

    # ── drag & drop ──────────────────────────────────────────
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasText() or ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self._update_style(True)

    def dragLeaveEvent(self, ev):
        self._update_style(self._path is not None)

    def dropEvent(self, ev):
        path = None
        if ev.mimeData().hasText():
            path = Path(ev.mimeData().text().strip())
        elif ev.mimeData().hasUrls():
            urls = ev.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
        if path and path.exists():
            self.set_path(path)
        self._update_style(self._path is not None)
        ev.acceptProposedAction()

    # ── click → file dialog  /  drag → canvas ────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.pos()
            ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            if (ev.pos() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
                p, _ = QFileDialog.getOpenFileName(
                    self, f"CAM{self.cam_id} Source Select",
                    str(INPUT_DIR),
                    "Video files (*.mov *.mp4 *.mxf *.avi *.r3d);;All files (*)"
                )
                if p:
                    self.set_path(Path(p))
            self._press_pos = None

    def mouseMoveEvent(self, ev):
        if (self._press_pos is not None and
                (ev.pos() - self._press_pos).manhattanLength()
                >= QApplication.startDragDistance()):
            self._press_pos = None
            self._start_cam_drag()

    def _start_cam_drag(self):
        pm = QPixmap(72, 26)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.setBrush(QColor(0, 0, 0, 200))
        p.drawRoundedRect(1, 1, 70, 24, 4, 4)
        p.setPen(QColor("#ffffff"))
        font = QFont("Consolas", 9)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRect(0, 0, 72, 26), Qt.AlignmentFlag.AlignCenter, f"CAM{self.cam_id}")
        p.end()
        mime = QMimeData()
        mime.setData("application/x-nexstitch-cam", str(self.cam_id).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)

    # ── public ───────────────────────────────────────────────
    def set_path(self, path: Path | None):
        self._path = path
        if path:
            self._lbl_file.setText(path.name)
            self._lbl_file.setStyleSheet("color:#cccccc; font-size:9px;")
            self._btn_clear.setVisible(True)
        else:
            self._lbl_file.setText("— None —")
            self._lbl_file.setStyleSheet("color:#555555; font-size:9px;")
            self._btn_clear.setVisible(False)
        self._update_style(path is not None)
        self.mapping_changed.emit(self.cam_id, path)

    def get_path(self) -> Path | None:
        return self._path

    def _clear(self):
        self.set_path(None)

    def _update_style(self, active: bool):
        border = "2px solid #ffffff66" if active else "1px solid #333333"
        bg     = "#1e1e1e" if active else "#141414"
        self.setStyleSheet(
            f"CamSlot{{background:{bg}; border:{border}; border-radius:4px;}}"
        )


# ── SourcePanel ────────────────────────────────────────────────────────────────
class SourcePanel(QWidget):
    camera_selected = pyqtSignal(int)           # cam_id (1-based)
    files_refreshed = pyqtSignal(dict)          # dict[int, Path|None]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_files: list[Path] = []
        self._linked_files: list[Path] = []          # "Link" 옵션으로 추가된 파일들
        self._mapping:   dict[int, Path | None] = {i: None for i in range(1, 6)}
        self._build_ui()
        self._load_saved_mapping()

    def _build_ui(self):
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)

        # ── File list section ──────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr1 = QLabel("SOURCE FILES")
        hdr1.setStyleSheet("color:#888888; font-size:9px; font-weight:bold; letter-spacing:1px;")
        hdr_row.addWidget(hdr1)
        hdr_row.addStretch()

        btn_add = QPushButton("+")
        btn_add.setFixedSize(20, 20)
        btn_add.setToolTip("Add video files (.mov / .mp4 / .r3d)")
        btn_add.setStyleSheet(
            "QPushButton{color:#00ff88; background:#0d1f14; border:1px solid #00ff8866;"
            " border-radius:3px; font-size:13px; font-weight:bold; padding-bottom:1px;}"
            "QPushButton:hover{background:#1a3a2a; border-color:#00ff88;}"
            "QPushButton:pressed{background:#0a2a1a;}"
        )
        btn_add.clicked.connect(self._add_files)
        hdr_row.addWidget(btn_add)
        lay.addLayout(hdr_row)

        self._file_list = _DraggableFileList()
        self._file_list.setFixedHeight(160)
        self._file_list.setStyleSheet(
            "QListWidget{background:#0d0d0d; border:1px solid #2a2a2a; font-size:9px;}"
            "QListWidget::item:selected{background:#1a3a2a;}"
        )
        self._file_list.itemClicked.connect(self._on_list_clicked)
        self._file_list.customContextMenuRequested.connect(self._show_context_menu)
        lay.addWidget(self._file_list)

        self._lbl_status = QLabel("0 files")
        self._lbl_status.setStyleSheet("color:#888888; font-size:9px;")
        lay.addWidget(self._lbl_status)

        btn_row = QHBoxLayout()
        btn_ref = QPushButton("Refresh F5")
        btn_ref.setFixedHeight(22)
        btn_ref.clicked.connect(self.refresh)
        btn_row.addWidget(btn_ref)

        btn_auto = QPushButton("Auto Mapping")
        btn_auto.setFixedHeight(22)
        btn_auto.setToolTip("Auto-assign files to CAM1~5 in list order")
        btn_auto.clicked.connect(self._auto_map)
        btn_row.addWidget(btn_auto)
        lay.addLayout(btn_row)

        # ── Separator ─────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#2a2a2a;")
        lay.addWidget(sep)

        # ── Cam slots section ─────────────────────────────
        hdr2 = QLabel("CAMERA MAPPING")
        hdr2.setStyleSheet("color:#888888; font-size:9px; font-weight:bold; letter-spacing:1px;")
        lay.addWidget(hdr2)

        hint = QLabel("Drag or click to assign source")
        hint.setStyleSheet("color:#444444; font-size:8px;")
        lay.addWidget(hint)

        self._slots: dict[int, CamSlot] = {}
        for cam_id in range(1, 6):
            slot = CamSlot(cam_id)
            slot.mapping_changed.connect(self._on_slot_changed)
            lay.addWidget(slot)
            self._slots[cam_id] = slot

        # ── Save mapping button ────────────────────────────
        btn_save = QPushButton("Save Mapping")
        btn_save.setFixedHeight(24)
        btn_save.setStyleSheet("background:#1a2a1a; color:#00ff88; border:1px solid #00ff88;")
        btn_save.clicked.connect(self._save_mapping)
        lay.addWidget(btn_save)

        lay.addStretch()

    # ── public ────────────────────────────────────────────────
    def refresh(self):
        self._all_files = get_mov_files()
        # Merge linked files that aren't already discovered
        existing_lower = {f.name.lower() for f in self._all_files}
        for lf in self._linked_files:
            if lf.exists() and lf.name.lower() not in existing_lower:
                self._all_files.append(lf)
                existing_lower.add(lf.name.lower())

        self._file_list.clear()
        for f in self._all_files:
            self._add_item_to_list(f)

        n = len(self._all_files)
        color = "#00ff88" if n >= 5 else "#ffaa00" if n > 0 else "#ff4444"
        self._lbl_status.setText(f"{n} files found")
        self._lbl_status.setStyleSheet(f"color:{color}; font-size:9px;")
        self._emit_mapping()

    def _add_item_to_list(self, path: Path):
        try:
            info = get_video_info(str(path))
            label = f"{path.name}  {info['w']}×{info['h']} {info['total']}f"
        except Exception:
            label = path.name
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        self._file_list.addItem(item)

    # ── + 버튼: 파일 추가 ─────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Source Files",
            str(INPUT_DIR),
            "Video files (*.mov *.mp4 *.r3d *.MOV *.MP4 *.R3D);;All files (*)"
        )
        if not paths:
            return

        # Copy / Link 선택 다이얼로그
        msg = QMessageBox(self)
        msg.setWindowTitle("Add Files")
        msg.setText(f"{len(paths)}개 파일을 추가합니다.")
        msg.setInformativeText(
            "Copy to project folder  →  input/ 폴더로 복사\n"
            "Link original  →  원본 경로 그대로 참조"
        )
        msg.setIcon(QMessageBox.Icon.Question)
        btn_copy   = msg.addButton("Copy",   QMessageBox.ButtonRole.AcceptRole)
        btn_link   = msg.addButton("Link",   QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is btn_cancel or clicked is None:
            return

        do_copy = (clicked is btn_copy)
        existing_lower = {f.name.lower() for f in self._all_files}

        for p_str in paths:
            src = Path(p_str)
            if not src.exists():
                continue

            if do_copy:
                dest = INPUT_DIR / src.name
                if not dest.exists():
                    try:
                        shutil.copy2(src, dest)
                    except Exception as e:
                        QMessageBox.warning(self, "Copy Error", f"{src.name}\n{e}")
                        continue
                target = dest
            else:
                # Link: 원본 경로 그대로
                if src not in self._linked_files:
                    self._linked_files.append(src)
                target = src

            if target.name.lower() not in existing_lower:
                self._all_files.append(target)
                existing_lower.add(target.name.lower())
                self._add_item_to_list(target)

        n = len(self._all_files)
        color = "#00ff88" if n >= 5 else "#ffaa00" if n > 0 else "#ff4444"
        self._lbl_status.setText(f"{n} files found")
        self._lbl_status.setStyleSheet(f"color:{color}; font-size:9px;")
        self._emit_mapping()

    # ── 우클릭 컨텍스트 메뉴 ──────────────────────────────────
    def _show_context_menu(self, pos):
        item = self._file_list.itemAt(pos)
        if item is None:
            return
        path_str = item.data(Qt.ItemDataRole.UserRole)
        path = Path(path_str) if path_str else None

        menu = QMenu(self)
        act_remove = menu.addAction("Remove from list")
        act_show   = menu.addAction("Show in Explorer")

        chosen = menu.exec(self._file_list.mapToGlobal(pos))

        if chosen is act_remove:
            row = self._file_list.row(item)
            self._file_list.takeItem(row)
            if path and path in self._all_files:
                self._all_files.remove(path)
            if path and path in self._linked_files:
                self._linked_files.remove(path)
            n = len(self._all_files)
            color = "#00ff88" if n >= 5 else "#ffaa00" if n > 0 else "#ff4444"
            self._lbl_status.setText(f"{n} files found")
            self._lbl_status.setStyleSheet(f"color:{color}; font-size:9px;")
            self._emit_mapping()

        elif chosen is act_show and path:
            subprocess.Popen(f'explorer /select,"{path}"')

    @property
    def files(self) -> list[Path]:
        return self._all_files

    # ── slots ──────────────────────────────────────────────────
    def _on_list_clicked(self, item: QListWidgetItem):
        # clicking a file in the list just selects it (drag it to assign)
        pass

    def _on_slot_changed(self, cam_id: int, path):
        self._mapping[cam_id] = Path(path) if path else None
        self._emit_mapping()
        self._autosave_mapping()

    def _autosave_mapping(self):
        data = {cam_id: str(p) if p else None for cam_id, p in self._mapping.items()}
        save_cam_mapping(data)

    def _auto_map(self):
        for cam_id in range(1, 6):
            idx = cam_id - 1
            path = self._all_files[idx] if idx < len(self._all_files) else None
            self._slots[cam_id].set_path(path)

    def _save_mapping(self):
        data = {cam_id: str(p) if p else None for cam_id, p in self._mapping.items()}
        save_cam_mapping(data)

    def _load_saved_mapping(self):
        saved = load_cam_mapping()
        has_valid = False
        for cam_id, path_str in saved.items():
            if path_str:
                p = Path(path_str)
                if p.exists():
                    self._slots[cam_id].set_path(p)
                    has_valid = True
        self.refresh()  # populates self._all_files
        # No saved mapping → auto-map from input dir
        if not has_valid and self._all_files:
            self._auto_map()

    def _emit_mapping(self):
        self.files_refreshed.emit(dict(self._mapping))
