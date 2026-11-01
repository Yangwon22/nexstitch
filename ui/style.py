# ── Python color constants (import these instead of hardcoding) ──────────────
C_BG      = "#1a1a1a"   # main background
C_PANEL   = "#181818"   # panel / sidebar background
C_DEEP    = "#111111"   # toolbar / deep background
C_INPUT   = "#0d0d0d"   # input field background

C_ACCENT  = "#00ff88"   # primary green accent (active / selected)
C_ACCENT2 = "#00aa55"   # dimmer green (hover / fill)

C_WHITE   = "#ffffff"   # full white — primary text / borders
C_TEXT    = "#cccccc"   # normal body text
C_DIM     = "#888888"   # dimmed / secondary text
C_MUTED   = "#666666"   # inactive / hint text
C_DARK    = "#555555"   # very dim text

C_BORDER  = "#2a2a2a"   # standard border
C_BORDER2 = "#1e1e1e"   # darker border
C_SEP     = "#222222"   # separator line

C_ERR     = "#ff4444"   # error
C_WARN    = "#ffaa00"   # warning

# Camera feather-overlay colors — kept per-cam for visual identification.
# Do NOT use these for UI buttons / labels (those should be white + green).
C_CAM_FEATHER = {
    1: "#00c8ff",
    2: "#64ff64",
    3: "#ffa000",
    4: "#ff5050",
    5: "#c864ff",
}

# Shared style snippets used across panels
MONO = ("font-family:'JetBrains Mono','Consolas',monospace;"
        " font-size:10px; color:#00ff88;")

GRP  = ("QGroupBox { color:#666666; font-size:9px; border:1px solid #222222;"
        " border-radius:3px; margin-top:8px; padding-top:6px; }"
        "QGroupBox::title { subcontrol-origin:margin; left:8px; }")

# ─────────────────────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #1a1a1a;
    color: #ffffff;
}
QWidget {
    background-color: #1a1a1a;
    color: #ffffff;
    font-family: 'Rajdhani', 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    font-weight: 600;
}
/* ── 패널 ── */
QFrame#panel {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
}
/* ── 버튼 ── */
QPushButton {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 26px;
}
QPushButton:hover  { background-color: #4d4d4d; }
QPushButton:pressed { background-color: #2a2a2a; }
QPushButton#accent {
    background-color: #00aa55;
    color: #000000;
    font-weight: bold;
    border: none;
}
QPushButton#accent:hover  { background-color: #00ff88; }
QPushButton#accent:pressed { background-color: #007733; }
QPushButton#danger {
    background-color: #6b2020;
    color: #ffaaaa;
    border: 1px solid #aa4444;
}
QPushButton#danger:hover { background-color: #8b2020; }
/* ── 슬라이더 ── */
QSlider::groove:horizontal {
    background: #3d3d3d;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00ff88;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #00aa55; border-radius: 2px; }
/* ── 탭 ── */
QTabWidget::pane {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-top: none;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #aaaaaa;
    padding: 6px 16px;
    border: 1px solid #3a3a3a;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1a1a1a;
    color: #00ff88;
    border-bottom: 2px solid #00ff88;
}
QTabBar::tab:hover:!selected { background-color: #3a3a3a; color: #ffffff; }
/* ── 입력 ── */
QDoubleSpinBox, QSpinBox {
    background-color: #3d3d3d;
    color: #00ff88;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 6px;
    min-height: 22px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
    font-weight: 400;
}
QLineEdit, QComboBox {
    background-color: #3d3d3d;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 6px;
    min-height: 22px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
    border: 1px solid #00ff88;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #3d3d3d;
    color: #ffffff;
    selection-background-color: #00aa55;
}
/* ── 리스트 ── */
QListWidget {
    background-color: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
}
QListWidget::item { padding: 6px 8px; color: #cccccc; }
QListWidget::item:selected {
    background-color: #00aa55;
    color: #000000;
}
QListWidget::item:hover:!selected { background-color: #333333; }
/* ── 스크롤바 ── */
QScrollBar:vertical {
    background: #2d2d2d;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #555555;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #00ff88; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
/* ── 라벨 ── */
QLabel#section_header {
    color: #00ff88;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel#status_ok  { color: #00ff88; }
QLabel#status_err { color: #ff4444; }
QLabel#status_warn { color: #ffaa00; }
/* ── 진행바 ── */
QProgressBar {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 3px;
    height: 12px;
    text-align: center;
    color: #ffffff;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #00ff88;
    border-radius: 2px;
}
/* ── 분리선 ── */
QSplitter::handle {
    background-color: #3a3a3a;
}
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical   { height: 4px; }
/* ── 메뉴바 ── */
QMenuBar {
    background-color: #252525;
    color: #cccccc;
    border-bottom: 1px solid #3a3a3a;
    font-family: 'Rajdhani', 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    font-weight: 600;
}
QMenuBar::item:selected { background-color: #00aa55; color: #000000; }
QMenu {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3a3a3a;
}
QMenu::item:selected { background-color: #00aa55; color: #000000; }
/* ── 체크박스 ── */
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #555555;
    border-radius: 2px;
    background: #3d3d3d;
}
QCheckBox::indicator:checked { background: #00ff88; border-color: #00ff88; }
/* ── GroupBox ── */
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 4px;
    color: #aaaaaa;
    font-size: 11px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #00ff88;
}
"""
