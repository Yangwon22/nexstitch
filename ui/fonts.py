from pathlib import Path
from PyQt6.QtGui import QFontDatabase, QFont
from PyQt6.QtWidgets import QApplication

_FONT_DIR = Path(__file__).parent.parent / "assets" / "fonts"

FONT_UI    = "Rajdhani"
FONT_MONO  = "JetBrains Mono"
FONT_FALLBACK_UI   = "Segoe UI"
FONT_FALLBACK_MONO = "Consolas"


def load_fonts() -> tuple[str, str]:
    """
    Load Rajdhani and JetBrains Mono from assets/fonts/.
    Returns (ui_family, mono_family) — falls back to system fonts on failure.
    """
    ui_family   = FONT_FALLBACK_UI
    mono_family = FONT_FALLBACK_MONO

    raj_path = _FONT_DIR / "Rajdhani-Bold.ttf"
    if raj_path.exists():
        fid = QFontDatabase.addApplicationFont(str(raj_path))
        if fid != -1:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                ui_family = families[0]

    jb_path = _FONT_DIR / "JetBrainsMono-Regular.ttf"
    if jb_path.exists():
        fid = QFontDatabase.addApplicationFont(str(jb_path))
        if fid != -1:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                mono_family = families[0]

    return ui_family, mono_family


def make_ui_font(size: int = 11, bold: bool = True) -> QFont:
    f = QFont(FONT_UI)
    f.setPointSize(size)
    f.setBold(bold)
    return f


def make_mono_font(size: int = 10) -> QFont:
    f = QFont(FONT_MONO)
    f.setPointSize(size)
    return f
