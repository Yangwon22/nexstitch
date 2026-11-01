import sys
from PyQt6.QtWidgets import QApplication, QDialog

from ui.style import DARK_STYLE
from ui.fonts import load_fonts


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NexStitch")

    load_fonts()
    app.setStyleSheet(DARK_STYLE)

    # ── Project Manager ─────────────────────────────────────────────────────
    # Imported here (not at top-level) so that core.settings is NOT yet imported
    # by any UI module.  After the user selects a project, project.apply_to_settings()
    # patches core.settings / core.calibration / core.video in-place, THEN we
    # import MainWindow — at which point all `from core.settings import X` bindings
    # in source_panel.py / stitch_canvas.py pick up the project-specific paths.
    from ui.project_manager import ProjectManagerDialog
    pm = ProjectManagerDialog()
    if pm.exec() != QDialog.DialogCode.Accepted or pm.selected_project is None:
        sys.exit(0)

    project = pm.selected_project
    project.apply_to_settings()

    # ── Main window ─────────────────────────────────────────────────────────
    from ui.main_window import MainWindow
    window = MainWindow(project=project)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
