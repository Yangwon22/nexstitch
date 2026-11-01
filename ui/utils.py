import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """RGB uint8 ndarray → QPixmap"""
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def fit_pixmap(pixmap: QPixmap, max_w: int, max_h: int) -> QPixmap:
    return pixmap.scaled(max_w, max_h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


def gray_to_pixmap(gray: np.ndarray) -> QPixmap:
    rgb = np.stack([gray, gray, gray], axis=2)
    return numpy_to_pixmap(rgb)
