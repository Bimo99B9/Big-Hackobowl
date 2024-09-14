from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QDesktopWidget

class DeepfakeOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setWindowOpacity(0.8)

        # Get the screen geometry without the taskbar (work area)
        screen_geometry = QDesktopWidget().availableGeometry()
        self.setGeometry(0, 0, screen_geometry.width(), screen_geometry.height())
        self.showFullScreen()
        self.bbox = QRect()

    def set_bbox(self, bbox):
        """Set the bounding box of detected deepfake"""
        self.bbox = bbox
        self.update()

    def paintEvent(self, event):
        if not self.bbox.isNull():
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 30, Qt.SolidLine)
            painter.setPen(pen)
            # Only draw inside the bounding box to avoid unwanted shadow effects
            painter.setClipRect(self.bbox)
            painter.drawRect(self.bbox)
