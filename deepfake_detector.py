import sys
from PyQt5.QtWidgets import QApplication
from ui.deepfake_gui import DeepfakeDetectorGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Prevent quitting when the window is closed
    gui = DeepfakeDetectorGUI()
    gui.show()
    sys.exit(app.exec_())
