import winsound
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QProgressBar, QSystemTrayIcon, QMenu, QAction, QApplication
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import QThread, QTimer
from detection.deepfake_thread import DeepfakeDetectionThread
from ui.deepfake_overlay import DeepfakeOverlay
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import QThread, Qt, QTimer, QRect

class DeepfakeDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Big Hackobowl - Deepfake Detector")  # Change window title
        self.setWindowIcon(QIcon("logo.png"))
        self.init_ui()

        # Load the two icons (normal and alert)
        self.normal_icon = QIcon("logo.png")
        self.alert_icon = QIcon("logo_alert.png")  # Load a red icon for deepfake alert

        # Set up system tray
        self.tray_icon = QSystemTrayIcon(self.normal_icon, self)
        self.tray_icon.setToolTip("Deepfake Detector")

        # Set up system tray menu
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        quit_action = QAction("Quit", self)
        show_action.triggered.connect(self.show)
        quit_action.triggered.connect(self.quit_application)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Keep the window always on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Timer for flashing icons and logo
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_icons)

        self.is_flashing = False  # Flag to track icon flashing state

        # Timer for deepfake alert timeout
        self.timeout_timer = QTimer()
        self.timeout_timer.timeout.connect(self.reset_to_initial_state)
        self.timeout_duration = 500

        self.overlay = DeepfakeOverlay()  # Transparent overlay

    def init_ui(self):
        layout = QVBoxLayout()

        # Add logo and center it
        self.logo_label = QLabel(self)
        pixmap = QPixmap("logo.png").scaled(150, 150, Qt.KeepAspectRatio)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)  # Center the logo
        layout.addWidget(self.logo_label)

        # Capture and Stop Capture Buttons
        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # Add FPS and Deepfake Probability Progress Bar
        self.probability_bar = QProgressBar()
        self.probability_bar.setMaximum(100)
        layout.addWidget(QLabel("Deepfake Probability:"))
        layout.addWidget(self.probability_bar)

        self.fps_bar = QProgressBar()
        self.fps_bar.setMaximum(60)  # Assuming max 60 FPS
        layout.addWidget(QLabel("FPS:"))
        layout.addWidget(self.fps_bar)

        # Font customization for the info label
        self.info_label = QLabel(
            "Deepfake Probability: 0.00, FPS: 0.00, Faces Detected: 0"
        )
        self.info_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.info_label)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.error_label)

        self.setLayout(layout)

    def toggle_icons(self):
        """Alternate between normal and alert icons in both system tray and logo"""
        if self.is_flashing:
            self.tray_icon.setIcon(self.normal_icon)
            pixmap = QPixmap("logo.png").scaled(150, 150, Qt.KeepAspectRatio)
            self.logo_label.setPixmap(pixmap)
        else:
            self.tray_icon.setIcon(self.alert_icon)
            pixmap = QPixmap("logo_alert.png").scaled(150, 150, Qt.KeepAspectRatio)
            self.logo_label.setPixmap(pixmap)
        self.is_flashing = not self.is_flashing

    def start_detection(self):
        # Set up the detection thread
        self.detection_thread = QThread()
        self.detection_worker = DeepfakeDetectionThread()
        self.detection_worker.moveToThread(self.detection_thread)
        self.detection_worker.deepfake_detected.connect(self.handle_deepfake_detection)
        self.detection_worker.info_updated.connect(self.update_info)
        self.detection_thread.started.connect(self.detection_worker.run)
        self.detection_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_detection(self):
        if hasattr(self, "detection_worker"):
            self.detection_worker.stop()
            self.detection_thread.quit()
            self.detection_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_to_initial_state()

    def handle_deepfake_detection(self, probability, bbox):
        """Handle deepfake detection logic, start flashing icons and play sound"""
        if probability > 0.7:  # Threshold for detecting a deepfake
            self.error_label.setText(
                f"Deepfake detected with probability {probability:.2f}!"
            )
            self.overlay.set_bbox(bbox)  # Show the red rectangle on the overlay

            # Start flashing the icons
            self.flash_timer.start(500)  # Flash every 500ms

            # Play subtle, lighter alert sound
            frequency = 1200  # Set Frequency to 1200 Hertz for a softer sound
            duration = 400  # Set Duration to 400 ms for a shorter alert
            winsound.Beep(frequency, duration)

            # Reset the timeout timer every time a deepfake is detected
            self.timeout_timer.start(self.timeout_duration)

        else:
            # If no deepfake detected, start the timeout timer for resetting the UI
            self.timeout_timer.start(self.timeout_duration)

    def update_info(self, probability, fps, num_faces, bbox):
        """Update probability, fps, and number of faces"""
        self.info_label.setText(
            f"Deepfake Probability: {probability:.2f}, FPS: {fps:.2f}, Faces Detected: {num_faces}"
        )
        self.probability_bar.setValue(int(probability * 100))
        self.fps_bar.setValue(int(fps))

    def reset_to_initial_state(self):
        """Resets the UI and icons to their initial state"""
        self.error_label.clear()  # Clear error label
        self.overlay.set_bbox(QRect())  # Clear the overlay bbox
        if self.flash_timer.isActive():
            self.flash_timer.stop()  # Stop flashing
        self.is_flashing = False  # Reset flashing state
        self.tray_icon.setIcon(self.normal_icon)
        pixmap = QPixmap("logo.png").scaled(150, 150, Qt.KeepAspectRatio)
        self.logo_label.setPixmap(pixmap)

    def quit_application(self):
        self.tray_icon.hide()
        QApplication.quit()
