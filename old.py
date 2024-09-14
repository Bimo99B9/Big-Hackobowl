import sys
import time
import logging
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QProgressBar,
    QSystemTrayIcon,
    QMenu,
    QAction,
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter, QColor, QPen
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer, QRect
import mss
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import face_recognition
import winsound  # For alert sound

# Setup logging
logging.basicConfig(filename="deepfake_detection.log", level=logging.INFO)

# Directory to save detected deepfakes
DEEPFAKE_SAVE_DIR = "detected_deepfakes"

# Ensure the deepfake folder exists
if not os.path.exists(DEEPFAKE_SAVE_DIR):
    os.makedirs(DEEPFAKE_SAVE_DIR)

# Load deepfake detection model
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
model.eval()


# Function to save the detected deepfake image if not already saved
def save_deepfake_image(image, timestamp):
    filename = f"deepfake_{timestamp}.jpg"
    filepath = os.path.join(DEEPFAKE_SAVE_DIR, filename)

    # Avoid duplicate saves by checking if the file already exists
    if not os.path.exists(filepath):
        cv2.imwrite(filepath, image)
        logging.info(f"Deepfake image saved: {filepath}")


# Detection Thread
class DeepfakeDetectionThread(QObject):
    info_updated = pyqtSignal(
        float, float, int, QRect
    )  # Deepfake probability, FPS, num_faces, face bounding box
    finished = pyqtSignal()
    deepfake_detected = pyqtSignal(
        float, QRect
    )  # Send probability and face bounding box for further processing

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Captures the entire primary screen
                while self._running:
                    start_time = time.perf_counter()
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    deepfake_probability = 0.0
                    num_faces = 0
                    face_bbox = QRect()

                    # Face Detection using face_recognition
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(
                        frame_rgb, model="hog"
                    )
                    num_faces = len(face_locations)

                    if num_faces > 0:
                        face_areas = [
                            (top, right, bottom, left, (bottom - top) * (right - left))
                            for (top, right, bottom, left) in face_locations
                        ]
                        face_areas.sort(key=lambda x: x[4], reverse=True)
                        top, right, bottom, left, _ = face_areas[0]

                        # Add padding to the face bounding box
                        height = bottom - top
                        width = right - left
                        padding_h = int(height * 0.6)
                        padding_w = int(width * 0.6)

                        # Ensure coordinates are within frame bounds
                        top_padded = max(0, top - padding_h)
                        left_padded = max(0, left - padding_w)
                        bottom_padded = min(frame.shape[0], bottom + padding_h)
                        right_padded = min(frame.shape[1], right + padding_w)

                        # Extract face image with padding
                        face_img = frame[
                            top_padded:bottom_padded, left_padded:right_padded
                        ]

                        # Preprocess the face for deepfake detection
                        face_img_resized = cv2.resize(face_img, (224, 224))
                        inputs = processor(images=face_img_resized, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model(**inputs)
                            probabilities = torch.softmax(outputs.logits, dim=1)
                            deepfake_probability = probabilities[0][1].item()

                        if deepfake_probability > 0.5:
                            timestamp = int(time.time())
                            save_deepfake_image(face_img, timestamp)
                            face_bbox = QRect(
                                left_padded,
                                top_padded,
                                right_padded - left_padded,
                                bottom_padded - top_padded,
                            )
                            # Emit signal to notify about deepfake detection
                            self.deepfake_detected.emit(deepfake_probability, face_bbox)

                        log_prediction(deepfake_probability, num_faces)

                    # Frame processing time and FPS
                    end_time = time.perf_counter()
                    fps = (
                        1 / (end_time - start_time)
                        if (end_time - start_time) > 0
                        else float("inf")
                    )
                    self.info_updated.emit(
                        deepfake_probability, fps, num_faces, face_bbox
                    )

        except Exception as e:
            logging.error(f"An error occurred in detection thread: {e}")
            print(f"An error occurred in detection thread: {e}")
        finally:
            self.finished.emit()


# Function to log predictions
def log_prediction(probability, num_faces):
    logging.info(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Deepfake Probability: {probability:.2f}, Faces Detected: {num_faces}"
    )


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


# Main Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Prevent quitting when the window is closed
    gui = DeepfakeDetectorGUI()
    gui.show()
    sys.exit(app.exec_())
