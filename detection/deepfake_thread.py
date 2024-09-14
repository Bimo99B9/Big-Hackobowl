import logging
import time
import cv2
import numpy as np
import mss
import torch
import face_recognition
from PyQt5.QtCore import QObject, pyqtSignal, QRect
from detection.deepfake_utils import save_deepfake_image, log_prediction # Load deepfake detection model

from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
model.eval()

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
