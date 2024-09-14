import os
import cv2
import logging

logging.basicConfig(filename="logs/deepfake_detection.log", level=logging.INFO)

DEEPFAKE_SAVE_DIR = "detected_deepfakes"
if not os.path.exists(DEEPFAKE_SAVE_DIR):
    os.makedirs(DEEPFAKE_SAVE_DIR)

def save_deepfake_image(image, timestamp):
    filename = f"deepfake_{timestamp}.jpg"
    filepath = os.path.join(DEEPFAKE_SAVE_DIR, filename)
    if not os.path.exists(filepath):
        cv2.imwrite(filepath, image)
        logging.info(f"Deepfake image saved: {filepath}")

def log_prediction(probability, num_faces):
    logging.info(f"Deepfake Probability: {probability:.2f}, Faces Detected: {num_faces}")
