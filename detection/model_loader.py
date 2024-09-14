from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_model():
    processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model.eval()
    return processor, model
