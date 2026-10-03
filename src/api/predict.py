import numpy as np
import torch
import mlflow.pytorch
from PIL import Image

MODEL_PATH = "mlruns/1/models/m-3948c520e5ac4d57b470c886d1cde089/artifacts"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Tuberculosis",
]

# Load once at import time (i.e. once per API process)
_model = mlflow.pytorch.load_model(MODEL_PATH)
_model.eval()


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))   # HWC → CHW
    image_array = np.expand_dims(image_array, axis=0)    # add batch dim

    return image_array.astype("float32")


def predict_image(image):
    image_array = preprocess_image(image)

    tensor = torch.from_numpy(image_array)               # (1,3,224,224)

    with torch.no_grad():
        logits = _model(tensor).cpu().numpy()[0]         # (2,)

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    predicted_index = int(np.argmax(probabilities))

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "probability": float(probabilities[predicted_index]),
        "normal_probability": float(probabilities[0]),
        "tuberculosis_probability": float(probabilities[1]),
    }