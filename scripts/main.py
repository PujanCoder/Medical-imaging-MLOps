from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from src.api.predict import preprocess_image
import requests
import numpy as np


app = FastAPI(
    title="Medical Imaging TB Classification API",
    description="API for Tuberculosis classification from chest X-ray images",
    version="1.0.0"
)


MLFLOW_URL = "http://127.0.0.1:5001/invocations"

CLASS_NAMES = [
    "Normal",
    "Tuberculosis"
]


@app.get("/")
def root():
    return {
        "message": "Medical Imaging TB Classification API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Save temporarily in memory
        image_array = np.array(image).astype("float32") / 255.0

        # Resize
        image = image.resize((224, 224))

        image_array = np.array(image).astype("float32") / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image_array = (
            image_array - mean
        ) / std

        # HWC → CHW
        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        image_array = image_array.astype("float32")

        # Send to MLflow
        response = requests.post(
            MLFLOW_URL,
            json={
                "inputs": image_array.tolist()
            },
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        # Get model logits
        logits = np.array(
            result["predictions"][0]
        )

        # Softmax
        exp_logits = np.exp(
            logits - np.max(logits)
        )

        probabilities = (
            exp_logits /
            exp_logits.sum()
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        prediction = CLASS_NAMES[
            predicted_index
        ]

        probability = float(
            probabilities[predicted_index]
        )

        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": round(
                probability,
                4
            ),
            "normal_probability": round(
                float(probabilities[0]),
                4
            ),
            "tuberculosis_probability": round(
                float(probabilities[1]),
                4
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )