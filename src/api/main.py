from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from src.api.predict import predict_image


app = FastAPI(
    title="Medical Imaging TB Classification API",
    description="TB classification API using ResNet18 and MLflow",
    version="1.0.0"
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

PREDICTION_REQUESTS = Counter(
    "tb_prediction_requests_total",
    "Total number of prediction requests"
)

PREDICTION_SUCCESS = Counter(
    "tb_prediction_success_total",
    "Total number of successful predictions"
)

PREDICTION_ERRORS = Counter(
    "tb_prediction_errors_total",
    "Total number of failed predictions"
)

PREDICTION_LATENCY = Histogram(
    "tb_prediction_latency_seconds",
    "Prediction latency in seconds"
)

NORMAL_PREDICTIONS = Counter(
    "tb_normal_predictions_total",
    "Total number of Normal predictions"
)

TB_PREDICTIONS = Counter(
    "tb_tuberculosis_predictions_total",
    "Total number of Tuberculosis predictions"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Medical Imaging TB Classification API",
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    PREDICTION_REQUESTS.inc()

    start_time = time.time()

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        result = predict_image(image)

        prediction = result["prediction"]

        if prediction == "Normal":

            NORMAL_PREDICTIONS.inc()

        elif prediction == "Tuberculosis":

            TB_PREDICTIONS.inc()

        PREDICTION_SUCCESS.inc()

        return {
            "filename": file.filename,
            **{
                key: round(value, 4)
                if isinstance(value, float)
                else value
                for key, value in result.items()
            }
        }

    except Exception as e:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        latency = time.time() - start_time

        PREDICTION_LATENCY.observe(latency)