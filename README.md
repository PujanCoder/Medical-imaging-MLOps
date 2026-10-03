# Medical Imaging MLOps

An end-to-end **Medical Imaging MLOps platform for Tuberculosis (TB) classification from chest X-ray images**, built with PyTorch, MLflow, FastAPI, Prometheus, and Grafana.

The project goes beyond model training by implementing **experiment tracking, model management, API serving, production metrics, and real-time monitoring**.

---

## 🚀 Project Overview

This project builds a complete machine-learning lifecycle:

```text
Chest X-ray Dataset
        │
        ▼
Data Validation / Processing
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
MLflow Model Registry
        │
        ▼
FastAPI Prediction API
        │
        ├──────────────► Prometheus
        │                    │
        │                    ▼
        │                 Grafana
        │
        ▼
Production Monitoring
```

The current system supports:

* Chest X-ray TB classification
* PyTorch-based deep learning
* MLflow experiment tracking
* MLflow Model Registry
* FastAPI model serving
* Prometheus metrics
* Grafana dashboards
* Prediction latency monitoring
* Prediction error monitoring
* Prediction distribution monitoring
* API health monitoring

---

# 🧠 Machine Learning Problem

The model classifies chest X-ray images into two classes:

```text
Normal
Tuberculosis
```

The project is designed as an MLOps system rather than only a machine-learning notebook.

The focus is on the complete lifecycle:

```text
Data → Training → Evaluation → Registry → Deployment → Monitoring
```

> **Important:** This project is intended for educational and engineering purposes. It is not a medical diagnostic system and should not be used for clinical diagnosis or treatment decisions.

---

# 🛠️ Technology Stack

| Component           | Technology   |
| ------------------- | ------------ |
| Language            | Python 3.11  |
| Deep Learning       | PyTorch      |
| Model               | ResNet18     |
| Experiment Tracking | MLflow       |
| Model Registry      | MLflow       |
| API                 | FastAPI      |
| API Server          | Uvicorn      |
| Image Processing    | Pillow       |
| Numerical Computing | NumPy        |
| Monitoring          | Prometheus   |
| Visualization       | Grafana      |
| Environment         | Conda        |
| Version Control     | Git / GitHub |
| OS                  | Windows      |

---

# 📁 Project Structure

```text
Medical-imaging-MLOps/
│
├── data/
│   ├── raw/
│   │   └── chest/
│   │       ├── Normal/
│   │       └── Tuberculosis/
│   │
│   └── processed/
│
├── models/
│
├── monitoring/
│   ├── data_drift/
│   └── model_monitor/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── predict.py
│   │
│   ├── automation/
│   │
│   ├── evaluation/
│   │
│   ├── features/
│   │
│   ├── models/
│   │
│   ├── monitoring/
│   │
│   ├── logger.py
│   ├── pipeline.py
│   └── __init__.py
│
├── mlruns/
│
├── prometheus/
│   └── prometheus.yml
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Do not commit large datasets, model binaries, MLflow artifacts, or secrets to GitHub. Use `.gitignore` and external storage where appropriate.

---

# 💻 1. Clone the Repository

Replace the URL below with your actual GitHub repository URL:

```powershell
git clone https://github.com/YOUR_USERNAME/Medical-imaging-MLOps.git
```

Enter the project:

```powershell
cd Medical-imaging-MLOps
```

---

# 🐍 2. Create the Conda Environment

Create the environment:

```powershell
conda create -n medical-mlops python=3.11 -y
```

Activate it:

```powershell
conda activate medical-mlops
```

Verify Python:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 📦 3. Install Dependencies

If the repository contains `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Important packages include:

```text
torch
torchvision
numpy
pandas
scikit-learn
Pillow
tqdm
mlflow
fastapi
uvicorn
python-multipart
prometheus-client
```

If a package is missing during execution, install it with:

```powershell
pip install PACKAGE_NAME
```

Example:

```powershell
pip install prometheus-client
```

---

# 📊 4. Dataset

The expected dataset structure is:

```text
data/
└── raw/
    └── chest/
        ├── Normal/
        └── Tuberculosis/
```

Example:

```text
data/raw/chest/
├── Normal/
│   ├── image001.jpeg
│   ├── image002.jpeg
│   └── ...
│
└── Tuberculosis/
    ├── image001.jpeg
    ├── image002.jpeg
    └── ...
```

The dataset should be available locally before running the training pipeline.

---

# 🔍 5. Verify the Dataset

Before training, verify that the expected directories exist:

```powershell
Get-ChildItem .\data\raw\chest
```

Check the classes:

```powershell
Get-ChildItem .\data\raw\chest -Directory
```

Count files:

```powershell
(Get-ChildItem .\data\raw\chest\Normal -File).Count
```

```powershell
(Get-ChildItem .\data\raw\chest\Tuberculosis -File).Count
```

---

# 🧪 6. Run the MLOps Pipeline

The main pipeline should be executed as a Python module:

```powershell
python -m src.pipeline
```

Do **not** run:

```powershell
python src/pipeline.py
```

because the project uses package imports such as:

```python
from src.logger import logger
```

Running the module with:

```powershell
python -m src.pipeline
```

ensures Python correctly resolves the `src` package.

---

# 📈 7. MLflow Experiment Tracking

Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

MLflow allows you to inspect:

* Experiments
* Runs
* Parameters
* Metrics
* Artifacts
* Models

---

# 🗂️ 8. MLflow Model Registry

The trained model can be registered through MLflow.

The local MLflow structure may look like:

```text
mlruns/
└── 1/
    └── models/
        └── <model-id>/
            └── artifacts/
```

The registered model can then be loaded by the FastAPI application.

For example:

```python
import mlflow.pytorch

model = mlflow.pytorch.load_model(MODEL_PATH)
```

---

# 🌐 9. Start the FastAPI Application

Activate the environment:

```powershell
conda activate medical-mlops
```

Start FastAPI:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open.

---

# ❤️ 10. Check API Health

Open:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "Medical Imaging TB Classification API",
  "status": "running"
}
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

# 📚 11. FastAPI Documentation

FastAPI automatically generates interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can use the Swagger UI to test:

```text
POST /predict
```

without manually writing a curl command.

---

# 🩻 12. Make a Prediction

The prediction endpoint is:

```text
POST /predict
```

It expects an uploaded image using the field:

```text
file
```

Using PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@PATH_TO_IMAGE"
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@data\raw\chest\Normal\example.jpeg"
```

The API returns information such as:

```json
{
  "filename": "example.jpeg",
  "prediction": "Normal",
  "probability": 0.98,
  "normal_probability": 0.98,
  "tuberculosis_probability": 0.02
}
```

The exact probabilities depend on the model.

---

# 📊 13. Prometheus Monitoring

The API exposes Prometheus metrics through:

```text
http://127.0.0.1:8000/metrics
```

Test it:

```powershell
curl.exe http://127.0.0.1:8000/metrics
```

The API exposes metrics including:

```text
tb_prediction_requests_total
tb_prediction_success_total
tb_prediction_errors_total
tb_prediction_latency_seconds
tb_normal_predictions_total
tb_tuberculosis_predictions_total
```

---

# ⚙️ 14. Install Prometheus

Download the Windows AMD64 version of Prometheus.

Extract it somewhere such as:

```text
C:\prometheus
```

The directory should contain:

```text
prometheus.exe
prometheus.yml
```

---

# 📝 15. Prometheus Configuration

Create/edit:

```text
prometheus.yml
```

Use:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "medical-imaging-api"
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

This tells Prometheus to collect metrics from:

```text
http://127.0.0.1:8000/metrics
```

---

# ▶️ 16. Start Prometheus

Open a new PowerShell:

```powershell
cd C:\prometheus
```

Start Prometheus:

```powershell
.\prometheus.exe --config.file=prometheus.yml
```

Keep the terminal open.

Prometheus should be available at:

```text
http://127.0.0.1:9090
```

---

# 🔎 17. Verify Prometheus Target

Open:

```text
http://127.0.0.1:9090/targets
```

You should see:

```text
medical-imaging-api
```

with:

```text
State: UP
```

This means:

```text
FastAPI → Prometheus
```

is working.

---

# 📊 18. Useful Prometheus Queries

### Total prediction requests

```promql
tb_prediction_requests_total
```

### Prediction request rate

```promql
rate(tb_prediction_requests_total[5m])
```

### Successful prediction rate

```promql
rate(tb_prediction_success_total[5m])
```

### Prediction error rate

```promql
rate(tb_prediction_errors_total[5m])
```

### Normal predictions

```promql
tb_normal_predictions_total
```

### Tuberculosis predictions

```promql
tb_tuberculosis_predictions_total
```

### Average prediction latency

```promql
rate(tb_prediction_latency_seconds_sum[5m])
/
rate(tb_prediction_latency_seconds_count[5m])
```

### P95 prediction latency

```promql
histogram_quantile(
  0.95,
  rate(tb_prediction_latency_seconds_bucket[5m])
)
```

---

# 📈 19. Grafana

Grafana is used to visualize the Prometheus metrics.

After installing Grafana, make sure its Windows service is running.

Check:

```powershell
Get-Service Grafana*
```

If stopped:

```powershell
Start-Service Grafana
```

Open:

```text
http://127.0.0.1:3000
```

---

# 🔌 20. Connect Grafana to Prometheus

Inside Grafana:

```text
Connections
    ↓
Data sources
    ↓
Add data source
    ↓
Prometheus
```

Set the Prometheus URL to:

```text
http://127.0.0.1:9090
```

Click:

```text
Save & test
```

Grafana should successfully connect to Prometheus.

---

# 📊 21. Grafana Dashboard

Recommended dashboard panels:

### Prediction Request Rate

```promql
rate(tb_prediction_requests_total[5m])
```

### Successful Predictions

```promql
rate(tb_prediction_success_total[5m])
```

### Prediction Errors

```promql
rate(tb_prediction_errors_total[5m])
```

### Normal Predictions

```promql
tb_normal_predictions_total
```

### Tuberculosis Predictions

```promql
tb_tuberculosis_predictions_total
```

### Average Prediction Latency

```promql
rate(tb_prediction_latency_seconds_sum[5m])
/
rate(tb_prediction_latency_seconds_count[5m])
```

### P95 Latency

```promql
histogram_quantile(
  0.95,
  rate(tb_prediction_latency_seconds_bucket[5m])
)
```

---

# 🏗️ MLOps Architecture

```text
                         ┌─────────────────────┐
                         │   Chest X-ray Data  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Processing     │
                         │ & Validation        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Training      │
                         │ ResNet18 / PyTorch  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Evaluation    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MLflow Tracking     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MLflow Model        │
                         │ Registry            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FastAPI             │
                         │ Model Serving       │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │   Prometheus    │       │    Prediction   │
              │   Monitoring    │       │     Results     │
              └────────┬────────┘       └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Grafana     │
              │   Dashboard     │
              └─────────────────┘
```

---

# 🔄 Complete Local Startup Process

When starting the project from a fresh computer session, use the following sequence.

## Terminal 1 — MLflow

```powershell
cd "C:\Users\<YOUR_USERNAME>\OneDrive\Documents\Medical-imaging-MLOps"
conda activate medical-mlops

mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

---

## Terminal 2 — FastAPI

```powershell
cd "C:\Users\<YOUR_USERNAME>\OneDrive\Documents\Medical-imaging-MLOps"
conda activate medical-mlops

uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

API:

```text
http://127.0.0.1:8000
```

---

## Terminal 3 — Prometheus

```powershell
cd C:\prometheus

.\prometheus.exe --config.file=prometheus.yml
```

Prometheus:

```text
http://127.0.0.1:9090
```

---

## Grafana

Check:

```powershell
Get-Service Grafana*
```

If necessary:

```powershell
Start-Service Grafana
```

Grafana:

```text
http://127.0.0.1:3000
```

---

# 🔗 Service Map

| Service    | Port | Purpose                                |
| ---------- | ---: | -------------------------------------- |
| MLflow     | 5000 | Experiment tracking / model management |
| FastAPI    | 8000 | Model serving                          |
| Prometheus | 9090 | Metrics collection                     |
| Grafana    | 3000 | Monitoring dashboard                   |

---

# 🧪 Testing the Complete System

After all services are running:

### 1. Check API

```powershell
curl.exe http://127.0.0.1:8000/health
```

### 2. Check metrics

```powershell
curl.exe http://127.0.0.1:8000/metrics
```

### 3. Check Prometheus

Open:

```text
http://127.0.0.1:9090/targets
```

Confirm:

```text
medical-imaging-api → UP
```

### 4. Make a prediction

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@PATH_TO_IMAGE"
```

### 5. Check metrics again

```powershell
curl.exe http://127.0.0.1:8000/metrics
```

Prediction counters should now increase.

### 6. Open Grafana

```text
http://127.0.0.1:3000
```

Your dashboard should display the prediction activity.

---

# 🚨 Troubleshooting

## FastAPI connection refused

Check whether Uvicorn is running:

```powershell
Get-NetTCPConnection -LocalPort 8000
```

Start it again:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## Prometheus target is DOWN

Check:

```text
http://127.0.0.1:8000/metrics
```

If that doesn't work, FastAPI isn't running correctly.

Also verify:

```yaml
targets:
  - "127.0.0.1:8000"
```

in `prometheus.yml`.

---

## Grafana refuses connection

Check:

```powershell
Get-Service Grafana*
```

Start it:

```powershell
Start-Service Grafana
```

Then:

```text
http://127.0.0.1:3000
```

---

## MLflow model loading error

Check the model path used by:

```text
src/api/predict.py
```

The model path must point to an existing MLflow model artifact.

---

## `ModuleNotFoundError`

Make sure the environment is activated:

```powershell
conda activate medical-mlops
```

Then install the missing package:

```powershell
pip install PACKAGE_NAME
```

---

# 🔐 Environment Variables

If future versions of the project use credentials, cloud storage, databases, or external APIs, store them in environment variables rather than committing secrets to GitHub.

Example:

```powershell
$env:VARIABLE_NAME="your-value"
```

Never commit:

```text
.env
credentials
API keys
passwords
cloud credentials
private certificates
```

---

# 🚀 Future MLOps Improvements

The current system provides model serving and operational monitoring. Planned improvements include:

* [ ] Automated data validation
* [ ] Data drift detection
* [ ] Prediction drift detection
* [ ] Model performance monitoring
* [ ] Automated evaluation
* [ ] Prometheus alert rules
* [ ] Grafana alerts
* [ ] Automated retraining
* [ ] Model promotion workflow
* [ ] CI/CD pipeline
* [ ] Automated model deployment
* [ ] Failure detection and recovery
* [ ] Model rollback
* [ ] ML Failure Autopilot

---

# 🎯 Project Goal

The goal of this project is to demonstrate how a machine-learning model can move from experimentation to a monitored production-style system.

Instead of stopping at:

```text
Dataset
   ↓
Train Model
   ↓
Accuracy
```

the project implements:

```text
Dataset
   ↓
Training
   ↓
Evaluation
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
API Deployment
   ↓
Metrics
   ↓
Monitoring
   ↓
Alerts
   ↓
Automated Recovery
```

---

# 👨‍💻 Author

**Pujan Pandey**

Machine Learning / MLOps Developer

Interests:

* Machine Learning
* MLOps
* Deep Learning
* Model Deployment
* Observability
* AI Infrastructure
* Quantum Computing

---

# ⭐ If You Find This Project Useful

Feel free to explore the repository, experiment with the pipeline, and extend the monitoring and automation components.

```text
Machine Learning + Engineering + Automation = MLOps
```
