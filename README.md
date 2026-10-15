# Medical Imaging MLOps

An end-to-end **MLOps platform for medical image classification**, designed to automate the complete machine learning lifecycle — from dataset validation and model training to deployment, monitoring, failure detection, automated recovery, and model quality control.

The project uses a **Tuberculosis chest X-ray classification** workflow to demonstrate production-oriented MLOps practices.
# 🛠️ Tech Stack



### Machine Learning



![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)



![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)



![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-ee4c2c?logo=pytorch)



![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)



![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)



### MLOps



![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)



![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)



![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)



![Grafana](https://img.shields.io/badge/Grafana-Observability-F46800?logo=grafana)



### Image Processing



![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB?logo=python)



### Development



![Conda](https://img.shields.io/badge/Conda-Environment-44A833?logo=anaconda)



![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)



![GitHub](https://img.shields.io/badge/GitHub-Code%20Hosting-181717?logo=github)



### License



![License](https://img.shields.io/badge/License-MIT-green)
## 🚀 MLOps Pipeline

```text
                    ┌─────────────────┐
                    │     Dataset     │
                    │ Validation      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Preprocessing   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Training  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Evaluation    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Experiment      │
                    │ Tracking MLflow │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Model Registry │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ API Deployment  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Monitoring    │
                    │ Prometheus      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Failure         │
                    │ Detection      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Failure         │
                    │ Autopilot       │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Automated       │
                    │ Recovery        │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Retraining    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Quality   │
                    │     Gate        │
                    └───────┬─┬───────┘
                            ↓ ↓
                         PASS FAIL
                          ↓     ↓
                    Register   Reject
                     Model     Model
```

## 🧠 Key Features

### Machine Learning

* Chest X-ray image classification
* ResNet18-based deep learning model
* Automated dataset validation
* Reproducible train/validation/test splitting
* Model evaluation with classification metrics

### MLOps

* MLflow experiment tracking
* MLflow Model Registry
* Model versioning
* Automated model evaluation
* Reproducible training pipeline
* Model artifact management

### API & Deployment

* FastAPI inference API
* Health and metrics endpoints
* Production-oriented model serving
* Container-ready architecture

### Observability

* Prometheus metrics
* Model and API monitoring
* Data drift monitoring
* Performance monitoring
* Failure detection and logging

### 🤖 ML Failure Autopilot

The project includes an automated failure-recovery architecture that can:

```text
Failure
   ↓
Detect
   ↓
Analyze
   ↓
Select Recovery Strategy
   ↓
Execute Recovery
   ↓
Retry
   ↓
Validate
   ↓
Success / Escalate
```

The system classifies failures such as:

* Data failures
* Preprocessing failures
* Training failures
* Runtime failures
* Resource failures
* Evaluation failures
* MLflow/model artifact failures
* Connection failures
* Configuration failures

This allows the pipeline to respond to recoverable failures without requiring every failure to be manually investigated.

## 🔄 Automated Retraining

When model performance or data conditions require retraining, the system can execute a separate retraining workflow:

```text
Trigger
   ↓
Dataset Verification
   ↓
Preprocessing
   ↓
New Model
   ↓
Retraining
   ↓
Evaluation
   ↓
Quality Gate
```

## 🛡️ Model Quality Gate

A retrained model is **not automatically promoted** simply because training succeeded.

The Model Quality Gate compares the candidate model against the current baseline using metrics such as:

* Accuracy
* F1 score
* Tuberculosis recall
* Maximum allowed performance degradation

```text
Candidate Model
       ↓
   Quality Gate
     ↙     ↘
  PASS     FAIL
   ↓         ↓
Register   Reject
Promote    Candidate
```

This prevents a poorly performing retrained model from automatically replacing a better production model.

> The quality thresholds in this project are engineering/demo thresholds and do not represent clinical validation or medical-use approval.

## 📊 Observability Stack

| Component             | Purpose                                  |
| --------------------- | ---------------------------------------- |
| **MLflow**            | Experiment tracking and model registry   |
| **FastAPI**           | Model serving and inference API          |
| **Prometheus**        | Metrics collection                       |
| **Monitoring**        | System and model observability           |
| **Data Drift**        | Detect changes in incoming data          |
| **Failure Autopilot** | Automated failure detection and recovery |
| **Quality Gate**      | Candidate model validation               |

## 🧪 Testing

The project includes unit and integration tests covering:

* Failure detection
* Failure analysis
* Recovery policies
* Recovery engine
* Autopilot recovery flow
* Model quality gate
* API endpoints

Run the complete test suite with:

```bash
pytest -v
```

## 📁 Project Structure

```text
Medical-imaging-MLOps/
│
├── src/
│   ├── api/
│   ├── automation/
│   │   ├── autopilot.py
│   │   ├── failure_detector.py
│   │   ├── failure_analyzer.py
│   │   ├── recovery_engine.py
│   │   ├── recovery_policies.py
│   │   ├── recovery_validator.py
│   │   ├── retraining.py
│   │   └── model_quality_gate.py
│   │
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── logger.py
│   └── pipeline.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
├── artifacts/
├── monitoring/
├── mlruns/
├── pyproject.toml
└── README.md
```

## ▶️ Running the Pipeline

Activate the project environment:

```bash
conda activate medical-mlops
```

Run the main ML pipeline:

```bash
python -m src.pipeline
```

Run automated retraining:

```bash
python -m src.automation.retraining
```

Run tests:

```bash
pytest -v
```

## 🎯 Project Objective

The goal of this project is to demonstrate how a machine learning system can move beyond a simple **train → evaluate → deploy** workflow toward a more automated and resilient MLOps architecture.

The system is designed around four major principles:

```text
Automation
    +
Observability
    +
Reliability
    +
Controlled Model Promotion
```

Rather than treating model training as the end of the ML lifecycle, the project focuses on what happens **after deployment** — monitoring the system, detecting failures, recovering from operational problems, retraining when necessary, and preventing degraded models from being promoted automatically.

---

# 👨‍💻 Author

**Pujan Pandey**

Machine Learning / MLOps Developer

### Interests

* Machine Learning
* MLOps
* Deep Learning
* Model Deployment
* Observability
* AI Infrastructure
* Quantum Computing

---

# ⭐ If You Find This Project Useful

If you find this project useful for learning or building MLOps systems, consider giving the repository a **star**.

Feedback, ideas, and improvements are always welcome.

Feel free to explore the repository, experiment with the pipeline, and extend the monitoring and automation components.

```text
Machine Learning + Engineering + Automation = MLOps
```
