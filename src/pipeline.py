from src.logger import logger

from src.data.ingestion import ingest_data
from src.data.preprocessing import create_data_loaders

from src.models.model import create_model
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model

import torch
import mlflow
from mlflow.models import infer_signature


def main():

    logger.info("========================================")
    logger.info("Starting Medical Imaging ML Pipeline")
    logger.info("========================================")

    # 1. DATA INGESTION
    logger.info("Step 1/6: Data ingestion")
    ingest_data()

    # 2. DATA PREPROCESSING
    logger.info("Step 2/6: Data preprocessing")

    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_dir="data/raw/chest",
        batch_size=16,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        image_size=224,
        num_workers=0,
        seed=42
    )

    logger.info(f"Dataset information: {dataset_info}")

    # 3. DEVICE + MLFLOW SETUP
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "Medical-Imaging-TB-Classification"
    )

    # 4. MODEL CREATION
    logger.info("Step 3/6: Creating model")

    model = create_model(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    # 5. TRAINING + EVALUATION
    logger.info("Step 4/6: Training model")

    with mlflow.start_run():

        mlflow.log_params({
            "model": "ResNet18",
            "pretrained": True,
            "epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "image_size": 224,
            "train_size": dataset_info["train_size"],
            "validation_size": dataset_info["val_size"],
            "test_size": dataset_info["test_size"],
            "total_size": dataset_info["total_size"],
            "device": str(device)
        })

        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=1,
            learning_rate=0.001,
            class_weights=None,
            save_dir="artifacts/models"
        )

        logger.info(
            f"Training results: {training_results}"
        )

        mlflow.log_metrics({
            "train_loss": training_results["train_loss"][-1],
            "train_accuracy": training_results["train_acc"][-1],
            "val_loss": training_results["val_loss"][-1],
            "val_accuracy": training_results["val_acc"][-1]
        })

        # EVALUATION
        logger.info("Step 5/6: Evaluating model")

        class_names = [
            "Normal",
            "Tuberculosis"
        ]

        metrics, predictions, targets = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            class_names=class_names
        )

        logger.info(
            f"Evaluation metrics: {metrics}"
        )

        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"]
        })

        # MLFLOW MODEL REGISTRATION
        logger.info(
            "Registering model: TB-ResNet18"
        )

        sample_inputs, _ = next(iter(test_loader))

        sample_inputs = sample_inputs[:1].to(device)

        model.eval()

        with torch.no_grad():
            sample_outputs = model(sample_inputs)

        signature = infer_signature(
            sample_inputs.cpu().numpy(),
            sample_outputs.cpu().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="TB-ResNet18",
            serialization_format="pickle",
            input_example=sample_inputs.cpu().numpy(),
            signature=signature
        )

        logger.info(
            "Model registered successfully: TB-ResNet18"
        )

    # 6. PIPELINE COMPLETE
    logger.info("Step 6/6: Pipeline completed")

    logger.info("========================================")
    logger.info("Medical Imaging ML Pipeline Completed")
    logger.info("========================================")

    return metrics


if __name__ == "__main__":
    main()