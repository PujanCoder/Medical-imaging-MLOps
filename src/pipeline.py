from src.logger import logger

from src.data.ingestion import ingest_data
from src.data.preprocessing import create_data_loaders

from src.models.model import create_model
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model

from src.automation.autopilot import FailureAutopilot

import torch
import mlflow
from mlflow.models import infer_signature


def main():

    logger.info("========================================")
    logger.info("Starting Medical Imaging ML Pipeline")
    logger.info("========================================")

    autopilot = FailureAutopilot()

    # ==========================================================
    # 1. DATA INGESTION
    # ==========================================================

    logger.info("Step 1/6: Data ingestion")

    autopilot.run_stage(
        stage_name="data_ingestion",
        stage_function=ingest_data,
        recovery_functions={
            "RETRY_DATA_INGESTION": ingest_data,
        },
    )

    # ==========================================================
    # 2. DATA PREPROCESSING
    # ==========================================================

    logger.info("Step 2/6: Data preprocessing")

    def preprocess():

        return create_data_loaders(
            data_dir="data/raw/chest",
            batch_size=16,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            image_size=224,
            num_workers=0,
            seed=42,
        )

    loaders = autopilot.run_stage(
        stage_name="preprocessing",
        stage_function=preprocess,
        recovery_functions={
            "RETRY_PREPROCESSING": preprocess,
        },
    )

    (
        train_loader,
        val_loader,
        test_loader,
        dataset_info,
    ) = loaders

    logger.info(
        f"Dataset information: {dataset_info}"
    )

    # ==========================================================
    # 3. DEVICE + MLFLOW SETUP
    # ==========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(
        f"Using device: {device}"
    )

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "Medical-Imaging-TB-Classification"
    )

    # ==========================================================
    # 4. MODEL CREATION
    # ==========================================================

    logger.info("Step 3/6: Creating model")

    def build_model():

        model = create_model(
            num_classes=2,
            pretrained=True,
        )

        return model.to(device)

    model = autopilot.run_stage(
        stage_name="model_creation",
        stage_function=build_model,
        recovery_functions={
            "RETRY_MODEL_CREATION": build_model,
        },
    )

    # ==========================================================
    # 5. TRAINING + EVALUATION + REGISTRATION
    # ==========================================================

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
            "device": str(device),
        })

        # ------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------

        def train():

            return train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=1,
                learning_rate=0.001,
                class_weights=None,
                save_dir="artifacts/models",
            )

        training_results = autopilot.run_stage(
            stage_name="training",
            stage_function=train,
            recovery_functions={
                "RETRY_TRAINING": train,
                "RETRY_STAGE": train,
            },
        )

        logger.info(
            f"Training results: {training_results}"
        )

        mlflow.log_metrics({
            "train_loss": training_results["train_loss"][-1],
            "train_accuracy": training_results["train_acc"][-1],
            "val_loss": training_results["val_loss"][-1],
            "val_accuracy": training_results["val_acc"][-1],
        })

        # ------------------------------------------------------
        # EVALUATION
        # ------------------------------------------------------

        logger.info(
            "Step 5/6: Evaluating model"
        )

        class_names = [
            "Normal",
            "Tuberculosis",
        ]

        def evaluate():

            return evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                class_names=class_names,
            )

        evaluation_result = autopilot.run_stage(
            stage_name="evaluation",
            stage_function=evaluate,
            recovery_functions={
                "RETRY_EVALUATION": evaluate,
            },
        )

        metrics, predictions, targets = evaluation_result

        logger.info(
            f"Evaluation metrics: {metrics}"
        )

        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
        })

        # ------------------------------------------------------
        # MODEL REGISTRATION
        # ------------------------------------------------------

        logger.info(
            "Step 6/6: Registering model: TB-ResNet18"
        )

        def register_model():

            sample_inputs, _ = next(
                iter(test_loader)
            )

            sample_inputs = sample_inputs[:1].to(device)

            model.eval()

            with torch.no_grad():

                sample_outputs = model(
                    sample_inputs
                )

            signature = infer_signature(
                sample_inputs.cpu().numpy(),
                sample_outputs.cpu().numpy(),
            )

            mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name="TB-ResNet18",
                serialization_format="pickle",
                input_example=sample_inputs.cpu().numpy(),
                signature=signature,
            )

            return {
                "success": True,
                "message": "Model registered successfully.",
            }

        autopilot.run_stage(
            stage_name="model_registration",
            stage_function=register_model,
            recovery_functions={
                "RETRY_MODEL_REGISTRATION": register_model,
            },
        )

        logger.info(
            "Model registered successfully: TB-ResNet18"
        )

    # ==========================================================
    # PIPELINE COMPLETE
    # ==========================================================

    logger.info(
        "Medical Imaging ML Pipeline Completed"
    )

    logger.info("========================================")

    return metrics


if __name__ == "__main__":
    main()