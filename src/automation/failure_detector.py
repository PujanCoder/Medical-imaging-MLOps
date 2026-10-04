from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import traceback


@dataclass
class FailureEvent:
    stage: str
    failure_type: str
    severity: str
    message: str
    exception_type: str
    recoverable: bool
    timestamp: str
    traceback_text: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class FailureDetector:
    """
    Detects and classifies failures occurring in the MLOps pipeline.
    """

    FAILURE_RULES = {
        "FileNotFoundError": ("DATA_FAILURE", "high", True),
        "PermissionError": ("FILE_PERMISSION_FAILURE", "high", False),
        "ValueError": ("CONFIGURATION_FAILURE", "medium", True),
        "RuntimeError": ("RUNTIME_FAILURE", "high", True),
        "MemoryError": ("RESOURCE_FAILURE", "high", True),
        "TimeoutError": ("TIMEOUT_FAILURE", "medium", True),
        "ConnectionError": ("CONNECTION_FAILURE", "high", True),
        "KeyError": ("CONFIGURATION_FAILURE", "medium", True),
        "ImportError": ("DEPENDENCY_FAILURE", "high", False),
        "ModuleNotFoundError": ("DEPENDENCY_FAILURE", "high", False),
    }

    def detect(self, stage: str, exception: Exception) -> FailureEvent:

        exception_name = type(exception).__name__
        message = str(exception)

        failure_type, severity, recoverable = self._classify_exception(
            stage,
            exception_name,
            message,
        )

        return FailureEvent(
            stage=stage,
            failure_type=failure_type,
            severity=severity,
            message=message,
            exception_type=exception_name,
            recoverable=recoverable,
            timestamp=datetime.now(timezone.utc).isoformat(),
            traceback_text=traceback.format_exc(),
        )

    def _classify_exception(
        self,
        stage: str,
        exception_name: str,
        message: str,
    ):

        message_lower = message.lower()

        # Stage-specific classification
        if stage == "data_ingestion":
            return "DATA_FAILURE", "high", True

        if stage == "preprocessing":
            return "PREPROCESSING_FAILURE", "high", True

        if stage == "model_creation":
            return "MODEL_CREATION_FAILURE", "high", True

        if stage == "training":
            if any(
                keyword in message_lower
                for keyword in [
                    "out of memory",
                    "cuda out of memory",
                    "memory"
                ]
            ):
                return "RESOURCE_FAILURE", "high", True

            return "TRAINING_FAILURE", "high", True

        if stage == "evaluation":
            return "EVALUATION_FAILURE", "high", True

        if stage == "model_registration":
            if any(
                keyword in message_lower
                for keyword in [
                    "artifact",
                    "model artifact",
                    "mlflow",
                    "model not found"
                ]
            ):
                return "MODEL_ARTIFACT_FAILURE", "high", True

            return "MODEL_REGISTRATION_FAILURE", "high", True

        # Generic classification
        if any(
            keyword in message_lower
            for keyword in [
                "dataset",
                "image",
                "file not found",
                "no such file",
                "directory"
            ]
        ):
            return "DATA_FAILURE", "high", True

        if any(
            keyword in message_lower
            for keyword in [
                "mlflow",
                "artifact",
                "model artifact",
                "model not found"
            ]
        ):
            return "MODEL_ARTIFACT_FAILURE", "high", True

        if any(
            keyword in message_lower
            for keyword in [
                "out of memory",
                "cuda out of memory",
                "memory"
            ]
        ):
            return "RESOURCE_FAILURE", "high", True

        if exception_name in self.FAILURE_RULES:
            return self.FAILURE_RULES[exception_name]

        return "UNKNOWN_FAILURE", "high", False