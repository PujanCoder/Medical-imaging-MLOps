from dataclasses import dataclass, asdict

from src.automation.failure_detector import FailureEvent


@dataclass
class FailureAnalysis:
    stage: str
    failure_type: str
    severity: str
    root_cause: str
    recommended_action: str
    recoverable: bool

    def to_dict(self):
        return asdict(self)


class FailureAnalyzer:
    """
    Analyzes a FailureEvent and determines the appropriate
    recovery action.
    """

    def analyze(self, failure: FailureEvent):

        failure_type = failure.failure_type

        if failure_type == "DATA_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="Dataset or required data resource is unavailable or invalid.",
                recommended_action="RETRY_DATA_INGESTION",
                recoverable=True,
            )

        if failure_type == "PREPROCESSING_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="Dataset preprocessing or dataloader creation failed.",
                recommended_action="RETRY_PREPROCESSING",
                recoverable=True,
            )

        if failure_type == "MODEL_CREATION_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="The ML model could not be created.",
                recommended_action="RETRY_MODEL_CREATION",
                recoverable=True,
            )

        if failure_type == "TRAINING_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="The model training process failed.",
                recommended_action="RETRY_TRAINING",
                recoverable=True,
            )

        if failure_type == "RESOURCE_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="The operation exceeded available system resources.",
                recommended_action="REDUCE_RESOURCE_USAGE",
                recoverable=True,
            )

        if failure_type == "EVALUATION_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="Model evaluation failed.",
                recommended_action="RETRY_EVALUATION",
                recoverable=True,
            )

        if failure_type == "MODEL_REGISTRATION_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="MLflow model registration failed.",
                recommended_action="RETRY_MODEL_REGISTRATION",
                recoverable=True,
            )

        if failure_type == "MODEL_ARTIFACT_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="Required MLflow model artifact is missing or inaccessible.",
                recommended_action="RECOVER_MODEL_ARTIFACT",
                recoverable=True,
            )

        if failure_type == "RUNTIME_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="A runtime exception interrupted the pipeline.",
                recommended_action="RETRY_STAGE",
                recoverable=True,
            )

        if failure_type == "TIMEOUT_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="The operation exceeded the allowed execution time.",
                recommended_action="RETRY_STAGE",
                recoverable=True,
            )

        if failure_type == "CONNECTION_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="A required external service could not be reached.",
                recommended_action="RETRY_CONNECTION",
                recoverable=True,
            )

        if failure_type == "CONFIGURATION_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="Pipeline configuration is invalid.",
                recommended_action="REVIEW_CONFIGURATION",
                recoverable=False,
            )

        if failure_type == "DEPENDENCY_FAILURE":
            return FailureAnalysis(
                stage=failure.stage,
                failure_type=failure_type,
                severity=failure.severity,
                root_cause="A required Python dependency is unavailable.",
                recommended_action="INSTALL_DEPENDENCY",
                recoverable=False,
            )

        return FailureAnalysis(
            stage=failure.stage,
            failure_type=failure_type,
            severity=failure.severity,
            root_cause="The failure could not be automatically diagnosed.",
            recommended_action="ESCALATE",
            recoverable=False,
        )