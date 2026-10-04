import pytest

from src.automation.failure_detector import FailureDetector
from src.automation.failure_analyzer import FailureAnalyzer
from src.automation.recovery_engine import RecoveryEngine


def test_failure_detection():
    detector = FailureDetector()

    try:
        raise FileNotFoundError("Dataset directory was not found")

    except Exception as error:
        failure = detector.detect(
            stage="data_ingestion",
            exception=error,
        )

    assert failure.failure_type == "DATA_FAILURE"
    assert failure.stage == "data_ingestion"
    assert failure.recoverable is True


def test_failure_analysis():
    detector = FailureDetector()
    analyzer = FailureAnalyzer()

    try:
        raise FileNotFoundError("Dataset directory was not found")

    except Exception as error:
        failure = detector.detect(
            stage="data_ingestion",
            exception=error,
        )

    analysis = analyzer.analyze(failure)

    assert analysis.failure_type == "DATA_FAILURE"
    assert analysis.recommended_action == "RETRY_DATA_INGESTION"
    assert analysis.recoverable is True


def test_successful_recovery():
    detector = FailureDetector()
    analyzer = FailureAnalyzer()
    engine = RecoveryEngine()

    try:
        raise RuntimeError("Training failed")

    except Exception as error:
        failure = detector.detect(
            stage="training",
            exception=error,
        )

    # Force the detected runtime failure to represent
    # a training failure for this controlled test.
    failure.failure_type = "TRAINING_FAILURE"

    analysis = analyzer.analyze(failure)

    recovery_called = {"count": 0}

    def fake_training():
        recovery_called["count"] += 1

        return {
            "success": True,
            "message": "Training completed successfully.",
        }

    result = engine.recover(
        failure=failure,
        analysis=analysis,
        recovery_functions={
            "RETRY_TRAINING": fake_training,
        },
    )

    assert result["success"] is True
    assert result["action"] == "RETRY_TRAINING"
    assert recovery_called["count"] == 1


def test_failed_recovery():
    detector = FailureDetector()
    analyzer = FailureAnalyzer()
    engine = RecoveryEngine()

    try:
        raise RuntimeError("Training failed")

    except Exception as error:
        failure = detector.detect(
            stage="training",
            exception=error,
        )

    failure.failure_type = "TRAINING_FAILURE"

    analysis = analyzer.analyze(failure)

    def failed_training():
        return {
            "success": False,
            "message": "Training still failed.",
        }

    result = engine.recover(
        failure=failure,
        analysis=analysis,
        recovery_functions={
            "RETRY_TRAINING": failed_training,
        },
    )

    assert result["success"] is False
    assert result["action"] == "RETRY_TRAINING"
    assert result["attempts"] == 3