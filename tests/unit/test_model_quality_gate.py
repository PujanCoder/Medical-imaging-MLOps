from src.automation.model_quality_gate import ModelQualityGate


def test_bad_retrained_model_is_rejected():

    baseline = {
        "accuracy": 0.9730,
        "f1": 0.9727,
        "classification_report": {
            "Tuberculosis": {
                "recall": 0.8909
            }
        }
    }

    candidate = {
        "accuracy": 0.8365,
        "f1": 0.7719,
        "classification_report": {
            "Tuberculosis": {
                "recall": 0.0636
            }
        }
    }

    gate = ModelQualityGate()

    result = gate.evaluate(
        candidate_metrics=candidate,
        baseline_metrics=baseline,
    )

    assert result.passed is False
    assert "Tuberculosis recall" in result.reason


def test_good_candidate_model_is_accepted():

    baseline = {
        "accuracy": 0.9730,
        "f1": 0.9727,
        "classification_report": {
            "Tuberculosis": {
                "recall": 0.8909
            }
        }
    }

    candidate = {
        "accuracy": 0.9750,
        "f1": 0.9740,
        "classification_report": {
            "Tuberculosis": {
                "recall": 0.9000
            }
        }
    }

    gate = ModelQualityGate()

    result = gate.evaluate(
        candidate_metrics=candidate,
        baseline_metrics=baseline,
    )

    assert result.passed is True