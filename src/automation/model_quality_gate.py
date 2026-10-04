from dataclasses import dataclass, asdict


@dataclass
class QualityGateResult:
    passed: bool
    reason: str
    candidate_metrics: dict
    baseline_metrics: dict

    def to_dict(self):
        return asdict(self)


class ModelQualityGate:
    """
    Determines whether a retrained model is safe to replace
    the current/baseline model.
    """

    def __init__(
        self,
        minimum_accuracy: float = 0.70,
        minimum_f1: float = 0.70,
        minimum_tb_recall: float = 0.80,
        maximum_accuracy_drop: float = 0.02,
        maximum_f1_drop: float = 0.02,
        maximum_tb_recall_drop: float = 0.05,
    ):
        self.minimum_accuracy = minimum_accuracy
        self.minimum_f1 = minimum_f1
        self.minimum_tb_recall = minimum_tb_recall

        self.maximum_accuracy_drop = maximum_accuracy_drop
        self.maximum_f1_drop = maximum_f1_drop
        self.maximum_tb_recall_drop = maximum_tb_recall_drop

    def evaluate(
        self,
        candidate_metrics: dict,
        baseline_metrics: dict,
    ):

        candidate_accuracy = candidate_metrics["accuracy"]
        candidate_f1 = candidate_metrics["f1"]

        candidate_tb_recall = self._get_tb_recall(
            candidate_metrics
        )

        baseline_accuracy = baseline_metrics["accuracy"]
        baseline_f1 = baseline_metrics["f1"]

        baseline_tb_recall = self._get_tb_recall(
            baseline_metrics
        )

        failures = []

        # --------------------------------------------------
        # ABSOLUTE QUALITY CHECKS
        # --------------------------------------------------

        if candidate_accuracy < self.minimum_accuracy:
            failures.append(
                f"Accuracy {candidate_accuracy:.4f} "
                f"is below minimum {self.minimum_accuracy:.4f}"
            )

        if candidate_f1 < self.minimum_f1:
            failures.append(
                f"F1 {candidate_f1:.4f} "
                f"is below minimum {self.minimum_f1:.4f}"
            )

        if candidate_tb_recall < self.minimum_tb_recall:
            failures.append(
                f"Tuberculosis recall {candidate_tb_recall:.4f} "
                f"is below minimum {self.minimum_tb_recall:.4f}"
            )

        # --------------------------------------------------
        # BASELINE COMPARISON
        # --------------------------------------------------

        accuracy_drop = (
            baseline_accuracy - candidate_accuracy
        )

        f1_drop = (
            baseline_f1 - candidate_f1
        )

        tb_recall_drop = (
            baseline_tb_recall - candidate_tb_recall
        )

        if accuracy_drop > self.maximum_accuracy_drop:
            failures.append(
                f"Accuracy dropped by {accuracy_drop:.4f}"
            )

        if f1_drop > self.maximum_f1_drop:
            failures.append(
                f"F1 dropped by {f1_drop:.4f}"
            )

        if tb_recall_drop > self.maximum_tb_recall_drop:
            failures.append(
                f"Tuberculosis recall dropped by "
                f"{tb_recall_drop:.4f}"
            )

        # --------------------------------------------------
        # FINAL DECISION
        # --------------------------------------------------

        if failures:

            return QualityGateResult(
                passed=False,
                reason="; ".join(failures),
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
            )

        return QualityGateResult(
            passed=True,
            reason="Candidate model passed all quality checks.",
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
        )

    @staticmethod
    def _get_tb_recall(metrics: dict) -> float:

        report = metrics.get(
            "classification_report",
            {}
        )

        tuberculosis = report.get(
            "Tuberculosis",
            {}
        )

        recall = tuberculosis.get(
            "recall"
        )

        if recall is None:
            raise ValueError(
                "Tuberculosis recall is missing from metrics."
            )

        return recall