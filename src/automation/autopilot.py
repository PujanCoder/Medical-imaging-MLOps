import logging

from src.automation.failure_detector import FailureDetector
from src.automation.failure_analyzer import FailureAnalyzer
from src.automation.recovery_engine import RecoveryEngine


logger = logging.getLogger(__name__)


class FailureAutopilot:
    """
    Central controller for automatic pipeline failure recovery.

    Flow:

        Detect
          ↓
        Analyze
          ↓
        Decide
          ↓
        Recover
          ↓
        Retry
          ↓
        Validate
    """

    def __init__(self):

        self.detector = FailureDetector()
        self.analyzer = FailureAnalyzer()
        self.engine = RecoveryEngine()

    def run_stage(
        self,
        stage_name,
        stage_function,
        recovery_functions=None,
    ):
        """
        Execute a pipeline stage.

        If the stage fails:

            failure detector
                    ↓
            failure analyzer
                    ↓
            recovery engine
                    ↓
                retry
                    ↓
               validation
        """

        try:

            logger.info(
                "Autopilot executing stage: %s",
                stage_name,
            )

            return stage_function()

        except Exception as error:

            logger.exception(
                "Pipeline stage failed: %s",
                stage_name,
            )

            # 1. DETECT
            failure = self.detector.detect(
                stage=stage_name,
                exception=error,
            )

            logger.error(
                "Failure detected: type=%s severity=%s",
                failure.failure_type,
                failure.severity,
            )

            # 2. ANALYZE
            analysis = self.analyzer.analyze(
                failure
            )

            logger.error(
                "Failure analysis: root_cause=%s action=%s",
                analysis.root_cause,
                analysis.recommended_action,
            )

            # 3. RECOVER + RETRY + VALIDATE
            recovery_result = self.engine.recover(
                failure=failure,
                analysis=analysis,
                recovery_functions=recovery_functions,
            )

            # 4. VALIDATE RESULT
            if recovery_result["success"]:

                logger.info(
                    "AUTOPILOT RECOVERY SUCCESS: stage=%s attempts=%s",
                    stage_name,
                    recovery_result["attempts"],
                )

                return recovery_result.get(
                    "result"
                )

            # 5. ESCALATE
            logger.error(
                "AUTOPILOT RECOVERY FAILED: stage=%s",
                stage_name,
            )

            raise RuntimeError(
                f"Autopilot could not recover stage "
                f"'{stage_name}': "
                f"{recovery_result['message']}"
            )