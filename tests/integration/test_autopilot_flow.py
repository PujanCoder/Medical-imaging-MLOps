from src.automation.autopilot import FailureAutopilot


def test_autopilot_failure_recovery_flow():

    autopilot = FailureAutopilot()

    state = {
        "attempts": 0
    }

    def unstable_stage():

        state["attempts"] += 1

        if state["attempts"] == 1:
            raise RuntimeError("Temporary training failure")

        return {
            "success": True,
            "message": "Stage recovered successfully."
        }

    result = autopilot.run_stage(
        stage_name="training",
        stage_function=unstable_stage,
        recovery_functions={
            "RETRY_TRAINING": unstable_stage,
            "RETRY_STAGE": unstable_stage,
        },
    )

    assert result["success"] is True
    assert state["attempts"] == 2