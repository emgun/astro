from astro_assistant.planner import DeterministicPlanner
from astro_assistant.verification import verify_plan


def test_verifier_accepts_resolved_supported_scenario() -> None:
    plan = DeterministicPlanner().plan(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml"
    )

    result = verify_plan(plan)

    assert result.passed is True
    assert result.diagnostics == []


def test_verifier_rejects_silent_scenario_substitution() -> None:
    plan = DeterministicPlanner().plan("Run the local OD demo")
    tampered = plan.model_copy(
        update={
            "user_intent": (
                "Run local OD on examples/scenarios/leo_two_station_angles.yaml"
            )
        }
    )

    result = verify_plan(tampered)

    assert result.passed is False
    assert any("requested scenario" in diagnostic.message for diagnostic in result.diagnostics)


def test_verifier_rejects_output_paths_outside_artifact_directory() -> None:
    plan = DeterministicPlanner().plan("Run the local OD demo")
    bad_steps = list(plan.steps)
    bad_steps[1] = bad_steps[1].model_copy(
        update={
            "inputs": {
                **bad_steps[1].inputs,
                "output": "/tmp/not-astro/measurements.json",
            }
        }
    )
    tampered = plan.model_copy(update={"steps": bad_steps})

    result = verify_plan(tampered)

    assert result.passed is False
    assert any("artifact directory" in diagnostic.message for diagnostic in result.diagnostics)
