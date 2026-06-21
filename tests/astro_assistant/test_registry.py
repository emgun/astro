from astro_assistant.models import AstroToolName, WorkflowStep
from astro_assistant.registry import build_command_spec


def test_validate_scenario_command_spec() -> None:
    step = WorkflowStep(
        step_id="validate_scenario",
        tool=AstroToolName.VALIDATE_SCENARIO,
        description="Validate scenario.",
        inputs={"scenario_path": "examples/scenarios/leo_two_station_od.yaml"},
        risk="read_only",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "validate",
        "examples/scenarios/leo_two_station_od.yaml",
    ]
    assert command.cwd == "/workspace"
    assert command.writes == []


def test_estimate_measurements_command_spec_records_output_write() -> None:
    step = WorkflowStep(
        step_id="estimate",
        tool=AstroToolName.ESTIMATE_MEASUREMENTS,
        description="Estimate initial state.",
        inputs={
            "scenario_path": "examples/scenarios/leo_two_station_od.yaml",
            "measurements_path": "/tmp/astro-assistant/measurements.json",
            "backend": "local",
            "output": "/tmp/astro-assistant/estimate.json",
        },
        risk="writes_artifacts",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "estimate-measurements",
        "examples/scenarios/leo_two_station_od.yaml",
        "/tmp/astro-assistant/measurements.json",
        "--backend",
        "local",
        "--output",
        "/tmp/astro-assistant/estimate.json",
    ]
    assert command.writes == ["/tmp/astro-assistant/estimate.json"]
