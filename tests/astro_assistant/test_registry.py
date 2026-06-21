from typing import Any

import pytest

from astro_assistant.models import AstroToolName, WorkflowStep
from astro_assistant.registry import build_command_spec


SCENARIO_PATH = "examples/scenarios/leo_two_station_od.yaml"
MEASUREMENTS_PATH = "/tmp/astro-assistant/measurements.json"
ESTIMATE_PATH = "/tmp/astro-assistant/estimate.json"
TDM_PATH = "/tmp/astro-assistant/measurements.tdm"


def test_validate_scenario_command_spec() -> None:
    step = WorkflowStep(
        step_id="validate_scenario",
        tool=AstroToolName.VALIDATE_SCENARIO,
        description="Validate scenario.",
        inputs={"scenario_path": SCENARIO_PATH},
        risk="read_only",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "validate",
        SCENARIO_PATH,
    ]
    assert command.cwd == "/workspace"
    assert command.writes == []


def test_synth_measurements_command_spec_records_output_write() -> None:
    step = WorkflowStep(
        step_id="synth",
        tool=AstroToolName.SYNTH_MEASUREMENTS,
        description="Synthesize measurements.",
        inputs={
            "scenario_path": SCENARIO_PATH,
            "backend": "local",
            "output": MEASUREMENTS_PATH,
        },
        risk="writes_artifacts",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "synth-measurements",
        SCENARIO_PATH,
        "--backend",
        "local",
        "--output",
        MEASUREMENTS_PATH,
    ]
    assert command.cwd == "/workspace"
    assert command.writes == [MEASUREMENTS_PATH]


def test_export_measurements_command_spec_records_output_write() -> None:
    step = WorkflowStep(
        step_id="export",
        tool=AstroToolName.EXPORT_MEASUREMENTS,
        description="Export measurements.",
        inputs={
            "measurements_path": MEASUREMENTS_PATH,
            "format": "tdm",
            "output": TDM_PATH,
        },
        risk="writes_artifacts",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "export-measurements",
        MEASUREMENTS_PATH,
        "--format",
        "tdm",
        "--output",
        TDM_PATH,
    ]
    assert command.cwd == "/workspace"
    assert command.writes == [TDM_PATH]


def test_estimate_measurements_command_spec_records_output_write() -> None:
    step = WorkflowStep(
        step_id="estimate",
        tool=AstroToolName.ESTIMATE_MEASUREMENTS,
        description="Estimate initial state.",
        inputs={
            "scenario_path": SCENARIO_PATH,
            "measurements_path": MEASUREMENTS_PATH,
            "backend": "local",
            "output": ESTIMATE_PATH,
        },
        risk="writes_artifacts",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "estimate-measurements",
        SCENARIO_PATH,
        MEASUREMENTS_PATH,
        "--backend",
        "local",
        "--output",
        ESTIMATE_PATH,
    ]
    assert command.writes == [ESTIMATE_PATH]


@pytest.mark.parametrize(
    ("tool", "inputs", "missing_key"),
    [
        (AstroToolName.VALIDATE_SCENARIO, {}, "scenario_path"),
        (AstroToolName.SYNTH_MEASUREMENTS, {"output": MEASUREMENTS_PATH}, "scenario_path"),
        (AstroToolName.SYNTH_MEASUREMENTS, {"scenario_path": SCENARIO_PATH}, "output"),
        (AstroToolName.EXPORT_MEASUREMENTS, {"output": TDM_PATH}, "measurements_path"),
        (
            AstroToolName.EXPORT_MEASUREMENTS,
            {"measurements_path": MEASUREMENTS_PATH},
            "output",
        ),
        (
            AstroToolName.ESTIMATE_MEASUREMENTS,
            {"measurements_path": MEASUREMENTS_PATH, "output": ESTIMATE_PATH},
            "scenario_path",
        ),
        (
            AstroToolName.ESTIMATE_MEASUREMENTS,
            {"scenario_path": SCENARIO_PATH, "output": ESTIMATE_PATH},
            "measurements_path",
        ),
        (
            AstroToolName.ESTIMATE_MEASUREMENTS,
            {"scenario_path": SCENARIO_PATH, "measurements_path": MEASUREMENTS_PATH},
            "output",
        ),
    ],
)
def test_build_command_spec_rejects_missing_required_inputs(
    tool: AstroToolName, inputs: dict[str, Any], missing_key: str
) -> None:
    step = WorkflowStep(
        step_id="missing_input",
        tool=tool,
        description="Missing input.",
        inputs=inputs,
        risk="writes_artifacts",
    )

    with pytest.raises(ValueError, match=f"missing_input requires string input {missing_key!r}"):
        build_command_spec(step)


@pytest.mark.parametrize("bad_value", ["", 123])
def test_build_command_spec_rejects_empty_or_non_string_required_input(
    bad_value: object,
) -> None:
    step = WorkflowStep(
        step_id="bad_required",
        tool=AstroToolName.VALIDATE_SCENARIO,
        description="Bad required input.",
        inputs={"scenario_path": bad_value},
        risk="read_only",
    )

    with pytest.raises(
        ValueError, match="bad_required requires string input 'scenario_path'"
    ):
        build_command_spec(step)


@pytest.mark.parametrize(
    ("tool", "inputs", "optional_key"),
    [
        (
            AstroToolName.SYNTH_MEASUREMENTS,
            {"scenario_path": SCENARIO_PATH, "output": MEASUREMENTS_PATH, "backend": ""},
            "backend",
        ),
        (
            AstroToolName.ESTIMATE_MEASUREMENTS,
            {
                "scenario_path": SCENARIO_PATH,
                "measurements_path": MEASUREMENTS_PATH,
                "output": ESTIMATE_PATH,
                "backend": "",
            },
            "backend",
        ),
        (
            AstroToolName.EXPORT_MEASUREMENTS,
            {"measurements_path": MEASUREMENTS_PATH, "output": TDM_PATH, "format": ""},
            "format",
        ),
    ],
)
def test_build_command_spec_rejects_empty_optional_inputs(
    tool: AstroToolName, inputs: dict[str, Any], optional_key: str
) -> None:
    step = WorkflowStep(
        step_id="bad_optional",
        tool=tool,
        description="Bad optional input.",
        inputs=inputs,
        risk="writes_artifacts",
    )

    with pytest.raises(ValueError, match=f"bad_optional requires string input {optional_key!r}"):
        build_command_spec(step)
