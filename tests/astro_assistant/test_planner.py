import pytest

from astro_assistant.models import ArtifactKind, AstroToolName, RiskLevel
from astro_assistant.planner import DeterministicPlanner

UNSUPPORTED_PROMPT_MESSAGE = "deterministic planner currently supports the local OD demo only"


def test_deterministic_planner_builds_local_od_workflow() -> None:
    planner = DeterministicPlanner()
    scenario_path = "examples/scenarios/leo_two_station_od.yaml"
    artifact_dir = "/tmp/astro-assistant/leo_two_station_od"
    measurements_json = f"{artifact_dir}/measurements.json"
    measurements_tdm = f"{artifact_dir}/measurements.tdm"
    estimate_json = f"{artifact_dir}/estimate.json"

    plan = planner.plan("Run the local OD demo and export TDM")

    assert plan.plan_id == "local-od-demo"
    assert [step.step_id for step in plan.steps] == [
        "validate_scenario",
        "synth_measurements",
        "export_measurements_tdm",
        "estimate_state",
    ]
    assert [step.tool for step in plan.steps] == [
        AstroToolName.VALIDATE_SCENARIO,
        AstroToolName.SYNTH_MEASUREMENTS,
        AstroToolName.EXPORT_MEASUREMENTS,
        AstroToolName.ESTIMATE_MEASUREMENTS,
    ]
    assert [step.risk for step in plan.steps] == [
        RiskLevel.READ_ONLY,
        RiskLevel.WRITES_ARTIFACTS,
        RiskLevel.WRITES_ARTIFACTS,
        RiskLevel.WRITES_ARTIFACTS,
    ]
    assert plan.steps[0].inputs == {"scenario_path": scenario_path}
    assert plan.steps[1].inputs == {
        "scenario_path": scenario_path,
        "backend": "local",
        "output": measurements_json,
    }
    assert plan.steps[2].inputs == {
        "measurements_path": measurements_json,
        "format": "tdm",
        "output": measurements_tdm,
    }
    assert plan.steps[3].inputs == {
        "scenario_path": scenario_path,
        "measurements_path": measurements_json,
        "backend": "local",
        "output": estimate_json,
    }
    assert [[artifact.kind for artifact in step.outputs] for step in plan.steps] == [
        [ArtifactKind.SCENARIO],
        [ArtifactKind.MEASUREMENTS_JSON],
        [ArtifactKind.MEASUREMENTS_TDM],
        [ArtifactKind.ESTIMATE_JSON],
    ]
    assert [[artifact.path for artifact in step.outputs] for step in plan.steps] == [
        [scenario_path],
        [measurements_json],
        [measurements_tdm],
        [estimate_json],
    ]


def test_deterministic_planner_accepts_local_orbit_determination_demo() -> None:
    planner = DeterministicPlanner()

    plan = planner.plan(
        "Run the local orbit-determination demo, export the generated measurements as TDM, "
        "estimate the initial state, and write a trace."
    )

    assert plan.plan_id == "local-od-demo"


def test_deterministic_planner_binds_requested_supported_scenario() -> None:
    planner = DeterministicPlanner()

    plan = planner.plan(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml "
        "and export TDM."
    )

    assert plan.steps[0].inputs["scenario_path"] == (
        "examples/scenarios/leo_two_station_angles.yaml"
    )
    assert plan.steps[1].inputs["scenario_path"] == (
        "examples/scenarios/leo_two_station_angles.yaml"
    )
    assert plan.steps[3].inputs["scenario_path"] == (
        "examples/scenarios/leo_two_station_angles.yaml"
    )
    assert plan.steps[1].inputs["output"] == (
        "/tmp/astro-assistant/leo_two_station_angles/measurements.json"
    )
    assert plan.steps[2].inputs["output"] == (
        "/tmp/astro-assistant/leo_two_station_angles/measurements.tdm"
    )
    assert plan.steps[3].inputs["output"] == (
        "/tmp/astro-assistant/leo_two_station_angles/estimate.json"
    )


def test_deterministic_planner_resolves_supported_scenario_alias() -> None:
    planner = DeterministicPlanner()

    plan = planner.plan("Run the local OD workflow for radiometric media")

    assert plan.steps[0].inputs["scenario_path"] == (
        "examples/scenarios/leo_radiometric_media.yaml"
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Run OD with a live provider",
        "Perform operational orbit determination",
        "Run food demo",
    ],
)
def test_deterministic_planner_rejects_unsupported_od_prompts(prompt: str) -> None:
    planner = DeterministicPlanner()

    with pytest.raises(ValueError, match=UNSUPPORTED_PROMPT_MESSAGE):
        planner.plan(prompt)
