from pathlib import Path

import yaml

from astro_assistant.planner import DeterministicPlanner
from astro_assistant.scenarios import SUPPORTED_LOCAL_OD_SCENARIOS


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_local_od_workflow_pack_manifest_matches_supported_scenarios() -> None:
    manifest_path = Path("examples/workflows/local_od/manifest.yaml")

    manifest = _load_yaml(manifest_path)

    assert manifest["workflow_id"] == "local-od-demo"
    assert manifest["artifact_root"] == "/tmp/astro-assistant"
    assert manifest["default_scenario"] == "examples/scenarios/leo_two_station_od.yaml"
    assert manifest["golden_prompts"] == "examples/workflows/local_od/golden_prompts.yaml"
    assert manifest["supported_scenarios"] == [
        f"examples/scenarios/{filename}" for filename in SUPPORTED_LOCAL_OD_SCENARIOS
    ]
    assert manifest["steps"] == [
        "validate_scenario",
        "synth_measurements",
        "export_measurements_tdm",
        "estimate_state",
    ]
    assert manifest["report_metrics"] == [
        "measurement_count",
        "tdm_line_count",
        "estimate_converged",
        "estimate_iterations",
        "estimate_rms",
        "jacobian_rank",
        "residual_count",
        "max_abs_residual",
    ]


def test_local_od_golden_prompt_fixtures_cover_supported_scenarios() -> None:
    manifest = _load_yaml(Path("examples/workflows/local_od/manifest.yaml"))
    golden_path = Path(manifest["golden_prompts"])

    assert golden_path.exists()
    golden = _load_yaml(golden_path)

    fixtures = golden["fixtures"]
    expected_paths = [
        f"examples/scenarios/{filename}" for filename in SUPPORTED_LOCAL_OD_SCENARIOS
    ]

    assert golden["workflow_id"] == manifest["workflow_id"]
    assert golden["default_prompt"] == "Run the local OD demo"
    assert golden["default_scenario"] == manifest["default_scenario"]
    assert [fixture["scenario_path"] for fixture in fixtures] == expected_paths
    assert [fixture["scenario_id"] for fixture in fixtures] == list(
        SUPPORTED_LOCAL_OD_SCENARIOS.values()
    )


def test_local_od_golden_prompt_fixtures_build_expected_plans() -> None:
    manifest = _load_yaml(Path("examples/workflows/local_od/manifest.yaml"))
    golden = _load_yaml(Path(manifest["golden_prompts"]))
    planner = DeterministicPlanner()

    default_plan = planner.plan(golden["default_prompt"])
    assert default_plan.steps[0].inputs["scenario_path"] == golden["default_scenario"]

    for fixture in golden["fixtures"]:
        plan = planner.plan(fixture["prompt"])

        assert plan.plan_id == manifest["workflow_id"]
        assert plan.title == f"Local Orbit Determination Demo: {fixture['scenario_id']}"
        assert [step.step_id for step in plan.steps] == manifest["steps"]
        assert plan.steps[0].inputs["scenario_path"] == fixture["scenario_path"]
        assert plan.steps[1].inputs["scenario_path"] == fixture["scenario_path"]
        assert plan.steps[1].inputs["output"] == (
            f"{fixture['artifact_dir']}/measurements.json"
        )
        assert plan.steps[2].inputs["measurements_path"] == (
            f"{fixture['artifact_dir']}/measurements.json"
        )
        assert plan.steps[2].inputs["output"] == f"{fixture['artifact_dir']}/measurements.tdm"
        assert plan.steps[3].inputs["scenario_path"] == fixture["scenario_path"]
        assert plan.steps[3].inputs["measurements_path"] == (
            f"{fixture['artifact_dir']}/measurements.json"
        )
        assert plan.steps[3].inputs["output"] == f"{fixture['artifact_dir']}/estimate.json"
