import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_twin.io import load_twin_result, load_twin_scenario
from astro_twin.runner import run_digital_twin


def test_load_twin_scenario_reads_reference_example() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")

    assert scenario.scenario_id == "leo-observer"
    assert scenario.orbit_scenario == "examples/scenarios/leo_two_body.yaml"
    assert scenario.power.battery_capacity_wh == 1200.0


def test_load_twin_scenario_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    try:
        load_twin_scenario(path)
    except Exception as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("expected invalid scenario error")


def test_load_twin_result_rejects_rc1_artifact_without_margin_units(
    tmp_path: Path,
) -> None:
    result = run_digital_twin(load_twin_scenario("examples/twin/leo_observer.yaml"))
    payload = result.model_dump(mode="json")
    payload["workflow"] = "integrated_digital_twin_v1"
    for margin in payload["margin_report"]["margins"]:
        margin.pop("unit")
    payload["margin_report"]["limiting_margin"].pop("unit")
    path = tmp_path / "rc1-twin-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="unit"):
        load_twin_result(path)
