from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import ForceModelName


def test_load_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    assert scenario.scenario_id == "leo-two-body"
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.propagation.sample_count == 11


def test_load_scenario_reports_yaml_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_id: [broken", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="Could not parse scenario file"):
        load_scenario(path)


def test_load_scenario_reports_utf8_decode_errors_as_read_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad-encoding.yaml"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(InvalidScenarioError, match="Could not read scenario file"):
        load_scenario(path)


def test_load_scenario_requires_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="must contain a mapping"):
        load_scenario(path)


def test_load_scenario_wraps_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-scenario.yaml"
    path.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="is invalid"):
        load_scenario(path)
