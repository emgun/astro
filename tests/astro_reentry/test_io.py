from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_reentry.io import (
    format_reentry_summary,
    load_reentry_result,
    load_reentry_scenario,
)
from astro_reentry.simulation import simulate_reentry_local
from tests.astro_reentry.helpers import make_reentry_scenario


def test_load_checked_in_reentry_examples() -> None:
    ballistic = load_reentry_scenario("examples/reentry/ballistic_capsule.yaml")
    lifting = load_reentry_scenario("examples/reentry/lifting_bank_schedule.yaml")
    guided = load_reentry_scenario("examples/reentry/guided_lifting_body.yaml")

    assert ballistic.guidance.mode == "ballistic"
    assert lifting.guidance.mode == "bank_schedule"
    assert guided.guidance.mode == "target_tracking"


def test_reentry_result_json_round_trip_and_summary(tmp_path: Path) -> None:
    result = simulate_reentry_local(make_reentry_scenario())
    result_path = tmp_path / "reentry.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_reentry_result(result_path)
    summary = format_reentry_summary(loaded)

    assert loaded == result
    assert "Peak heat rate W/m^2" in summary
    assert "Limiting margin:" in summary


def test_reentry_loader_wraps_yaml_and_validation_errors(tmp_path: Path) -> None:
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("scenario_id: [", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="Could not parse reentry scenario"):
        load_reentry_scenario(invalid_yaml)

    invalid_mapping = tmp_path / "list.yaml"
    invalid_mapping.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="must contain a mapping"):
        load_reentry_scenario(invalid_mapping)
