from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_mission.io import (
    format_mission_lifecycle_summary,
    load_mission_lifecycle_result,
    load_mission_lifecycle_scenario,
    write_mission_artifact_bundle,
    write_mission_lifecycle_result,
)
from astro_mission.runner import run_mission_lifecycle


def test_lifecycle_result_round_trip_and_artifact_bundle(tmp_path: Path) -> None:
    directory = tmp_path
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    result = run_mission_lifecycle(scenario)
    result_path = directory / "lifecycle.json"
    artifact_directory = directory / "artifacts"

    write_mission_lifecycle_result(result_path, result)
    write_mission_artifact_bundle(artifact_directory, result)

    loaded = load_mission_lifecycle_result(result_path)
    assert loaded.model_dump(mode="json") == result.model_dump(mode="json")
    assert "Status: pass" in format_mission_lifecycle_summary(loaded)
    expected = {
        "launch.json",
        "orbit-scenario.yaml",
        "operations-trajectory.json",
        "digital-twin.json",
        "deorbit-scenario.yaml",
        "deorbit-trajectory.json",
        "reentry-scenario.yaml",
        "reentry-result.json",
        "manifest.json",
    }
    assert {path.name for path in artifact_directory.iterdir()} == expected
    manifest = json.loads((artifact_directory / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["sequence"] for entry in manifest["entries"]] == [1, 2, 3, 4, 5]

    blocked_directory = directory / "not-a-directory"
    blocked_directory.write_text("blocked", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="Could not create mission artifact directory"):
        write_mission_artifact_bundle(blocked_directory, result)
