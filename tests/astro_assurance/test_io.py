from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import astro_assurance.io as assurance_io
from astro_assurance.io import (
    format_mission_assurance_summary,
    load_mission_assurance_result,
    load_post_launch_assurance_scenario,
    verify_mission_assurance_artifact_bundle,
    write_mission_assurance_artifact_bundle,
    write_mission_assurance_result,
)
from astro_assurance.models import MissionAssuranceCase
from astro_assurance.runner import run_post_launch_assurance
from astro_core.errors import InvalidScenarioError


def test_assurance_result_round_trip_and_artifact_bundle(tmp_path: Path) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    result = run_post_launch_assurance(scenario)
    result_path = tmp_path / "assurance.json"
    artifact_directory = tmp_path / "artifacts"

    write_mission_assurance_result(result_path, result)
    write_mission_assurance_artifact_bundle(artifact_directory, result)

    loaded = load_mission_assurance_result(result_path)
    assert loaded.model_dump(mode="json") == result.model_dump(mode="json")
    assert "Status: pass" in format_mission_assurance_summary(loaded)
    expected = {
        "launch.json",
        "measurements.json",
        "estimate.json",
        "candidate-maneuver.json",
        "nominal-trajectory.json",
        "truth-trajectory.json",
        "estimated-corrected-trajectory.json",
        "truth-corrected-trajectory.json",
        "corrected-digital-twin.json",
        "nominal-scenario.yaml",
        "truth-scenario.yaml",
        "estimated-corrected-scenario.yaml",
        "truth-corrected-scenario.yaml",
        "continuity-report.json",
        "margin-report.json",
        "decision.json",
        "manifest.json",
    }
    assert {path.name for path in artifact_directory.iterdir()} == expected
    manifest = verify_mission_assurance_artifact_bundle(artifact_directory)
    assert [entry.sequence for entry in manifest.entries] == list(range(1, 17))
    assert {entry.artifact_name for entry in manifest.entries} == expected - {"manifest.json"}
    assert all(Path(reference.path).is_absolute() for reference in manifest.inputs)

    blocked_directory = tmp_path / "not-a-directory"
    blocked_directory.write_text("blocked", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="already exists"):
        write_mission_assurance_artifact_bundle(blocked_directory, result)


def test_assurance_bundle_verifier_rejects_tampered_artifact(tmp_path: Path) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    result = run_post_launch_assurance(scenario)
    artifact_directory = tmp_path / "artifacts"
    write_mission_assurance_artifact_bundle(artifact_directory, result)
    decision_path = artifact_directory / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["passed"] = not decision["passed"]
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="digest mismatch: decision.json"):
        verify_mission_assurance_artifact_bundle(artifact_directory)


def test_assurance_bundle_verifier_rejects_manifest_omission(tmp_path: Path) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    result = run_post_launch_assurance(scenario)
    artifact_directory = tmp_path / "artifacts"
    write_mission_assurance_artifact_bundle(artifact_directory, result)
    manifest_path = artifact_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"] = [
        entry for entry in manifest["entries"] if entry["artifact_name"] != "decision.json"
    ]
    for index, entry in enumerate(manifest["entries"], start=1):
        entry["sequence"] = index
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (artifact_directory / "decision.json").unlink()

    with pytest.raises(InvalidScenarioError, match="fixed workflow artifact set"):
        verify_mission_assurance_artifact_bundle(artifact_directory)


def test_assurance_bundle_verifier_rejects_changed_assurance_input(tmp_path: Path) -> None:
    source = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    scenario_path = tmp_path / "assurance.yaml"
    scenario_path.write_text(
        yaml.safe_dump(source.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    scenario = load_post_launch_assurance_scenario(scenario_path)
    result = run_post_launch_assurance(scenario)
    artifact_directory = tmp_path / "artifacts"
    write_mission_assurance_artifact_bundle(artifact_directory, result)
    scenario_path.write_text(
        scenario_path.read_text(encoding="utf-8") + "# changed\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidScenarioError, match="assurance_scenario"):
        verify_mission_assurance_artifact_bundle(artifact_directory)


def test_assurance_bundle_publish_is_atomic_and_refuses_stale_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    result = run_post_launch_assurance(scenario)
    artifact_directory = tmp_path / "artifacts"
    original_write = assurance_io._write_text
    write_count = 0

    def interrupt_write(path: Path, content: str) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise InvalidScenarioError("simulated interrupted write")
        original_write(path, content)

    monkeypatch.setattr(assurance_io, "_write_text", interrupt_write)
    with pytest.raises(InvalidScenarioError, match="simulated interrupted write"):
        write_mission_assurance_artifact_bundle(artifact_directory, result)
    assert not artifact_directory.exists()
    assert not list(tmp_path.glob(".artifacts.*"))

    monkeypatch.setattr(assurance_io, "_write_text", original_write)
    artifact_directory.mkdir()
    (artifact_directory / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="already exists"):
        write_mission_assurance_artifact_bundle(artifact_directory, result)


def test_assurance_bundle_does_not_replace_destination_created_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    result = run_post_launch_assurance(scenario)
    artifact_directory = tmp_path / "artifacts"
    original_stage = assurance_io._write_mission_assurance_staging_bundle

    def stage_then_race(
        staging_directory: Path,
        assurance_result: MissionAssuranceCase,
    ) -> None:
        original_stage(staging_directory, assurance_result)
        artifact_directory.mkdir()

    monkeypatch.setattr(
        assurance_io,
        "_write_mission_assurance_staging_bundle",
        stage_then_race,
    )
    with pytest.raises(InvalidScenarioError, match="appeared during publication"):
        write_mission_assurance_artifact_bundle(artifact_directory, result)
    assert artifact_directory.exists()
    assert not any(artifact_directory.iterdir())
