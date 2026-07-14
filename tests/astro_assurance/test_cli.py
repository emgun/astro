from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from astro_assurance.io import load_post_launch_assurance_scenario
from astro_cli.main import app


def test_run_mission_assurance_writes_result_summary_and_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "assurance.json"
    summary = tmp_path / "summary.txt"
    artifacts = tmp_path / "artifacts"

    result = CliRunner().invoke(
        app,
        [
            "run-mission-assurance",
            "examples/assurance/post_launch_orbit_acquisition.yaml",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--artifacts-dir",
            str(artifacts),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert "Status: pass" in summary.read_text(encoding="utf-8")
    assert (artifacts / "manifest.json").exists()
    verification = CliRunner().invoke(
        app,
        ["verify-mission-assurance", str(artifacts)],
    )
    assert verification.exit_code == 0, verification.output
    assert "16 artifacts, 4 inputs" in verification.output


def test_run_mission_assurance_returns_one_for_completed_requirement_failure(
    tmp_path: Path,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    strict_requirements = scenario.requirements.model_copy(
        update={"maximum_truth_recovery_position_error_km": 0.01}
    )
    strict_scenario = scenario.model_copy(update={"requirements": strict_requirements})
    scenario_path = tmp_path / "strict-assurance.yaml"
    output = tmp_path / "strict-assurance.json"
    scenario_path.write_text(
        yaml.safe_dump(strict_scenario.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["run-mission-assurance", str(scenario_path), "--output", str(output)],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["continuity_report"]["all_passed"] is True
