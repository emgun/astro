from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from astro_cli.main import app
from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local

runner = CliRunner()


def test_simulate_reentry_command_writes_json_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "reentry.json"
    summary = tmp_path / "nested" / "reentry.txt"

    result = runner.invoke(
        app,
        [
            "simulate-reentry",
            "examples/reentry/ballistic_capsule.yaml",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workflow"] == "reentry_3dof_v1"
    assert payload["peaks"]["heat_rate"]["value"] > 0.0
    assert "Peak heat rate W/m^2" in summary.read_text(encoding="utf-8")


def test_optimize_reentry_command_writes_product_and_tuned_scenario(tmp_path: Path) -> None:
    output = tmp_path / "optimization.json"
    tuned = tmp_path / "tuned.yaml"

    result = runner.invoke(
        app,
        [
            "optimize-reentry",
            "examples/reentry/guided_lifting_body.yaml",
            "--output",
            str(output),
            "--tuned-scenario-output",
            str(tuned),
            "--maximum-iterations",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    tuned_payload = yaml.safe_load(tuned.read_text(encoding="utf-8"))
    assert payload["final_objective"] <= payload["initial_objective"]
    assert tuned_payload["guidance"]["mode"] == "target_tracking"


def test_handoff_reentry_command_writes_scenario(tmp_path: Path) -> None:
    trajectory = propagate_local(load_scenario("examples/scenarios/leo_two_body.yaml"))
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "handoff.yaml"

    result = runner.invoke(
        app,
        [
            "handoff-reentry",
            str(trajectory_path),
            "examples/reentry/ballistic_capsule.yaml",
            "--output",
            str(output),
            "--sample-index",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["workflow"] == "trajectory_reentry_handoff"
    assert payload["initial_state"]["velocity_km_s"] > 0.0


def test_simulate_reentry_command_rejects_unknown_backend(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "simulate-reentry",
            "examples/reentry/ballistic_capsule.yaml",
            "--backend",
            "missing",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert result.exit_code == 2
    assert "unsupported reentry backend: missing" in result.output
