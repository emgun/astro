import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from astro_cli.main import app
from astro_core.errors import NumericalConvergenceError

runner = CliRunner(mix_stderr=False)


def test_validate_command_accepts_example_scenario() -> None:
    result = runner.invoke(app, ["validate", "examples/scenarios/leo_two_body.yaml"])

    assert result.exit_code == 0
    assert "leo-two-body" in result.stdout


def test_propagate_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "local",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert payload["backend"] == "local"
    assert len(payload["samples"]) == 11


def test_synth_measurements_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "measurements.json"

    result = runner.invoke(
        app,
        [
            "synth-measurements",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert len(payload["measurements"]) == 22


def test_estimate_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "estimate.json"

    result = runner.invoke(
        app,
        [
            "estimate",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["rms"] < 3.0
    assert payload["metadata"]["workflow"] == "local_synthetic_demo"
    assert payload["metadata"]["source_scenario_id"] == "leo-two-body"
    assert payload["metadata"]["source_ground_station_count"] == 1
    assert payload["metadata"]["truth_ground_station_count"] == 2
    assert payload["metadata"]["demo_added_ground_stations"] == ["demo-y-axis-eci"]
    assert payload["metadata"]["demo_added_ground_station_geometry"] == [
        {
            "name": "demo-y-axis-eci",
            "position_eci_km": [0.0, 6378.1363, 0.0],
            "frame": "EME2000",
            "elevation_mask_deg": 0.0,
        }
    ]
    assert payload["metadata"]["initial_guess_position_delta_km"] == [1.0, -0.8, 0.6]
    assert payload["metadata"]["initial_guess_velocity_delta_km_s"] == [0.0005, -0.001, 0.0008]
    assert payload["metadata"]["measurement_count"] == 44


def test_propagate_command_reports_invalid_scenario(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    result = runner.invoke(app, ["propagate", str(scenario)])

    assert result.exit_code == 2
    assert "is invalid" in result.stderr


def test_propagate_command_reports_unsupported_backend() -> None:
    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
        ],
    )

    assert result.exit_code == 2
    assert "unsupported propagation backend: orekit" in result.stderr


def test_propagate_command_reports_output_write_error(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "trajectory.json"

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write trajectory" in result.stderr
    assert str(output) in result.stderr


def test_synth_measurements_command_reports_output_write_error(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "measurements.json"

    result = runner.invoke(
        app,
        [
            "synth-measurements",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write measurements" in result.stderr
    assert str(output) in result.stderr


def test_estimate_command_reports_output_write_error(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "estimate.json"

    result = runner.invoke(
        app,
        [
            "estimate",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write estimate" in result.stderr
    assert str(output) in result.stderr


def test_synth_measurements_command_reports_invalid_scenario(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "synth-measurements",
            str(scenario),
            "--output",
            str(tmp_path / "measurements.json"),
        ],
    )

    assert result.exit_code == 2
    assert "is invalid" in result.stderr


def test_estimate_command_reports_invalid_scenario(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "estimate",
            str(scenario),
            "--output",
            str(tmp_path / "estimate.json"),
        ],
    )

    assert result.exit_code == 2
    assert "is invalid" in result.stderr


def test_estimate_command_reports_numerical_convergence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_estimate(*_args: object, **_kwargs: object) -> object:
        raise NumericalConvergenceError("forced OD failure")

    monkeypatch.setattr("astro_cli.main.estimate_initial_state", fail_estimate)

    result = runner.invoke(
        app,
        [
            "estimate",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(tmp_path / "estimate.json"),
        ],
    )

    assert result.exit_code == 2
    assert "forced OD failure" in result.stderr
