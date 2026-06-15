import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from astro_backends.orekit import OrekitSmokeResult
from astro_cli.main import app
from astro_core.errors import NumericalConvergenceError, UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import CartesianState, MeasurementType, Scenario
from astro_dynamics.local import propagate_local
from astro_launch.io import load_launch_scenario
from astro_launch.local import propagate_launch_local
from astro_launch.reporting import generate_tuned_launch_report
from astro_od.io import load_measurements
from astro_od.measurements import generate_synthetic_measurements
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario

runner = CliRunner(mix_stderr=False)


def _observable_scenario() -> Scenario:
    return load_scenario(Path("examples/scenarios/leo_two_station_od.yaml"))


def _perturbed_scenario(scenario: Scenario) -> Scenario:
    perturbed_state = scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=(7001.0, -0.8, 0.6),
                velocity_km_s=(0.0005, 7.499, 1.0008),
            )
        }
    )
    return scenario.model_copy(update={"initial_state": perturbed_state})


def _write_scenario(path: Path, scenario: Scenario) -> None:
    payload = scenario.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_launch_scenario(path: Path) -> None:
    payload = make_launch_scenario().model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_pitch_program_launch_scenario(path: Path) -> None:
    payload = make_pitch_program_launch_scenario().model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_launch_trajectory(path: Path) -> None:
    trajectory = propagate_launch_local(make_launch_scenario())
    path.write_text(trajectory.model_dump_json(), encoding="utf-8")


def _write_tuned_launch_report(path: Path, *, iterations: int) -> None:
    report = generate_tuned_launch_report(
        make_pitch_program_launch_scenario(),
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=iterations,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )
    path.write_text(report.model_dump_json(), encoding="utf-8")


def _write_measurements(path: Path, scenario: Scenario) -> None:
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    path.write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "measurements": [record.model_dump(mode="json") for record in measurements],
            }
        ),
        encoding="utf-8",
    )


def _write_measurements_csv(path: Path, scenario: Scenario) -> None:
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    fieldnames = [
        "scenario_id",
        "measurement_type",
        "epoch",
        "observer",
        "observed_object",
        "value",
        "sigma",
        "units",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as measurement_file:
        writer = csv.DictWriter(measurement_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in measurements:
            payload = record.model_dump(mode="json")
            writer.writerow(
                {"scenario_id": scenario.scenario_id}
                | {
                    fieldname: (
                        json.dumps(payload["metadata"])
                        if fieldname == "metadata_json"
                        else payload[fieldname]
                    )
                    for fieldname in fieldnames
                    if fieldname != "scenario_id"
                }
            )


def _write_measurements_tdm(path: Path, scenario: Scenario) -> None:
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    station_index = {station.name: station for station in scenario.ground_stations}
    lines = [
        "CCSDS_TDM_VERS = 2.0",
        "CREATION_DATE = 2026-01-01T00:00:00Z",
        "ORIGINATOR = ASTRO_SUITE_TEST",
    ]
    for station in scenario.ground_stations:
        lines.extend(
            [
                "META_START",
                f"SCENARIO_ID = {scenario.scenario_id}",
                "TIME_SYSTEM = UTC",
                "MODE = SEQUENTIAL",
                f"PARTICIPANT_1 = {station.name}",
                f"PARTICIPANT_2 = {scenario.spacecraft.name}",
                "PATH = 1,2,1",
                "RANGE_UNITS = km",
                "META_STOP",
                "DATA_START",
            ]
        )
        for record in measurements:
            if record.observer != station.name:
                continue
            assert record.observer in station_index
            tdm_keyword = (
                "RANGE"
                if record.measurement_type is MeasurementType.RANGE
                else "DOPPLER_INSTANTANEOUS"
            )
            lines.append(f"{tdm_keyword} = {record.epoch.isoformat()} {record.value}")
        lines.append("DATA_STOP")
    path.write_text("\n".join(lines), encoding="utf-8")


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


def test_launch_command_writes_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "launch.yaml"
    output = tmp_path / "launch.json"
    _write_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "launch",
            str(scenario_path),
            "--backend",
            "local",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote launch trajectory" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "vertical-two-stage"
    assert payload["backend"] == "local"
    assert len(payload["samples"]) == 15
    assert payload["events"][-1]["event_type"] == "insertion"
    assert payload["insertion_state"]["central_body"] == "earth"
    assert payload["metadata"]["model"] == "vertical_1d"


def test_handoff_launch_command_writes_orbit_scenario(tmp_path: Path) -> None:
    launch_output = tmp_path / "launch.json"
    orbit_scenario_path = tmp_path / "insertion.yaml"
    _write_launch_trajectory(launch_output)

    result = runner.invoke(
        app,
        [
            "handoff-launch",
            str(launch_output),
            "--output",
            str(orbit_scenario_path),
            "--duration-s",
            "600",
            "--step-s",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert "wrote orbit scenario" in result.stdout
    scenario = load_scenario(orbit_scenario_path)
    propagated = propagate_local(scenario)
    launch_payload = json.loads(launch_output.read_text(encoding="utf-8"))
    assert scenario.scenario_id == "vertical-two-stage-insertion"
    assert scenario.initial_state.model_dump(mode="json") == launch_payload["insertion_state"]
    assert scenario.spacecraft.mass_kg == launch_payload["samples"][-1]["mass_kg"]
    assert scenario.propagation.duration_s == 600.0
    assert scenario.propagation.step_s == 60.0
    assert scenario.metadata["workflow"] == "launch_orbit_handoff"
    assert len(propagated.samples) == 11


def test_sweep_launch_pitch_command_writes_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "sweep.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "sweep-launch-pitch",
            str(scenario_path),
            "--point-index",
            "3",
            "--pitch-deg-values",
            "10,20,30",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote launch pitch sweep" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["point_index"] == 3
    assert payload["point_time_s"] == 110.0
    assert payload["baseline_pitch_deg"] == 20.0
    assert [case["pitch_deg"] for case in payload["cases"]] == [10.0, 20.0, 30.0]
    assert payload["best_case"]["pitch_deg"] in [10.0, 20.0, 30.0]
    assert payload["best_case"]["score"] == min(case["score"] for case in payload["cases"])


def test_sweep_launch_pitch_command_reports_invalid_pitch_values(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "sweep-launch-pitch",
            str(scenario_path),
            "--point-index",
            "3",
            "--pitch-deg-values",
            "10,bad,30",
            "--output",
            str(tmp_path / "sweep.json"),
        ],
    )

    assert result.exit_code == 2
    assert "pitch-deg-values must be comma-separated numbers" in result.stderr


def test_sweep_launch_pitch_command_reports_output_write_error(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "missing" / "sweep.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "sweep-launch-pitch",
            str(scenario_path),
            "--point-index",
            "3",
            "--pitch-deg-values",
            "10,20,30",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write launch pitch sweep" in result.stderr
    assert str(output) in result.stderr


def test_tune_launch_pitch_command_writes_json_and_tuned_scenario(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "tuning.json"
    tuned_scenario_output = tmp_path / "tuned.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "tune-launch-pitch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--initial-span-deg",
            "10",
            "--iterations",
            "2",
            "--output",
            str(output),
            "--tuned-scenario-output",
            str(tuned_scenario_output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote launch pitch tuning" in result.stdout
    assert "wrote tuned launch scenario" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    tuned_scenario = load_launch_scenario(tuned_scenario_output)
    tuned_pitches = {
        point["point_index"]: point["tuned_pitch_deg"] for point in payload["tuned_points"]
    }
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["point_indices"] == [2, 3]
    assert len(payload["iterations"]) == 2
    assert payload["best_case"]["score"] == min(
        case["score"] for iteration in payload["iterations"] for case in iteration["cases"]
    )
    assert tuned_scenario.guidance.pitch_program[2].pitch_deg == tuned_pitches[2]
    assert tuned_scenario.guidance.pitch_program[3].pitch_deg == tuned_pitches[3]


def test_tune_launch_pitch_command_reports_invalid_point_indices(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "tune-launch-pitch",
            str(scenario_path),
            "--point-indices",
            "2,bad",
            "--output",
            str(tmp_path / "tuning.json"),
        ],
    )

    assert result.exit_code == 2
    assert "point-indices must be two comma-separated integers" in result.stderr


def test_tune_launch_pitch_command_reports_tuned_scenario_write_error(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "tuning.json"
    tuned_scenario_output = tmp_path / "missing" / "tuned.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "tune-launch-pitch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--output",
            str(output),
            "--tuned-scenario-output",
            str(tuned_scenario_output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write tuned launch scenario" in result.stderr
    assert str(tuned_scenario_output) in result.stderr


def test_report_tuned_launch_command_writes_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "report.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "report-tuned-launch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--initial-span-deg",
            "10",
            "--iterations",
            "2",
            "--orbit-duration-s",
            "600",
            "--orbit-step-s",
            "60",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote tuned launch report" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["tuning_result"]["point_indices"] == [2, 3]
    assert payload["launch_trajectory"]["metadata"]["guidance_mode"] == "pitch_program"
    assert payload["orbit_scenario"]["metadata"]["workflow"] == "launch_orbit_handoff"
    assert len(payload["orbit_trajectory"]["samples"]) == 11
    assert payload["short_arc_metrics"]["sample_count"] == 11
    assert payload["insertion_metrics"]["altitude_miss_km"] == payload["launch_trajectory"][
        "target_miss"
    ]["altitude_miss_km"]
    assert payload["passed"] is False
    assert payload["insertion_assessment"]["passed"] is False
    assert payload["short_arc_assessment"]["passed"] is False
    assert [check["name"] for check in payload["insertion_assessment"]["checks"]] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
    ]
    assert [check["name"] for check in payload["short_arc_assessment"]["checks"]] == [
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]


def test_report_tuned_launch_command_reports_invalid_point_indices(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "report-tuned-launch",
            str(scenario_path),
            "--point-indices",
            "2,bad",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 2
    assert "point-indices must be two comma-separated integers" in result.stderr


def test_report_tuned_launch_command_reports_output_write_error(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "missing" / "report.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "report-tuned-launch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write tuned launch report" in result.stderr
    assert str(output) in result.stderr


def test_compare_tuned_launch_reports_command_writes_json(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _write_tuned_launch_report(baseline_path, iterations=1)
    _write_tuned_launch_report(candidate_path, iterations=2)

    result = runner.invoke(
        app,
        [
            "compare-tuned-launch-reports",
            str(baseline_path),
            str(candidate_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote tuned launch report comparison" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["baseline_scenario_id"] == "pitch-program-two-stage"
    assert payload["candidate_scenario_id"] == "pitch-program-two-stage"
    assert payload["baseline_passed"] is False
    assert payload["candidate_passed"] is False
    assert [metric["name"] for metric in payload["metric_deltas"]] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]


def test_batch_report_tuned_launch_command_writes_ranked_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "batch.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "batch-report-tuned-launch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--iterations-values",
            "1,2",
            "--initial-span-deg",
            "10",
            "--orbit-duration-s",
            "600",
            "--orbit-step-s",
            "60",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote tuned launch report batch" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["point_indices"] == [2, 3]
    assert [case["rank"] for case in payload["cases"]] == [1, 2]
    assert {case["iterations"] for case in payload["cases"]} == {1, 2}
    assert payload["best_case"] == payload["cases"][0]


def test_batch_report_tuned_launch_command_reports_invalid_iterations_values(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "batch-report-tuned-launch",
            str(scenario_path),
            "--iterations-values",
            "1,bad",
            "--output",
            str(tmp_path / "batch.json"),
        ],
    )

    assert result.exit_code == 2
    assert "iterations-values must be comma-separated positive integers" in result.stderr


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


def test_synth_measurements_command_accepts_orekit_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "measurements.json"
    seen_backends: list[str] = []

    def fake_backend_propagation(scenario: Scenario, backend: str) -> object:
        seen_backends.append(backend)
        return propagate_local(scenario)

    monkeypatch.setattr("astro_cli.main.propagate_with_backend", fake_backend_propagation)

    result = runner.invoke(
        app,
        [
            "synth-measurements",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen_backends == ["orekit"]
    payload = json.loads(output.read_text(encoding="utf-8"))
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


def test_estimate_command_accepts_orekit_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "estimate.json"
    seen_backends: list[str] = []

    def fake_backend_propagation(scenario: Scenario, backend: str) -> object:
        seen_backends.append(backend)
        return propagate_local(scenario)

    monkeypatch.setattr("astro_od.estimation.propagate_with_backend", fake_backend_propagation)
    monkeypatch.setattr("astro_cli.main.propagate_with_backend", fake_backend_propagation)

    result = runner.invoke(
        app,
        [
            "estimate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["propagation_backend"] == "orekit"
    assert set(seen_backends) == {"orekit"}


def test_estimate_measurements_command_writes_json(tmp_path: Path) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements(measurements_path, truth_scenario)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["rms"] < 3.0
    assert payload["metadata"]["workflow"] == "local_measurement_file"
    assert payload["metadata"]["source_scenario_id"] == "leo-two-station-od"
    assert payload["metadata"]["measurement_file"] == str(measurements_path)
    assert payload["metadata"]["measurement_format"] == "json"
    assert payload["metadata"]["measurement_count"] == 44
    assert "demo_added_ground_stations" not in payload["metadata"]


def test_estimate_measurements_command_accepts_orekit_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements(measurements_path, truth_scenario)

    def fake_backend_propagation(scenario: Scenario, backend: str) -> object:
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": backend})

    monkeypatch.setattr("astro_od.estimation.propagate_with_backend", fake_backend_propagation)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--backend",
            "orekit",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["propagation_backend"] == "orekit"
    assert payload["metadata"]["workflow"] == "local_measurement_file"


def test_estimate_measurements_command_accepts_csv(tmp_path: Path) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.csv"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements_csv(measurements_path, truth_scenario)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--format",
            "csv",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["metadata"]["workflow"] == "local_measurement_file"
    assert payload["metadata"]["measurement_format"] == "csv"
    assert payload["metadata"]["measurement_count"] == 44


def test_estimate_measurements_command_accepts_tdm(tmp_path: Path) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.tdm"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements_tdm(measurements_path, truth_scenario)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["metadata"]["workflow"] == "local_measurement_file"
    assert payload["metadata"]["measurement_format"] == "tdm"
    assert payload["metadata"]["measurement_count"] == 44


def test_export_measurements_command_writes_csv(tmp_path: Path) -> None:
    scenario = _observable_scenario()
    input_path = tmp_path / "measurements.json"
    output_path = tmp_path / "measurements.csv"
    _write_measurements(input_path, scenario)

    result = runner.invoke(
        app,
        [
            "export-measurements",
            str(input_path),
            "--format",
            "csv",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "wrote measurements" in result.stdout
    assert load_measurements(output_path, expected_scenario_id=scenario.scenario_id) == (
        load_measurements(input_path, expected_scenario_id=scenario.scenario_id)
    )


def test_export_measurements_command_writes_tdm(tmp_path: Path) -> None:
    scenario = _observable_scenario()
    input_path = tmp_path / "measurements.json"
    output_path = tmp_path / "measurements.tdm"
    _write_measurements(input_path, scenario)

    result = runner.invoke(
        app,
        [
            "export-measurements",
            str(input_path),
            "--format",
            "tdm",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    loaded = load_measurements(output_path, expected_scenario_id=scenario.scenario_id)
    expected = load_measurements(input_path, expected_scenario_id=scenario.scenario_id)
    assert len(loaded) == len(expected)
    assert sorted(
        [
            (
                record.measurement_type,
                record.epoch,
                record.observer,
                record.observed_object,
                record.value,
                record.sigma,
                record.units,
            )
            for record in loaded
        ]
    ) == sorted(
        [
            (
                record.measurement_type,
                record.epoch,
                record.observer,
                record.observed_object,
                record.value,
                record.sigma,
                record.units,
            )
            for record in expected
        ]
    )


def test_export_measurements_command_reports_invalid_format(tmp_path: Path) -> None:
    input_path = tmp_path / "measurements.json"
    _write_measurements(input_path, _observable_scenario())

    result = runner.invoke(
        app,
        [
            "export-measurements",
            str(input_path),
            "--format",
            "unsupported",
            "--output",
            str(tmp_path / "measurements.out"),
        ],
    )

    assert result.exit_code == 2
    assert "Unsupported measurement format" in result.stderr


def test_export_measurements_command_reports_output_write_error(tmp_path: Path) -> None:
    input_path = tmp_path / "measurements.json"
    _write_measurements(input_path, _observable_scenario())

    result = runner.invoke(
        app,
        [
            "export-measurements",
            str(input_path),
            "--format",
            "csv",
            "--output",
            str(tmp_path / "missing" / "measurements.csv"),
        ],
    )

    assert result.exit_code == 2
    assert "could not write measurements" in result.stderr


def test_propagate_command_reports_invalid_scenario(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    result = runner.invoke(app, ["propagate", str(scenario)])

    assert result.exit_code == 2
    assert "is invalid" in result.stderr


def test_propagate_command_reports_unavailable_orekit_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_orekit(_scenario: Scenario) -> object:
        raise UnsupportedBackendError("Orekit backend unavailable: install astro-suite[orekit]")

    monkeypatch.setattr("astro_cli.main.propagate_orekit", fail_orekit)

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
    assert "Orekit backend unavailable" in result.stderr


def test_propagate_command_writes_orekit_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "orekit.json"

    def fake_orekit(scenario: Scenario) -> object:
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": "orekit"})

    monkeypatch.setattr("astro_cli.main.propagate_orekit", fake_orekit)

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "orekit"


def test_launch_command_reports_unsupported_backend(tmp_path: Path) -> None:
    scenario_path = tmp_path / "launch.yaml"
    output = tmp_path / "launch.json"
    _write_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "launch",
            str(scenario_path),
            "--backend",
            "rocketpy",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "unsupported launch backend: rocketpy" in result.stderr


def test_handoff_launch_command_reports_unsupported_gravity(tmp_path: Path) -> None:
    launch_output = tmp_path / "launch.json"
    _write_launch_trajectory(launch_output)

    result = runner.invoke(
        app,
        [
            "handoff-launch",
            str(launch_output),
            "--output",
            str(tmp_path / "insertion.yaml"),
            "--gravity",
            "unsupported",
        ],
    )

    assert result.exit_code == 2
    assert "unsupported handoff gravity: unsupported" in result.stderr


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


def test_launch_command_reports_output_write_error(tmp_path: Path) -> None:
    scenario_path = tmp_path / "launch.yaml"
    output = tmp_path / "missing" / "launch.json"
    _write_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "launch",
            str(scenario_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write launch trajectory" in result.stderr
    assert str(output) in result.stderr


def test_handoff_launch_command_reports_output_write_error(tmp_path: Path) -> None:
    launch_output = tmp_path / "launch.json"
    output = tmp_path / "missing" / "insertion.yaml"
    _write_launch_trajectory(launch_output)

    result = runner.invoke(
        app,
        [
            "handoff-launch",
            str(launch_output),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write orbit scenario" in result.stderr
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


def test_launch_command_reports_invalid_scenario(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "launch",
            str(scenario),
            "--output",
            str(tmp_path / "launch.json"),
        ],
    )

    assert result.exit_code == 2
    assert "Launch scenario file" in result.stderr


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


def test_estimate_measurements_command_reports_invalid_measurement_file(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, _observable_scenario())
    measurements_path.write_text('{"scenario_id": "wrong", "measurements": []}', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "scenario_id" in result.stderr


def test_estimate_measurements_command_reports_output_write_error(tmp_path: Path) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "missing" / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements(measurements_path, truth_scenario)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "could not write estimate" in result.stderr
    assert str(output) in result.stderr


def test_orekit_smoke_command_reports_available_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = OrekitSmokeResult(
        available=True,
        wrapper="orekit_jpype",
        version="13.1.0",
        message="Orekit JPype VM, EME2000 frame, and UTC time scale are available.",
    )
    monkeypatch.setattr("astro_cli.main.run_orekit_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["orekit-smoke"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_orekit_smoke_command_exits_nonzero_when_wrapper_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = OrekitSmokeResult(
        available=False,
        wrapper="orekit_jpype",
        version=None,
        message="Orekit JPype wrapper is not installed.",
    )
    monkeypatch.setattr("astro_cli.main.run_orekit_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["orekit-smoke"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()
