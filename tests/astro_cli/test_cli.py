import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from astro_backends.dymos import DymosSmokeResult
from astro_backends.jax import JaxSmokeResult
from astro_backends.orekit import OrekitSmokeResult
from astro_backends.rocketpy import RocketPySmokeResult
from astro_backends.tudat import (
    TudatReferenceComparison,
    TudatReferenceComparisonCampaign,
    TudatSmokeResult,
)
from astro_cli.main import app
from astro_core.errors import NumericalConvergenceError, UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import CartesianState, MeasurementRecord, MeasurementType, Scenario
from astro_dynamics.conjunction import screen_conjunction
from astro_dynamics.ephemeris import dump_trajectory_oem
from astro_dynamics.local import propagate_local
from astro_dynamics.monte_carlo import run_initial_state_monte_carlo
from astro_launch.io import load_launch_scenario
from astro_launch.local import propagate_launch_local
from astro_launch.models import LaunchScenario, LaunchTrajectory
from astro_launch.reporting import generate_tuned_launch_report
from astro_launch.targeting import tune_pitch_program
from astro_od.estimation import estimate_initial_state
from astro_od.io import dump_measurements_tdm, load_measurements
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


def _write_orbit_trajectory(path: Path) -> None:
    trajectory = propagate_local(load_scenario(Path("examples/scenarios/leo_two_body.yaml")))
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


def test_import_earth_orientation_command_reads_iers_finals(tmp_path: Path) -> None:
    output = tmp_path / "eop.json"

    result = runner.invoke(
        app,
        [
            "import-earth-orientation",
            "examples/eop/finals2000A_excerpt.txt",
            "--format",
            "iers-finals",
            "--source",
            "cli-finals2000A",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote earth orientation" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"] == "cli-finals2000A"
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["ut1_minus_utc_s"] == 0.073


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


def test_export_trajectory_command_writes_csv(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    output = tmp_path / "trajectory.csv"
    _write_orbit_trajectory(trajectory_path)

    result = runner.invoke(
        app,
        [
            "export-trajectory",
            str(trajectory_path),
            "--format",
            "csv",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote trajectory" in result.stdout
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 11
    assert rows[0]["scenario_id"] == "leo-two-body"
    assert rows[0]["backend"] == "local"
    assert rows[0]["x_km"] == "7000.0"


def test_screen_conjunction_command_writes_json(tmp_path: Path) -> None:
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    output = tmp_path / "conjunction.json"
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.5, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    primary_path.write_text(primary.model_dump_json(indent=2), encoding="utf-8")
    secondary_path.write_text(secondary.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "screen-conjunction",
            str(primary_path),
            str(secondary_path),
            "--threshold-km",
            "1.0",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote conjunction screening" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "below_threshold"
    assert payload["miss_distance_km"] == 0.5
    assert payload["metadata"]["screening_model"] == "time_aligned_sample_minimum_distance"


def test_screen_conjunction_command_writes_covariance_probability(tmp_path: Path) -> None:
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    output = tmp_path / "conjunction.json"
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    primary_path.write_text(primary.model_dump_json(indent=2), encoding="utf-8")
    secondary_path.write_text(secondary.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "screen-conjunction",
            str(primary_path),
            str(secondary_path),
            "--threshold-km",
            "1.0",
            "--hard-body-radius-km",
            "0.02",
            "--probability-method",
            "density",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["probability_of_collision"] > 0.0
    assert payload["hard_body_radius_km"] == 0.02
    assert payload["metadata"]["probability_model"] == "encounter_plane_gaussian_density"


def test_screen_conjunction_command_writes_integrated_probability(tmp_path: Path) -> None:
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    output = tmp_path / "conjunction.json"
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    primary_path.write_text(primary.model_dump_json(indent=2), encoding="utf-8")
    secondary_path.write_text(secondary.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "screen-conjunction",
            str(primary_path),
            str(secondary_path),
            "--threshold-km",
            "1.0",
            "--hard-body-radius-km",
            "0.5",
            "--probability-method",
            "integrated",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["probability_of_collision"] > 0.0
    assert payload["metadata"]["probability_model"] == "encounter_plane_gaussian_integral"
    assert payload["metadata"]["probability_quadrature"] == "gauss_legendre_polar"


def test_assess_conjunction_command_writes_report(tmp_path: Path) -> None:
    screening_path = tmp_path / "conjunction.json"
    output = tmp_path / "conjunction-report.json"
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    screening = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.5,
        probability_method="integrated",
    )
    screening_path.write_text(screening.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "assess-conjunction",
            str(screening_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote conjunction assessment" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["assessment_status"] == "requires_review"
    assert payload["screening_status"] == "below_threshold"
    assert payload["has_collision_probability"] is True


def test_export_trajectory_command_writes_oem(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    output = tmp_path / "trajectory.oem"
    _write_orbit_trajectory(trajectory_path)

    result = runner.invoke(
        app,
        [
            "export-trajectory",
            str(trajectory_path),
            "--format",
            "oem",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote trajectory" in result.stdout
    payload = output.read_text(encoding="utf-8")
    assert payload.startswith("CCSDS_OEM_VERS = 2.0")
    assert "OBJECT_NAME = leo-two-body" in payload
    assert "2026-01-01T00:00:00.000000Z 7000.0 0.0 0.0 0.0 7.5 1.0" in payload


def test_propagate_attitude_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "attitude.json"

    result = runner.invoke(
        app,
        [
            "propagate-attitude",
            "examples/attitude/rigid_body_torque.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote attitude dynamics" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 3
    assert payload["samples"][-1]["angular_rate_rad_s"] == [0.0, 0.0, 1.0]
    assert payload["metadata"]["attitude_dynamics_model"] == "diagonal_rigid_body_torque"


def test_export_trajectory_command_writes_aem(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    output = tmp_path / "trajectory.aem"
    scenario = load_scenario(Path("examples/scenarios/leo_velocity_aligned_burn.yaml"))
    trajectory = propagate_local(scenario)
    trajectory_path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "export-trajectory",
            str(trajectory_path),
            "--format",
            "aem",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote trajectory" in result.stdout
    payload = output.read_text(encoding="utf-8")
    assert payload.startswith("CCSDS_AEM_VERS = 2.0")
    assert "ATTITUDE_TYPE = QUATERNION" in payload
    assert "COMMENT quaternion_order = QC Q1 Q2 Q3" in payload


def test_import_trajectory_command_reads_oem_with_scenario_context(tmp_path: Path) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    oem_path = tmp_path / "trajectory.oem"
    output = tmp_path / "trajectory.json"
    oem_path.write_text(dump_trajectory_oem(trajectory), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "import-trajectory",
            str(oem_path),
            "--format",
            "oem",
            "--scenario",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote trajectory" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert payload["backend"] == "local"
    assert payload["metadata"]["source_format"] == "ccsds_oem_kvn"
    assert len(payload["samples"]) == 11


def test_monte_carlo_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "monte_carlo.json"

    result = runner.invoke(
        app,
        [
            "monte-carlo",
            "examples/scenarios/leo_two_body.yaml",
            "--cases",
            "4",
            "--position-sigma-km",
            "0.01",
            "--velocity-sigma-km-s",
            "0.000001",
            "--seed",
            "7",
            "--backend",
            "local",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote monte carlo" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert payload["backend"] == "local"
    assert payload["seed"] == 7
    assert len(payload["cases"]) == 4
    assert {case["trajectory"]["backend"] for case in payload["cases"]} == {"local"}


def test_research_propagate_command_writes_local_json(tmp_path: Path) -> None:
    output = tmp_path / "research.json"

    result = runner.invoke(
        app,
        [
            "research-propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "local",
            "--cases",
            "2",
            "--position-sigma-km",
            "0.01",
            "--velocity-sigma-km-s",
            "0.000001",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote research propagation" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "local"
    assert payload["metadata"]["workflow"] == "research_propagation"
    assert len(payload["cases"]) == 2


def test_research_propagate_command_accepts_jax_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "research.json"
    seen_scenarios: list[str] = []

    def fake_jax(
        scenario: Scenario,
        *,
        cases: int,
        position_sigma_km: float,
        velocity_sigma_km_s: float,
        seed: int,
        include_sensitivities: bool,
    ) -> object:
        seen_scenarios.append(scenario.scenario_id)
        assert include_sensitivities is True
        result = run_initial_state_monte_carlo(
            scenario,
            cases=cases,
            position_sigma_km=position_sigma_km,
            velocity_sigma_km_s=velocity_sigma_km_s,
            seed=seed,
            backend="local",
        )
        return result.model_copy(update={"backend": "jax"})

    monkeypatch.setattr("astro_cli.main.research_propagate_jax", fake_jax)

    result = runner.invoke(
        app,
        [
            "research-propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "jax",
            "--cases",
            "2",
            "--position-sigma-km",
            "0.01",
            "--velocity-sigma-km-s",
            "0.000001",
            "--seed",
            "7",
            "--include-sensitivities",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen_scenarios == ["leo-two-body"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "jax"


def test_research_od_sensitivity_command_accepts_jax_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenario = _observable_scenario()
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "od_sensitivity.json"
    _write_measurements(measurements_path, scenario)
    seen: dict[str, object] = {}

    def fake_jax_od_sensitivity(
        candidate: Scenario,
        measurements: list[MeasurementRecord],
    ) -> object:
        seen["scenario_id"] = candidate.scenario_id
        seen["measurement_count"] = len(measurements)
        from astro_core.models import OdSensitivityResult

        return OdSensitivityResult(
            scenario_id=candidate.scenario_id,
            backend="jax",
            measurement_count=len(measurements),
            state_dimension=6,
            residuals=[0.0 for _ in measurements],
            jacobian=[[0.0 for _ in range(6)] for _ in measurements],
            metadata={"adapter": "jax"},
        )

    monkeypatch.setattr("astro_cli.main.research_od_sensitivity_jax", fake_jax_od_sensitivity)

    result = runner.invoke(
        app,
        [
            "research-od-sensitivity",
            "examples/scenarios/leo_two_station_od.yaml",
            str(measurements_path),
            "--backend",
            "jax",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == {"scenario_id": scenario.scenario_id, "measurement_count": 44}
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == scenario.scenario_id
    assert payload["backend"] == "jax"
    assert payload["measurement_count"] == 44
    assert payload["metadata"]["workflow"] == "research_od_sensitivity"


def test_research_estimate_command_accepts_jax_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "research_estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements(measurements_path, truth_scenario)
    seen: dict[str, object] = {}

    def fake_research_estimator(
        candidate: Scenario,
        measurements: list[MeasurementRecord],
        *,
        max_iterations: int,
    ) -> object:
        seen["scenario_id"] = candidate.scenario_id
        seen["measurement_count"] = len(measurements)
        seen["max_iterations"] = max_iterations
        result = estimate_initial_state(candidate, measurements)
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "backend": "jax_research_estimator",
                    "estimator": "jax_research_gauss_newton",
                }
            }
        )

    monkeypatch.setattr("astro_cli.main.research_estimate_jax", fake_research_estimator)

    result = runner.invoke(
        app,
        [
            "research-estimate",
            str(scenario_path),
            str(measurements_path),
            "--backend",
            "jax",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert seen == {
        "scenario_id": estimate_scenario.scenario_id,
        "measurement_count": 44,
        "max_iterations": 5,
    }
    assert payload["converged"] is True
    assert payload["metadata"]["workflow"] == "research_estimate"
    assert payload["metadata"]["estimator_mode"] == "jax"
    assert payload["metadata"]["backend"] == "jax_research_estimator"


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


def test_launch_command_accepts_rocketpy_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "launch.yaml"
    output = tmp_path / "rocketpy-launch.json"
    seen_backends: list[str] = []
    _write_launch_scenario(scenario_path)

    def fake_backend(scenario: LaunchScenario, backend: str) -> LaunchTrajectory:
        seen_backends.append(backend)
        return propagate_launch_local(scenario).model_copy(update={"backend": backend})

    monkeypatch.setattr("astro_cli.main.propagate_launch_with_backend", fake_backend)

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

    assert result.exit_code == 0
    assert seen_backends == ["rocketpy"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "rocketpy"


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


def test_optimize_launch_command_writes_local_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "optimization.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "optimize-launch",
            str(scenario_path),
            "--backend",
            "local",
            "--point-indices",
            "2,3",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote launch optimization" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["backend"] == "local"
    assert payload["point_indices"] == [2, 3]
    assert len(payload["iterations"]) == 1


def test_optimize_launch_command_accepts_dymos_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "optimization.json"
    seen_scenarios: list[str] = []
    _write_pitch_program_launch_scenario(scenario_path)

    def fake_dymos(scenario: LaunchScenario) -> object:
        seen_scenarios.append(scenario.scenario_id)
        return tune_pitch_program(
            scenario,
            point_indices=(2, 3),
            iterations=1,
        ).model_copy(update={"backend": "dymos"})

    monkeypatch.setattr("astro_cli.main.optimize_launch_dymos", fake_dymos)

    result = runner.invoke(
        app,
        [
            "optimize-launch",
            str(scenario_path),
            "--backend",
            "dymos",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen_scenarios == ["pitch-program-two-stage"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "dymos"


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


def test_dsn_calibration_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "dsn-calibration.json"

    result = runner.invoke(
        app,
        [
            "dsn-calibration",
            "examples/scenarios/leo_radiometric_weather_frequency.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-radiometric-weather-frequency"
    assert payload["calibration_model"] == "weather_frequency_range_delay"
    assert payload["sample_count"] == len(payload["samples"])
    assert payload["sample_count"] > 0
    assert payload["metadata"]["media_frequency_hz"] == 8.4e9
    assert "wrote DSN calibration" in result.stdout


def test_dsn_calibration_command_accepts_tdm_measurements(tmp_path: Path) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_radiometric_weather_frequency.yaml"))
    records = generate_synthetic_measurements(scenario, propagate_local(scenario))
    measurements_path = tmp_path / "radiometric-weather-frequency.tdm"
    measurements_path.write_text(
        dump_measurements_tdm(scenario.scenario_id, records),
        encoding="utf-8",
    )
    output = tmp_path / "dsn-calibration.json"

    result = runner.invoke(
        app,
        [
            "dsn-calibration",
            "examples/scenarios/leo_radiometric_weather_frequency.yaml",
            "--measurements",
            str(measurements_path),
            "--format",
            "tdm",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-radiometric-weather-frequency"
    assert payload["calibration_model"] == "weather_frequency_range_delay"
    assert payload["sample_count"] == 66
    assert payload["metadata"]["measurement_file"] == str(measurements_path)
    assert payload["metadata"]["measurement_format"] == "tdm"


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


def test_estimate_measurements_command_can_use_native_orekit_estimator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_scenario = _observable_scenario()
    estimate_scenario = _perturbed_scenario(truth_scenario)
    scenario_path = tmp_path / "estimate_scenario.yaml"
    measurements_path = tmp_path / "measurements.json"
    output = tmp_path / "estimate.json"
    _write_scenario(scenario_path, estimate_scenario)
    _write_measurements(measurements_path, truth_scenario)
    seen: dict[str, object] = {}

    def fake_native_estimator(
        scenario: Scenario,
        measurements: list[MeasurementRecord],
    ) -> object:
        seen["scenario_id"] = scenario.scenario_id
        seen["measurement_count"] = len(measurements)
        result = estimate_initial_state(scenario, measurements)
        return result.model_copy(
            update={"metadata": {**result.metadata, "backend": "orekit_batch_ls_estimator"}}
        )

    monkeypatch.setattr("astro_cli.main.estimate_orekit_native", fake_native_estimator)

    result = runner.invoke(
        app,
        [
            "estimate-measurements",
            str(scenario_path),
            str(measurements_path),
            "--estimator",
            "orekit-native",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert seen == {"scenario_id": estimate_scenario.scenario_id, "measurement_count": 44}
    assert payload["converged"] is True
    assert payload["metadata"]["workflow"] == "orekit_native_measurement_file"
    assert payload["metadata"]["estimator_mode"] == "orekit-native"
    assert payload["metadata"]["backend"] == "orekit_batch_ls_estimator"


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


def test_import_dsn_tracking_command_writes_measurement_json(tmp_path: Path) -> None:
    input_path = tmp_path / "dsn_tracking.csv"
    output_path = tmp_path / "dsn_measurements.json"
    input_path.write_text(
        "\n".join(
            [
                "scenario_id,tracking_format,observable,epoch,station,spacecraft,value,sigma,units,participant_path",
                "dsn-demo,odf,two_way_range,2026-01-01T00:00:00+00:00,DSS-14,demo-sat,12345.6,0.01,km,\"DSS-14,demo-sat,DSS-14\"",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import-dsn-tracking",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "wrote measurements" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "dsn-demo"
    assert payload["metadata"]["source_format"] == "normalized_dsn_tracking_csv"
    assert payload["measurements"][0]["measurement_type"] == "two_way_range"
    assert payload["measurements"][0]["metadata"]["dsn_tracking_format"] == "odf"


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
    def fail_backend(_scenario: Scenario, backend: str) -> object:
        assert backend == "orekit"
        raise UnsupportedBackendError("Orekit backend unavailable: install astro-suite[orekit]")

    monkeypatch.setattr("astro_cli.main.propagate_with_backend", fail_backend)

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

    def fake_backend(scenario: Scenario, backend: str) -> object:
        assert backend == "orekit"
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": "orekit"})

    monkeypatch.setattr("astro_cli.main.propagate_with_backend", fake_backend)

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


def test_propagate_command_accepts_tudat_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "tudat.json"
    seen_backends: list[str] = []

    def fake_backend(scenario: Scenario, backend: str) -> object:
        seen_backends.append(backend)
        return propagate_local(scenario).model_copy(update={"backend": backend})

    monkeypatch.setattr("astro_cli.main.propagate_with_backend", fake_backend)

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "tudat",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen_backends == ["tudat"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "tudat"


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
            "missing",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "unsupported launch backend: missing" in result.stderr


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


def test_rocketpy_smoke_command_reports_available_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = RocketPySmokeResult(
        available=True,
        package="rocketpy",
        version="1.12.1",
        message="RocketPy available.",
    )
    monkeypatch.setattr("astro_cli.main.run_rocketpy_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["rocketpy-smoke"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_rocketpy_smoke_command_exits_nonzero_when_package_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = RocketPySmokeResult(
        available=False,
        package="rocketpy",
        version=None,
        message="RocketPy unavailable.",
    )
    monkeypatch.setattr("astro_cli.main.run_rocketpy_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["rocketpy-smoke"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_dymos_smoke_command_reports_available_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = DymosSmokeResult(
        available=True,
        package="dymos",
        version="1.15.1",
        openmdao_version="3.44.0",
        message="Dymos available.",
    )
    monkeypatch.setattr("astro_cli.main.run_dymos_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["dymos-smoke"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_dymos_smoke_command_exits_nonzero_when_package_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = DymosSmokeResult(
        available=False,
        package="dymos",
        version=None,
        openmdao_version=None,
        message="Dymos unavailable.",
    )
    monkeypatch.setattr("astro_cli.main.run_dymos_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["dymos-smoke"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_tudat_smoke_command_reports_available_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = TudatSmokeResult(
        available=True,
        package="tudatpy",
        version="1.0.0",
        message="Tudat available.",
    )
    monkeypatch.setattr("astro_cli.main.run_tudat_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["tudat-smoke"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_compare_tudat_reference_command_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    output = tmp_path / "comparison.json"
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    _write_scenario(scenario_path, scenario)

    def fake_compare(
        candidate: Scenario,
        *,
        reference_backend: str,
        position_tolerance_km: float,
        velocity_tolerance_km_s: float,
    ) -> TudatReferenceComparison:
        assert candidate.scenario_id == scenario.scenario_id
        assert reference_backend == "local"
        assert position_tolerance_km == pytest.approx(0.25)
        assert velocity_tolerance_km_s == pytest.approx(0.002)
        return TudatReferenceComparison(
            scenario_id=candidate.scenario_id,
            candidate_backend="tudat",
            reference_backend=reference_backend,
            sample_count=11,
            position_tolerance_km=position_tolerance_km,
            velocity_tolerance_km_s=velocity_tolerance_km_s,
            max_position_delta_km=0.1,
            rms_position_delta_km=0.08,
            final_position_delta_km=0.07,
            max_velocity_delta_km_s=0.001,
            rms_velocity_delta_km_s=0.0008,
            final_velocity_delta_km_s=0.0007,
            passed=True,
            metadata={"tudat_runner": "native_two_body"},
        )

    monkeypatch.setattr("astro_cli.main.compare_tudat_to_reference", fake_compare)

    result = runner.invoke(
        app,
        [
            "compare-tudat-reference",
            str(scenario_path),
            "--reference-backend",
            "local",
            "--position-tolerance-km",
            "0.25",
            "--velocity-tolerance-km-s",
            "0.002",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote Tudat reference comparison" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == scenario.scenario_id
    assert payload["candidate_backend"] == "tudat"


def test_compare_tudat_campaign_command_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "scenario-a.yaml"
    second_path = tmp_path / "scenario-b.yaml"
    output = tmp_path / "campaign.json"
    first = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    second = first.model_copy(update={"scenario_id": "leo-two-body-campaign-second"})
    _write_scenario(first_path, first)
    _write_scenario(second_path, second)

    def fake_campaign(
        candidates: list[Scenario],
        *,
        reference_backend: str,
        position_tolerance_km: float,
        velocity_tolerance_km_s: float,
    ) -> TudatReferenceComparisonCampaign:
        assert [candidate.scenario_id for candidate in candidates] == [
            first.scenario_id,
            second.scenario_id,
        ]
        comparison = TudatReferenceComparison(
            scenario_id=first.scenario_id,
            candidate_backend="tudat",
            reference_backend=reference_backend,
            sample_count=11,
            position_tolerance_km=position_tolerance_km,
            velocity_tolerance_km_s=velocity_tolerance_km_s,
            max_position_delta_km=0.1,
            rms_position_delta_km=0.08,
            final_position_delta_km=0.07,
            max_velocity_delta_km_s=0.001,
            rms_velocity_delta_km_s=0.0008,
            final_velocity_delta_km_s=0.0007,
            passed=True,
            metadata={},
        )
        return TudatReferenceComparisonCampaign(
            campaign_id="tudat-reference-campaign",
            reference_backend=reference_backend,
            scenario_count=2,
            passed_count=2,
            failed_count=0,
            passed=True,
            max_position_delta_km=0.1,
            max_velocity_delta_km_s=0.001,
            comparisons=(
                comparison,
                comparison.model_copy(update={"scenario_id": second.scenario_id}),
            ),
            metadata={"workflow": "tudat_reference_comparison_campaign"},
        )

    monkeypatch.setattr("astro_cli.main.compare_tudat_campaign", fake_campaign)

    result = runner.invoke(
        app,
        [
            "compare-tudat-campaign",
            str(first_path),
            str(second_path),
            "--reference-backend",
            "local",
            "--position-tolerance-km",
            "0.25",
            "--velocity-tolerance-km-s",
            "0.002",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote Tudat comparison campaign" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_count"] == 2
    assert payload["passed"] is True
    assert [comparison["scenario_id"] for comparison in payload["comparisons"]] == [
        first.scenario_id,
        second.scenario_id,
    ]
    assert payload["reference_backend"] == "local"
    assert payload["passed"] is True


def test_tudat_smoke_command_exits_nonzero_when_package_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = TudatSmokeResult(
        available=False,
        package="tudatpy",
        version=None,
        message="Tudat unavailable.",
    )
    monkeypatch.setattr("astro_cli.main.run_tudat_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["tudat-smoke"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_jax_smoke_command_reports_available_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = JaxSmokeResult(
        available=True,
        package="jax",
        version="0.10.1",
        jaxlib_version="0.10.1",
        message="JAX available.",
    )
    monkeypatch.setattr("astro_cli.main.run_jax_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["jax-smoke"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()


def test_jax_smoke_command_exits_nonzero_when_package_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_result = JaxSmokeResult(
        available=False,
        package="jax",
        version=None,
        jaxlib_version=None,
        message="JAX unavailable.",
    )
    monkeypatch.setattr("astro_cli.main.run_jax_smoke", lambda: smoke_result)

    result = runner.invoke(app, ["jax-smoke"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == smoke_result.to_dict()
