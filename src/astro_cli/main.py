from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from astro_assistant.executor import WorkflowExecutor
from astro_assistant.planner import DeterministicPlanner
from astro_backends.dymos import (
    optimize_launch_dymos,
    run_dymos_multistage_pitch_program_optimization,
    run_dymos_pitch_program_optimization,
    run_dymos_smoke,
)
from astro_backends.jax import (
    research_estimate_jax,
    research_od_sensitivity_jax,
    research_propagate_jax,
    run_jax_smoke,
)
from astro_backends.orekit import estimate_orekit_native, run_orekit_smoke
from astro_backends.rocketpy import run_rocketpy_smoke
from astro_backends.tudat import compare_tudat_campaign, compare_tudat_to_reference, run_tudat_smoke
from astro_core.eop import load_iers_finals_eop
from astro_core.errors import (
    InvalidMeasurementFileError,
    InvalidScenarioError,
    NumericalConvergenceError,
    UnsupportedBackendError,
)
from astro_core.io import load_scenario, load_trajectory
from astro_core.models import CartesianState, ForceModelName, GroundStation, Scenario, Trajectory
from astro_dynamics.attitude import RigidBodyAttitudeConfig, propagate_rigid_body_attitude
from astro_dynamics.backends import propagate_with_backend
from astro_dynamics.conjunction import (
    ConjunctionScreeningResult,
    assess_conjunction_screening,
    screen_conjunction,
)
from astro_dynamics.ephemeris import (
    dump_trajectory_aem,
    dump_trajectory_ephemeris_csv,
    dump_trajectory_oem,
    dump_trajectory_opm,
    load_trajectory_aem,
    load_trajectory_oem,
    load_trajectory_opm,
)
from astro_dynamics.monte_carlo import run_initial_state_monte_carlo
from astro_launch.backends import propagate_launch_with_backend
from astro_launch.handoff import launch_trajectory_to_orbit_scenario
from astro_launch.io import load_launch_scenario, load_launch_trajectory, load_tuned_launch_report
from astro_launch.models import LaunchScenario, LaunchTrajectory, TunedLaunchReport
from astro_launch.reporting import (
    compare_tuned_launch_reports,
    generate_tuned_launch_report,
    generate_tuned_launch_report_batch,
)
from astro_launch.targeting import sweep_pitch_program, tune_pitch_program
from astro_od.calibration import (
    generate_dsn_calibration_product,
    generate_dsn_calibration_product_from_measurements,
    generate_station_calibration_product_from_measurements,
)
from astro_od.dsn import (
    load_dsn_binary_tracking_measurements,
    load_dsn_kvn_tracking_measurements,
    load_dsn_tracking_measurements,
)
from astro_od.estimation import estimate_initial_state
from astro_od.io import (
    dump_measurements_csv,
    dump_measurements_json,
    dump_measurements_tdm,
    load_measurement_product,
    load_measurements,
    resolve_measurement_format,
)
from astro_od.measurements import generate_synthetic_measurements

app = typer.Typer(help="Astro Suite flight dynamics workflows.")

INITIAL_GUESS_POSITION_DELTA_KM = (1.0, -0.8, 0.6)
INITIAL_GUESS_VELOCITY_DELTA_KM_S = (0.0005, -0.001, 0.0008)
DEMO_GROUND_STATION_CANDIDATES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("demo-y-axis-eci", (0.0, 6378.1363, 0.0)),
    ("demo-x-axis-eci", (6378.1363, 0.0, 0.0)),
)


def _load_scenario_or_exit(scenario_path: Path) -> Scenario:
    try:
        return load_scenario(scenario_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _load_trajectory_or_exit(trajectory_path: Path) -> Trajectory:
    try:
        return load_trajectory(trajectory_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _load_launch_scenario_or_exit(scenario_path: Path) -> LaunchScenario:
    try:
        return load_launch_scenario(scenario_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _load_launch_trajectory_or_exit(trajectory_path: Path) -> LaunchTrajectory:
    try:
        return load_launch_trajectory(trajectory_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _load_tuned_launch_report_or_exit(report_path: Path) -> TunedLaunchReport:
    try:
        return load_tuned_launch_report(report_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _write_text_or_exit(output: Path, payload: str, product_name: str) -> None:
    try:
        output.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        typer.echo(f"could not write {product_name} {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _parse_pitch_deg_values_or_exit(pitch_deg_values: str) -> list[float]:
    raw_values = [raw_value.strip() for raw_value in pitch_deg_values.split(",")]
    if not pitch_deg_values.strip() or any(raw_value == "" for raw_value in raw_values):
        typer.echo("pitch-deg-values must be comma-separated numbers", err=True)
        raise typer.Exit(code=2)
    try:
        return [float(raw_value) for raw_value in raw_values]
    except ValueError as exc:
        typer.echo("pitch-deg-values must be comma-separated numbers", err=True)
        raise typer.Exit(code=2) from exc


def _parse_point_indices_or_exit(point_indices: str) -> tuple[int, int]:
    raw_values = [raw_value.strip() for raw_value in point_indices.split(",")]
    if (
        not point_indices.strip()
        or len(raw_values) != 2
        or any(raw_value == "" for raw_value in raw_values)
    ):
        typer.echo("point-indices must be two comma-separated integers", err=True)
        raise typer.Exit(code=2)
    try:
        parsed_values = tuple(int(raw_value) for raw_value in raw_values)
    except ValueError as exc:
        typer.echo("point-indices must be two comma-separated integers", err=True)
        raise typer.Exit(code=2) from exc
    return parsed_values[0], parsed_values[1]


def _parse_iterations_values_or_exit(iterations_values: str) -> list[int]:
    raw_values = [raw_value.strip() for raw_value in iterations_values.split(",")]
    if not iterations_values.strip() or any(raw_value == "" for raw_value in raw_values):
        typer.echo("iterations-values must be comma-separated positive integers", err=True)
        raise typer.Exit(code=2)
    try:
        parsed_values = [int(raw_value) for raw_value in raw_values]
    except ValueError as exc:
        typer.echo("iterations-values must be comma-separated positive integers", err=True)
        raise typer.Exit(code=2) from exc
    if any(value <= 0 for value in parsed_values):
        typer.echo("iterations-values must be comma-separated positive integers", err=True)
        raise typer.Exit(code=2)
    return parsed_values


def _offset_cartesian_state(
    state: CartesianState,
    position_delta_km: tuple[float, float, float],
    velocity_delta_km_s: tuple[float, float, float],
) -> CartesianState:
    return CartesianState(
        position_km=(
            state.position_km[0] + position_delta_km[0],
            state.position_km[1] + position_delta_km[1],
            state.position_km[2] + position_delta_km[2],
        ),
        velocity_km_s=(
            state.velocity_km_s[0] + velocity_delta_km_s[0],
            state.velocity_km_s[1] + velocity_delta_km_s[1],
            state.velocity_km_s[2] + velocity_delta_km_s[2],
        ),
    )


def _with_estimation_demo_geometry(scenario: Scenario) -> tuple[Scenario, list[GroundStation]]:
    if len(scenario.ground_stations) >= 2:
        return scenario, []

    stations = list(scenario.ground_stations)
    station_names = {station.name for station in stations}
    added_stations: list[GroundStation] = []

    for station_name, station_position in DEMO_GROUND_STATION_CANDIDATES:
        if len(stations) >= 2:
            break
        if station_name in station_names:
            continue
        station = GroundStation(
            name=station_name,
            position_eci_km=station_position,
            frame=scenario.initial_state.frame,
            elevation_mask_deg=0.0,
        )
        stations.append(station)
        station_names.add(station_name)
        added_stations.append(station)

    return scenario.model_copy(update={"ground_stations": stations}), added_stations


def _with_estimation_demo_initial_guess(scenario: Scenario) -> Scenario:
    perturbed_cartesian = _offset_cartesian_state(
        scenario.initial_state.cartesian,
        position_delta_km=INITIAL_GUESS_POSITION_DELTA_KM,
        velocity_delta_km_s=INITIAL_GUESS_VELOCITY_DELTA_KM_S,
    )
    perturbed_initial_state = scenario.initial_state.model_copy(
        update={"cartesian": perturbed_cartesian}
    )
    return scenario.model_copy(update={"initial_state": perturbed_initial_state})


def _with_estimation_demo_metadata(
    result_metadata: dict[str, object],
    *,
    source_scenario: Scenario,
    truth_scenario: Scenario,
    demo_added_ground_stations: list[GroundStation],
    measurement_count: int,
) -> dict[str, object]:
    added_station_payloads = [
        station.model_dump(mode="json", exclude_none=True)
        for station in demo_added_ground_stations
    ]
    return {
        **result_metadata,
        "workflow": "local_synthetic_demo",
        "source_scenario_id": source_scenario.scenario_id,
        "source_ground_station_count": len(source_scenario.ground_stations),
        "truth_ground_station_count": len(truth_scenario.ground_stations),
        "demo_added_ground_stations": [station.name for station in demo_added_ground_stations],
        "demo_added_ground_station_geometry": added_station_payloads,
        "initial_guess_position_delta_km": list(INITIAL_GUESS_POSITION_DELTA_KM),
        "initial_guess_velocity_delta_km_s": list(INITIAL_GUESS_VELOCITY_DELTA_KM_S),
        "measurement_count": measurement_count,
    }


def _with_measurement_file_metadata(
    result_metadata: dict[str, object],
    *,
    scenario: Scenario,
    measurement_file: Path,
    measurement_format: str,
    measurement_count: int,
    estimator_mode: str = "suite",
) -> dict[str, object]:
    workflow = (
        "orekit_native_measurement_file"
        if estimator_mode == "orekit-native"
        else "local_measurement_file"
    )
    return {
        **result_metadata,
        "workflow": workflow,
        "estimator_mode": estimator_mode,
        "source_scenario_id": scenario.scenario_id,
        "measurement_file": str(measurement_file),
        "measurement_format": measurement_format,
        "measurement_count": measurement_count,
    }


@app.command("ask")
def ask_assistant(
    prompt: Annotated[str, typer.Argument(help="Natural language assistant request.")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Compile and print the workflow without executing it."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Evaluate the workflow for execution."),
    ] = False,
    approved: Annotated[
        bool,
        typer.Option("--approved", help="Approve execution of artifact-writing workflows."),
    ] = False,
    trace_output: Annotated[
        Path | None,
        typer.Option("--trace-output", help="Write the assistant trace JSON to a file."),
    ] = None,
) -> None:
    planner = DeterministicPlanner()
    try:
        plan = planner.plan(prompt)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    effective_dry_run = not execute
    trace = WorkflowExecutor().run(
        plan,
        dry_run=effective_dry_run,
        approved=approved,
        cwd=str(Path.cwd()),
    )
    payload = trace.model_dump_json(indent=2)
    if trace_output is not None:
        try:
            trace_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            typer.echo(f"could not write assistant trace {trace_output}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        _write_text_or_exit(trace_output, payload, "assistant trace")
    typer.echo(payload)
    if not trace.verification.passed:
        for diagnostic in trace.verification.diagnostics:
            typer.echo(diagnostic.message, err=True)
    if trace.warnings:
        for warning in trace.warnings:
            typer.echo(warning, err=True)
    if not trace.verification.passed or trace.warnings:
        raise typer.Exit(code=2)


@app.command()
def validate(scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a scenario file."""
    scenario = _load_scenario_or_exit(scenario_path)

    typer.echo(f"valid scenario: {scenario.scenario_id}")


@app.command("import-earth-orientation")
def import_earth_orientation(
    eop_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    eop_format: Annotated[
        str,
        typer.Option("--format", help="Earth-orientation input format: iers-finals."),
    ] = "iers-finals",
    source: Annotated[
        str,
        typer.Option("--source", help="Source label to store on the EOP table."),
    ] = "iers-finals",
) -> None:
    """Import an Earth-orientation table into suite JSON."""
    try:
        if eop_format.lower() != "iers-finals":
            raise ValueError(f"unsupported Earth-orientation format: {eop_format}")
        earth_orientation = load_iers_finals_eop(eop_path, source=source)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(
        output,
        earth_orientation.model_dump_json(indent=2),
        "earth orientation",
    )
    typer.echo(f"wrote earth orientation: {output}")


@app.command("orekit-smoke")
def orekit_smoke() -> None:
    """Run the optional Orekit JPype wrapper smoke gate."""
    result = run_orekit_smoke()

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)


@app.command("rocketpy-smoke")
def rocketpy_smoke() -> None:
    """Run the optional RocketPy package smoke gate."""
    result = run_rocketpy_smoke()

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)


@app.command("dymos-smoke")
def dymos_smoke() -> None:
    """Run the optional Dymos/OpenMDAO package smoke gate."""
    result = run_dymos_smoke()

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)


@app.command("tudat-smoke")
def tudat_smoke() -> None:
    """Run the optional TudatPy package smoke gate."""
    result = run_tudat_smoke()

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)


@app.command("jax-smoke")
def jax_smoke() -> None:
    """Run the optional JAX/JAXLIB package smoke gate."""
    result = run_jax_smoke()

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)


@app.command()
def propagate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    backend: Annotated[str, typer.Option()] = "local",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Propagate a scenario and write a trajectory product."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        trajectory = propagate_with_backend(scenario, backend)
    except UnsupportedBackendError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = trajectory.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        _write_text_or_exit(output, payload, "trajectory")
        typer.echo(f"wrote trajectory: {output}")


@app.command("compare-tudat-reference")
def compare_tudat_reference(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    reference_backend: Annotated[str, typer.Option()] = "local",
    position_tolerance_km: Annotated[float, typer.Option()] = 1.0e-3,
    velocity_tolerance_km_s: Annotated[float, typer.Option()] = 1.0e-6,
) -> None:
    """Compare Tudat propagation against a reference backend and write tolerance metrics."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        comparison = compare_tudat_to_reference(
            scenario,
            reference_backend=reference_backend,
            position_tolerance_km=position_tolerance_km,
            velocity_tolerance_km_s=velocity_tolerance_km_s,
        )
    except (UnsupportedBackendError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(
        output,
        comparison.model_dump_json(indent=2),
        "Tudat reference comparison",
    )
    typer.echo(f"wrote Tudat reference comparison: {output}")


@app.command("compare-tudat-campaign")
def compare_tudat_reference_campaign(
    scenario_paths: Annotated[list[Path], typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    reference_backend: Annotated[str, typer.Option()] = "local",
    position_tolerance_km: Annotated[float, typer.Option()] = 1.0e-3,
    velocity_tolerance_km_s: Annotated[float, typer.Option()] = 1.0e-6,
) -> None:
    """Run a calibrated Tudat comparison campaign across multiple scenarios."""
    scenarios = [_load_scenario_or_exit(scenario_path) for scenario_path in scenario_paths]
    try:
        campaign = compare_tudat_campaign(
            scenarios,
            reference_backend=reference_backend,
            position_tolerance_km=position_tolerance_km,
            velocity_tolerance_km_s=velocity_tolerance_km_s,
        )
    except (UnsupportedBackendError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(
        output,
        campaign.model_dump_json(indent=2),
        "Tudat comparison campaign",
    )
    typer.echo(f"wrote Tudat comparison campaign: {output}")


@app.command("export-trajectory")
def export_trajectory(
    trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    trajectory_format: Annotated[
        str,
        typer.Option("--format", help="Output trajectory format: csv, oem, opm, or aem."),
    ] = "csv",
) -> None:
    """Export a trajectory product to an ephemeris table."""
    trajectory = _load_trajectory_or_exit(trajectory_path)
    normalized_format = trajectory_format.lower()
    if normalized_format == "csv":
        payload = dump_trajectory_ephemeris_csv(trajectory)
    elif normalized_format == "oem":
        payload = dump_trajectory_oem(trajectory)
    elif normalized_format == "opm":
        payload = dump_trajectory_opm(trajectory)
    elif normalized_format == "aem":
        try:
            payload = dump_trajectory_aem(trajectory)
        except ValueError as exc:
            typer.echo(f"invalid AEM trajectory {trajectory_path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        typer.echo(f"unsupported trajectory export format: {trajectory_format}", err=True)
        raise typer.Exit(code=2)

    _write_text_or_exit(output, payload, "trajectory")
    typer.echo(f"wrote trajectory: {output}")


@app.command("import-trajectory")
def import_trajectory(
    trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    scenario_path: Annotated[Path, typer.Option("--scenario", exists=True, readable=True)],
    trajectory_format: Annotated[
        str,
        typer.Option("--format", help="Input trajectory format: oem, opm, or aem."),
    ] = "oem",
    state_trajectory_path: Annotated[
        Path | None,
        typer.Option(
            "--state-trajectory",
            exists=True,
            readable=True,
            help="Base suite trajectory JSON for AEM attitude-only import.",
        ),
    ] = None,
) -> None:
    """Import an external trajectory product into suite trajectory JSON."""
    scenario = _load_scenario_or_exit(scenario_path)
    normalized_format = trajectory_format.lower()
    try:
        payload = trajectory_path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.echo(f"could not read trajectory {trajectory_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if normalized_format == "oem":
        try:
            trajectory = load_trajectory_oem(payload, force_model=scenario.force_model)
        except ValueError as exc:
            typer.echo(f"invalid OEM trajectory {trajectory_path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    elif normalized_format == "opm":
        try:
            trajectory = load_trajectory_opm(payload, force_model=scenario.force_model)
        except ValueError as exc:
            typer.echo(f"invalid OPM trajectory {trajectory_path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    elif normalized_format == "aem":
        try:
            base_trajectory = (
                _load_trajectory_or_exit(state_trajectory_path)
                if state_trajectory_path is not None
                else propagate_with_backend(scenario, backend="local")
            )
            trajectory = load_trajectory_aem(payload, base_trajectory=base_trajectory)
        except (UnsupportedBackendError, ValueError) as exc:
            typer.echo(f"invalid AEM trajectory {trajectory_path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        typer.echo(f"unsupported trajectory import format: {trajectory_format}", err=True)
        raise typer.Exit(code=2)

    _write_text_or_exit(output, trajectory.model_dump_json(indent=2), "trajectory")
    typer.echo(f"wrote trajectory: {output}")


@app.command("screen-conjunction")
def screen_conjunction_command(
    primary_trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    secondary_trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    threshold_km: Annotated[float, typer.Option()] = 1.0,
    hard_body_radius_km: Annotated[float | None, typer.Option()] = None,
    probability_method: Annotated[str, typer.Option()] = "integrated",
) -> None:
    """Screen two time-aligned trajectory products for closest approach."""
    primary = _load_trajectory_or_exit(primary_trajectory_path)
    secondary = _load_trajectory_or_exit(secondary_trajectory_path)
    try:
        result = screen_conjunction(
            primary,
            secondary,
            threshold_km=threshold_km,
            hard_body_radius_km=hard_body_radius_km,
            probability_method=probability_method,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "conjunction screening")
    typer.echo(f"wrote conjunction screening: {output}")


@app.command("assess-conjunction")
def assess_conjunction_command(
    screening_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Assess a saved conjunction screening product for operational readiness."""
    try:
        screening = ConjunctionScreeningResult.model_validate_json(
            screening_path.read_text(encoding="utf-8")
        )
        report = assess_conjunction_screening(screening)
    except OSError as exc:
        typer.echo(f"could not read conjunction screening {screening_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(f"invalid conjunction screening {screening_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, report.model_dump_json(indent=2), "conjunction assessment")
    typer.echo(f"wrote conjunction assessment: {output}")


@app.command("propagate-attitude")
def propagate_attitude_command(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Propagate a diagonal rigid-body attitude torque profile."""
    try:
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = RigidBodyAttitudeConfig.model_validate(config_payload)
        result = propagate_rigid_body_attitude(config)
    except OSError as exc:
        typer.echo(f"could not read attitude config {config_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        typer.echo(f"invalid attitude config {config_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "attitude dynamics")
    typer.echo(f"wrote attitude dynamics: {output}")


@app.command("monte-carlo")
def monte_carlo(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    cases: Annotated[int, typer.Option()] = 16,
    position_sigma_km: Annotated[float, typer.Option()] = 0.0,
    velocity_sigma_km_s: Annotated[float, typer.Option()] = 0.0,
    seed: Annotated[int, typer.Option()] = 42,
    backend: Annotated[str, typer.Option()] = "local",
) -> None:
    """Run a seeded initial-state propagation ensemble."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        result = run_initial_state_monte_carlo(
            scenario,
            cases=cases,
            position_sigma_km=position_sigma_km,
            velocity_sigma_km_s=velocity_sigma_km_s,
            seed=seed,
            backend=backend,
        )
    except (ValueError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "monte carlo")
    typer.echo(f"wrote monte carlo: {output}")


@app.command("research-propagate")
def research_propagate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
    cases: Annotated[int, typer.Option()] = 16,
    position_sigma_km: Annotated[float, typer.Option()] = 0.0,
    velocity_sigma_km_s: Annotated[float, typer.Option()] = 0.0,
    seed: Annotated[int, typer.Option()] = 42,
    include_sensitivities: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a seeded research propagation workflow."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        if backend == "local":
            result = run_initial_state_monte_carlo(
                scenario,
                cases=cases,
                position_sigma_km=position_sigma_km,
                velocity_sigma_km_s=velocity_sigma_km_s,
                seed=seed,
                backend="local",
            )
        elif backend == "jax":
            result = research_propagate_jax(
                scenario,
                cases=cases,
                position_sigma_km=position_sigma_km,
                velocity_sigma_km_s=velocity_sigma_km_s,
                seed=seed,
                include_sensitivities=include_sensitivities,
            )
        else:
            raise UnsupportedBackendError(f"unsupported research propagation backend: {backend}")
    except (ValueError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    result = result.model_copy(
        update={"metadata": {**result.metadata, "workflow": "research_propagation"}}
    )
    _write_text_or_exit(output, result.model_dump_json(indent=2), "research propagation")
    typer.echo(f"wrote research propagation: {output}")


@app.command("research-od-sensitivity")
def research_od_sensitivity(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    measurements_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Measurement file format: auto, json, csv, or tdm."),
    ] = "auto",
    backend: Annotated[str, typer.Option()] = "jax",
) -> None:
    """Compute a research OD residual Jacobian product."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        resolved_measurement_format = resolve_measurement_format(
            measurements_path,
            measurement_format,
        )
        measurements = load_measurements(
            measurements_path,
            expected_scenario_id=scenario.scenario_id,
            measurement_format=resolved_measurement_format,
        )
        if backend != "jax":
            raise UnsupportedBackendError(
                f"research OD sensitivity backend {backend!r} is unsupported; use jax"
            )
        result = research_od_sensitivity_jax(scenario, measurements)
    except (InvalidMeasurementFileError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    result = result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "workflow": "research_od_sensitivity",
                "measurement_file": str(measurements_path),
                "measurement_format": resolved_measurement_format,
            }
        }
    )
    _write_text_or_exit(output, result.model_dump_json(indent=2), "OD sensitivity")
    typer.echo(f"wrote OD sensitivity: {output}")


@app.command("research-estimate")
def research_estimate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    measurements_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Measurement file format: auto, json, csv, or tdm."),
    ] = "auto",
    backend: Annotated[str, typer.Option()] = "jax",
    max_iterations: Annotated[int, typer.Option()] = 5,
) -> None:
    """Run a research OD estimator workflow."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        resolved_measurement_format = resolve_measurement_format(
            measurements_path,
            measurement_format,
        )
        measurements = load_measurements(
            measurements_path,
            expected_scenario_id=scenario.scenario_id,
            measurement_format=resolved_measurement_format,
        )
        if backend != "jax":
            raise UnsupportedBackendError(
                f"research estimate backend {backend!r} is unsupported; use jax"
            )
        result = research_estimate_jax(
            scenario,
            measurements,
            max_iterations=max_iterations,
        )
    except (InvalidMeasurementFileError, UnsupportedBackendError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    result = result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "workflow": "research_estimate",
                "estimator_mode": backend,
                "measurement_file": str(measurements_path),
                "measurement_format": resolved_measurement_format,
            }
        }
    )
    _write_text_or_exit(output, result.model_dump_json(indent=2), "research estimate")
    typer.echo(f"wrote research estimate: {output}")


@app.command()
def launch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    backend: Annotated[str, typer.Option()] = "local",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run a launch/ascent scenario and write a launch trajectory product."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    try:
        trajectory = propagate_launch_with_backend(scenario, backend)
    except UnsupportedBackendError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = trajectory.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        _write_text_or_exit(output, payload, "launch trajectory")
        typer.echo(f"wrote launch trajectory: {output}")


@app.command("handoff-launch")
def handoff_launch(
    launch_trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    duration_s: Annotated[float, typer.Option()] = 600.0,
    step_s: Annotated[float, typer.Option()] = 60.0,
    spacecraft_name: Annotated[str, typer.Option()] = "launch-payload",
    spacecraft_mass_kg: Annotated[float | None, typer.Option()] = None,
    area_m2: Annotated[float, typer.Option()] = 2.5,
    drag_coefficient: Annotated[float, typer.Option()] = 2.2,
    reflectivity_coefficient: Annotated[float, typer.Option()] = 1.3,
    gravity: Annotated[str, typer.Option()] = "two_body",
    scenario_id: Annotated[str | None, typer.Option()] = None,
    description: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Convert a launch trajectory product into an orbital propagation scenario."""
    trajectory = _load_launch_trajectory_or_exit(launch_trajectory_path)
    try:
        force_model = ForceModelName(gravity)
    except ValueError as exc:
        typer.echo(f"unsupported handoff gravity: {gravity}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        scenario = launch_trajectory_to_orbit_scenario(
            trajectory,
            duration_s=duration_s,
            step_s=step_s,
            spacecraft_name=spacecraft_name,
            spacecraft_mass_kg=spacecraft_mass_kg,
            area_m2=area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
            gravity=force_model,
            scenario_id=scenario_id,
            description=description,
        )
    except ValueError as exc:
        typer.echo(f"could not create orbit scenario: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    payload = yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False)
    _write_text_or_exit(output, payload.rstrip("\n"), "orbit scenario")
    typer.echo(f"wrote orbit scenario: {output}")


@app.command("sweep-launch-pitch")
def sweep_launch_pitch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    point_index: Annotated[int, typer.Option()] = 1,
    pitch_deg_values: Annotated[
        str,
        typer.Option(
            "--pitch-deg-values",
            help="Comma-separated candidate pitch angles in degrees.",
        ),
    ] = "10,20,30",
    altitude_weight: Annotated[float, typer.Option()] = 1.0,
    velocity_weight: Annotated[float, typer.Option()] = 1.0,
    radial_velocity_weight: Annotated[float, typer.Option()] = 1.0,
) -> None:
    """Sweep one launch pitch-program knot and write a targeting product."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    pitch_values = _parse_pitch_deg_values_or_exit(pitch_deg_values)
    try:
        result = sweep_pitch_program(
            scenario,
            point_index=point_index,
            pitch_values_deg=pitch_values,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
            radial_velocity_weight=radial_velocity_weight,
        )
    except ValueError as exc:
        typer.echo(f"could not sweep launch pitch: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "launch pitch sweep")
    typer.echo(f"wrote launch pitch sweep: {output}")


@app.command("tune-launch-pitch")
def tune_launch_pitch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    point_indices: Annotated[
        str,
        typer.Option(
            "--point-indices",
            help="Two comma-separated pitch-program point indices.",
        ),
    ] = "2,3",
    initial_span_deg: Annotated[float, typer.Option()] = 10.0,
    iterations: Annotated[int, typer.Option()] = 2,
    refinement_factor: Annotated[float, typer.Option()] = 0.5,
    altitude_weight: Annotated[float, typer.Option()] = 1.0,
    velocity_weight: Annotated[float, typer.Option()] = 1.0,
    radial_velocity_weight: Annotated[float, typer.Option()] = 1.0,
    tuned_scenario_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Tune two launch pitch-program knots and optionally write the tuned scenario."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    parsed_point_indices = _parse_point_indices_or_exit(point_indices)
    try:
        result = tune_pitch_program(
            scenario,
            point_indices=parsed_point_indices,
            initial_span_deg=initial_span_deg,
            iterations=iterations,
            refinement_factor=refinement_factor,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
            radial_velocity_weight=radial_velocity_weight,
        )
    except ValueError as exc:
        typer.echo(f"could not tune launch pitch: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "launch pitch tuning")
    typer.echo(f"wrote launch pitch tuning: {output}")
    if tuned_scenario_output is not None:
        tuned_scenario_payload = yaml.safe_dump(
            result.tuned_scenario.model_dump(mode="json"),
            sort_keys=False,
        )
        _write_text_or_exit(
            tuned_scenario_output,
            tuned_scenario_payload.rstrip("\n"),
            "tuned launch scenario",
        )
        typer.echo(f"wrote tuned launch scenario: {tuned_scenario_output}")


@app.command("optimize-launch")
def optimize_launch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
    point_indices: Annotated[
        str,
        typer.Option(
            "--point-indices",
            help="Two comma-separated pitch-program point indices for local optimization.",
        ),
    ] = "2,3",
    initial_span_deg: Annotated[float, typer.Option()] = 10.0,
    iterations: Annotated[int, typer.Option()] = 2,
    refinement_factor: Annotated[float, typer.Option()] = 0.5,
    altitude_weight: Annotated[float, typer.Option()] = 1.0,
    velocity_weight: Annotated[float, typer.Option()] = 1.0,
    radial_velocity_weight: Annotated[float, typer.Option()] = 1.0,
    dymos_mode: Annotated[
        str,
        typer.Option(
            "--dymos-mode",
            help=(
                "Dymos optimization mode: phase, pitch-program, "
                "or multistage-pitch-program."
            ),
        ),
    ] = "phase",
) -> None:
    """Run a launch optimization workflow and write an optimization product."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    try:
        if backend == "local":
            result = tune_pitch_program(
                scenario,
                point_indices=_parse_point_indices_or_exit(point_indices),
                initial_span_deg=initial_span_deg,
                iterations=iterations,
                refinement_factor=refinement_factor,
                altitude_weight=altitude_weight,
                velocity_weight=velocity_weight,
                radial_velocity_weight=radial_velocity_weight,
            )
        elif backend == "dymos":
            if dymos_mode == "phase":
                result = optimize_launch_dymos(scenario)
            elif dymos_mode == "pitch-program":
                result = optimize_launch_dymos(
                    scenario,
                    optimizer_runner=run_dymos_pitch_program_optimization,
                )
            elif dymos_mode == "multistage-pitch-program":
                result = optimize_launch_dymos(
                    scenario,
                    optimizer_runner=run_dymos_multistage_pitch_program_optimization,
                )
            else:
                raise UnsupportedBackendError(
                    f"unsupported Dymos launch optimization mode: {dymos_mode}"
                )
        else:
            raise UnsupportedBackendError(f"unsupported launch optimization backend: {backend}")
    except (ValueError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "launch optimization")
    typer.echo(f"wrote launch optimization: {output}")


@app.command("report-tuned-launch")
def report_tuned_launch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    point_indices: Annotated[
        str,
        typer.Option(
            "--point-indices",
            help="Two comma-separated pitch-program point indices.",
        ),
    ] = "2,3",
    initial_span_deg: Annotated[float, typer.Option()] = 10.0,
    iterations: Annotated[int, typer.Option()] = 2,
    refinement_factor: Annotated[float, typer.Option()] = 0.5,
    altitude_weight: Annotated[float, typer.Option()] = 1.0,
    velocity_weight: Annotated[float, typer.Option()] = 1.0,
    radial_velocity_weight: Annotated[float, typer.Option()] = 1.0,
    orbit_duration_s: Annotated[float, typer.Option()] = 600.0,
    orbit_step_s: Annotated[float, typer.Option()] = 60.0,
    spacecraft_name: Annotated[str, typer.Option()] = "launch-payload",
    spacecraft_mass_kg: Annotated[float | None, typer.Option()] = None,
    area_m2: Annotated[float, typer.Option()] = 2.5,
    drag_coefficient: Annotated[float, typer.Option()] = 2.2,
    reflectivity_coefficient: Annotated[float, typer.Option()] = 1.3,
    gravity: Annotated[str, typer.Option()] = "two_body",
) -> None:
    """Run tune, launch, orbit handoff, and short-arc propagation in one report."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    parsed_point_indices = _parse_point_indices_or_exit(point_indices)
    try:
        force_model = ForceModelName(gravity)
    except ValueError as exc:
        typer.echo(f"unsupported report gravity: {gravity}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        report = generate_tuned_launch_report(
            scenario,
            point_indices=parsed_point_indices,
            initial_span_deg=initial_span_deg,
            iterations=iterations,
            refinement_factor=refinement_factor,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
            radial_velocity_weight=radial_velocity_weight,
            orbit_duration_s=orbit_duration_s,
            orbit_step_s=orbit_step_s,
            spacecraft_name=spacecraft_name,
            spacecraft_mass_kg=spacecraft_mass_kg,
            area_m2=area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
            gravity=force_model,
        )
    except ValueError as exc:
        typer.echo(f"could not create tuned launch report: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, report.model_dump_json(indent=2), "tuned launch report")
    typer.echo(f"wrote tuned launch report: {output}")


@app.command("batch-report-tuned-launch")
def batch_report_tuned_launch(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    point_indices: Annotated[
        str,
        typer.Option(
            "--point-indices",
            help="Two comma-separated pitch-program point indices.",
        ),
    ] = "2,3",
    iterations_values: Annotated[
        str,
        typer.Option(
            "--iterations-values",
            help="Comma-separated positive iteration counts to run and rank.",
        ),
    ] = "1,2,3",
    initial_span_deg: Annotated[float, typer.Option()] = 10.0,
    refinement_factor: Annotated[float, typer.Option()] = 0.5,
    altitude_weight: Annotated[float, typer.Option()] = 1.0,
    velocity_weight: Annotated[float, typer.Option()] = 1.0,
    radial_velocity_weight: Annotated[float, typer.Option()] = 1.0,
    orbit_duration_s: Annotated[float, typer.Option()] = 600.0,
    orbit_step_s: Annotated[float, typer.Option()] = 60.0,
    spacecraft_name: Annotated[str, typer.Option()] = "launch-payload",
    spacecraft_mass_kg: Annotated[float | None, typer.Option()] = None,
    area_m2: Annotated[float, typer.Option()] = 2.5,
    drag_coefficient: Annotated[float, typer.Option()] = 2.2,
    reflectivity_coefficient: Annotated[float, typer.Option()] = 1.3,
    gravity: Annotated[str, typer.Option()] = "two_body",
) -> None:
    """Run multiple tuned launch reports and rank them by normalized target error."""
    scenario = _load_launch_scenario_or_exit(scenario_path)
    parsed_point_indices = _parse_point_indices_or_exit(point_indices)
    parsed_iterations_values = _parse_iterations_values_or_exit(iterations_values)
    try:
        force_model = ForceModelName(gravity)
    except ValueError as exc:
        typer.echo(f"unsupported report gravity: {gravity}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        batch = generate_tuned_launch_report_batch(
            scenario,
            point_indices=parsed_point_indices,
            iterations_values=parsed_iterations_values,
            initial_span_deg=initial_span_deg,
            refinement_factor=refinement_factor,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
            radial_velocity_weight=radial_velocity_weight,
            orbit_duration_s=orbit_duration_s,
            orbit_step_s=orbit_step_s,
            spacecraft_name=spacecraft_name,
            spacecraft_mass_kg=spacecraft_mass_kg,
            area_m2=area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
            gravity=force_model,
        )
    except ValueError as exc:
        typer.echo(f"could not create tuned launch report batch: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, batch.model_dump_json(indent=2), "tuned launch report batch")
    typer.echo(f"wrote tuned launch report batch: {output}")


@app.command("compare-tuned-launch-reports")
def compare_tuned_launch_report_products(
    baseline_report_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    candidate_report_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Compare two tuned launch report products and write metric deltas."""
    baseline_report = _load_tuned_launch_report_or_exit(baseline_report_path)
    candidate_report = _load_tuned_launch_report_or_exit(candidate_report_path)
    comparison = compare_tuned_launch_reports(baseline_report, candidate_report)

    _write_text_or_exit(
        output,
        comparison.model_dump_json(indent=2),
        "tuned launch report comparison",
    )
    typer.echo(f"wrote tuned launch report comparison: {output}")


@app.command()
def synth_measurements(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
) -> None:
    """Generate synthetic measurements for a propagated scenario."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        trajectory = propagate_with_backend(scenario, backend)
    except UnsupportedBackendError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    measurements = generate_synthetic_measurements(scenario, trajectory)
    payload = json.dumps(
        {
            "scenario_id": scenario.scenario_id,
            "measurements": [record.model_dump(mode="json") for record in measurements],
        },
        indent=2,
    )

    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command("dsn-calibration")
def dsn_calibration(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
    measurements_path: Annotated[
        Path | None,
        typer.Option("--measurements", exists=True, readable=True),
    ] = None,
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Input measurement format: auto, json, csv, or tdm."),
    ] = "auto",
) -> None:
    """Generate a DSN-style radiometric media calibration summary product."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        if measurements_path is None:
            trajectory = propagate_with_backend(scenario, backend)
            product = generate_dsn_calibration_product(scenario, trajectory)
        else:
            measurement_format_name = resolve_measurement_format(
                measurements_path,
                measurement_format,
            )
            measurements = load_measurements(
                measurements_path,
                expected_scenario_id=scenario.scenario_id,
                measurement_format=measurement_format_name,
            )
            product = generate_dsn_calibration_product_from_measurements(
                scenario.scenario_id,
                measurements,
                station_count=len(scenario.ground_stations),
                metadata={
                    "measurement_file": str(measurements_path),
                    "measurement_format": measurement_format_name,
                    "spacecraft": scenario.spacecraft.name,
                    "ground_stations": [station.name for station in scenario.ground_stations],
                },
            )
    except (InvalidMeasurementFileError, UnsupportedBackendError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, product.model_dump_json(indent=2), "DSN calibration")
    typer.echo(f"wrote DSN calibration: {output}")


@app.command("import-dsn-tracking")
def import_dsn_tracking(
    tracking_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Import normalized DSN ODF/TNF-style tracking rows into suite measurements."""
    try:
        product = load_dsn_tracking_measurements(tracking_path)
    except InvalidMeasurementFileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = json.dumps(
        {
            "scenario_id": product.scenario_id,
            "metadata": product.metadata or {},
            "measurements": [record.model_dump(mode="json") for record in product.measurements],
        },
        indent=2,
    )
    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command("import-dsn-binary-tracking")
def import_dsn_binary_tracking(
    tracking_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Import ASTRODSN1 binary DSN tracking bridge records into suite measurements."""
    try:
        product = load_dsn_binary_tracking_measurements(tracking_path)
    except InvalidMeasurementFileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = json.dumps(
        {
            "scenario_id": product.scenario_id,
            "metadata": product.metadata or {},
            "measurements": [record.model_dump(mode="json") for record in product.measurements],
        },
        indent=2,
    )
    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command("import-dsn-kvn-tracking")
def import_dsn_kvn_tracking(
    tracking_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Import strict DSN ODF/TNF KVN-style tracking text decks into suite measurements."""
    try:
        product = load_dsn_kvn_tracking_measurements(tracking_path)
    except InvalidMeasurementFileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = json.dumps(
        {
            "scenario_id": product.scenario_id,
            "metadata": product.metadata or {},
            "measurements": [record.model_dump(mode="json") for record in product.measurements],
        },
        indent=2,
    )
    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command("station-calibration")
def station_calibration(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    measurements_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Input measurement format: auto, json, csv, or tdm."),
    ] = "auto",
) -> None:
    """Estimate per-station measurement biases from truth-tagged measurement records."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        measurement_format_name = resolve_measurement_format(
            measurements_path,
            measurement_format,
        )
        measurements = load_measurements(
            measurements_path,
            expected_scenario_id=scenario.scenario_id,
            measurement_format=measurement_format_name,
        )
        product = generate_station_calibration_product_from_measurements(
            scenario.scenario_id,
            measurements,
            metadata={
                "measurement_file": str(measurements_path),
                "measurement_format": measurement_format_name,
                "spacecraft": scenario.spacecraft.name,
                "ground_stations": [station.name for station in scenario.ground_stations],
            },
        )
    except (InvalidMeasurementFileError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, product.model_dump_json(indent=2), "station calibration")
    typer.echo(f"wrote station calibration: {output}")


@app.command("export-measurements")
def export_measurements(
    measurements_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Output measurement format: auto, json, csv, or tdm."),
    ] = "auto",
) -> None:
    """Export suite JSON measurements to JSON, CSV, or TDM."""
    try:
        product = load_measurement_product(measurements_path)
        resolved_measurement_format = resolve_measurement_format(output, measurement_format)
        if resolved_measurement_format == "csv":
            payload = dump_measurements_csv(product.scenario_id, product.measurements)
        elif resolved_measurement_format == "tdm":
            payload = dump_measurements_tdm(product.scenario_id, product.measurements)
        else:
            payload = dump_measurements_json(product.scenario_id, product.measurements)
    except InvalidMeasurementFileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command()
def estimate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
) -> None:
    """Run a synthetic orbit-determination workflow."""
    source_scenario = _load_scenario_or_exit(scenario_path)
    truth_scenario, added_station_names = _with_estimation_demo_geometry(source_scenario)
    try:
        truth_trajectory = propagate_with_backend(truth_scenario, backend)
    except UnsupportedBackendError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)
    estimate_scenario = _with_estimation_demo_initial_guess(truth_scenario)

    try:
        result = estimate_initial_state(estimate_scenario, measurements, backend=backend)
    except (NumericalConvergenceError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    result = result.model_copy(
        update={
            "metadata": _with_estimation_demo_metadata(
                result.metadata,
                source_scenario=source_scenario,
                truth_scenario=truth_scenario,
                demo_added_ground_stations=added_station_names,
                measurement_count=len(measurements),
            )
        }
    )
    _write_text_or_exit(output, result.model_dump_json(indent=2), "estimate")
    typer.echo(f"wrote estimate: {output}")


@app.command("estimate-measurements")
def estimate_measurements(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    measurements_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    measurement_format: Annotated[
        str,
        typer.Option("--format", help="Measurement file format: auto, json, csv, or tdm."),
    ] = "auto",
    backend: Annotated[str, typer.Option()] = "local",
    estimator: Annotated[
        str,
        typer.Option("--estimator", help="Estimator mode: suite or orekit-native."),
    ] = "suite",
) -> None:
    """Run batch OD from an explicit measurement file."""
    scenario = _load_scenario_or_exit(scenario_path)
    try:
        estimator_mode = estimator.lower()
        if estimator_mode not in {"suite", "orekit-native"}:
            raise UnsupportedBackendError(
                f"Unsupported estimator mode {estimator!r}; use suite or orekit-native"
            )
        resolved_measurement_format = resolve_measurement_format(
            measurements_path,
            measurement_format,
        )
        measurements = load_measurements(
            measurements_path,
            expected_scenario_id=scenario.scenario_id,
            measurement_format=resolved_measurement_format,
        )
        if estimator_mode == "orekit-native":
            result = estimate_orekit_native(scenario, measurements)
        else:
            result = estimate_initial_state(scenario, measurements, backend=backend)
    except (InvalidMeasurementFileError, NumericalConvergenceError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    result = result.model_copy(
        update={
            "metadata": _with_measurement_file_metadata(
                result.metadata,
                scenario=scenario,
                measurement_file=measurements_path,
                measurement_format=resolved_measurement_format,
                measurement_count=len(measurements),
                estimator_mode=estimator_mode,
            )
        }
    )
    _write_text_or_exit(output, result.model_dump_json(indent=2), "estimate")
    typer.echo(f"wrote estimate: {output}")
