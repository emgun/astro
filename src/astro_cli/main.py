from __future__ import annotations

import json
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import typer
import yaml

from astro_assistant.capabilities import LocalODSupportReport, classify_local_od_support
from astro_assistant.executor import WorkflowExecutor
from astro_assistant.models import VerificationDiagnostic, VerificationResult
from astro_assistant.planner import DeterministicPlanner
from astro_assistant.reports import build_workflow_report, format_workflow_report
from astro_assistant.verification import verify_plan
from astro_assurance.covariance_empirical_cli import (
    run_empirical_covariance_campaign_command,
)
from astro_assurance.covariance_validation_cli import (
    assess_covariance_validation_command,
    validate_covariance_validation_command,
    verify_covariance_validation_command,
)
from astro_assurance.errors import MissionAssuranceError
from astro_assurance.io import (
    format_mission_assurance_summary,
    load_post_launch_assurance_scenario,
    verify_mission_assurance_artifact_bundle,
    write_mission_assurance_artifact_bundle,
)
from astro_assurance.lifecycle_review_cli import (
    review_mission_lifecycle_command,
    verify_mission_lifecycle_result_command,
    verify_mission_lifecycle_review_command,
)
from astro_assurance.model_form_cli import (
    run_model_form_matrix_command,
    validate_model_form_matrix_command,
    verify_model_form_matrix_command,
)
from astro_assurance.review_cli import (
    compare_assurance_reviews_command,
    review_assurance_validation_command,
    verify_assurance_review_comparison_command,
)
from astro_assurance.runner import run_post_launch_assurance
from astro_assurance.validation_cli import (
    inspect_assurance_calibration_command,
    run_assurance_validation_command,
    validate_assurance_validation_command,
    verify_assurance_validation_command,
)
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
from astro_mission.errors import MissionLifecycleError
from astro_mission.evidence import (
    run_mission_evidence_pack,
    verify_mission_evidence_pack,
)
from astro_mission.io import (
    format_mission_lifecycle_summary,
    load_mission_lifecycle_scenario,
    write_mission_artifact_bundle,
)
from astro_mission.runner import run_mission_lifecycle
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
from astro_operator.acquisition import acquire_reasoner_behavior_recording
from astro_operator.behavior import (
    load_reasoner_behavior_corpus,
    load_reasoner_behavior_replay,
    score_reasoner_behavior_corpus,
)
from astro_operator.command_execution import (
    CommandExecutionCoordinator,
    CommandToolRegistry,
    SimulatedBurnTool,
    SQLiteCommandExecutionStore,
)
from astro_operator.conditional_campaign_io import (
    run_conditional_campaign,
    verify_conditional_campaign,
)
from astro_operator.director import MissionDesignRun, build_mission_design_run
from astro_operator.director_io import (
    capture_resolved_base_scenario,
    load_mission_design_director_spec,
    verify_mission_design_director,
    write_mission_design_manifest,
    write_mission_design_run,
)
from astro_operator.engine import run_operator
from astro_operator.errors import OperatorError
from astro_operator.evaluation import load_adversarial_corpus, score_adversarial_corpus
from astro_operator.io import (
    capture_base_scenario_evidence,
    load_mission_operator_spec,
    load_operator_replay,
    verify_operator_run,
    write_operator_run,
)
from astro_operator.knowledge import trace_baseline_justification
from astro_operator.knowledge_io import (
    publish_mission_knowledge_graph,
    verify_mission_knowledge_graph,
)
from astro_operator.lifecycle import LifecycleCandidateEvaluator, resolve_lifecycle_references
from astro_operator.models import MissionBaselineContext, OperatorActionKind
from astro_operator.openrouter import DEFAULT_OPENROUTER_MODEL, OpenRouterReasoner
from astro_operator.operational_evidence import build_operational_evidence_registry
from astro_operator.orchestration import (
    MissionOrchestrationQuery,
    evaluate_mission_orchestration,
)
from astro_operator.reasoner import ConditionalReplayReasoner
from astro_operator.recording import (
    ReasonerBehaviorRecording,
    load_reasoner_behavior_recording,
    record_reasoner_behavior_replay,
    reserve_reasoner_behavior_recording,
    score_recorded_reasoner_behavior_corpus,
    write_reasoner_behavior_recording,
)
from astro_operator.world_state import reduce_world_state
from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.handoff import trajectory_to_reentry_scenario
from astro_reentry.io import (
    format_reentry_optimization_summary,
    format_reentry_summary,
    load_reentry_scenario,
)
from astro_reentry.models import ReentryOptimizationConfig, ReentryScenario
from astro_reentry.optimization import optimize_reentry_guidance
from astro_twin.constellation import run_constellation_twin
from astro_twin.constellation_io import (
    format_constellation_summary,
    load_constellation_twin_scenario,
)
from astro_twin.io import format_twin_summary, load_twin_scenario
from astro_twin.runner import run_digital_twin
from astro_uq.cli import (
    analyze_campaign_sensitivity_command,
    profile_campaign_command,
    run_campaign_command,
    summarize_campaign_command,
    validate_campaign,
)
from astro_uq.io import CampaignIOError

app = typer.Typer(help="Astro Suite flight dynamics workflows.")
app.command("validate-campaign")(validate_campaign)
app.command("run-campaign")(run_campaign_command)
app.command("summarize-campaign")(summarize_campaign_command)
app.command("profile-campaign")(profile_campaign_command)
app.command("analyze-campaign-sensitivity")(analyze_campaign_sensitivity_command)
app.command("validate-assurance-validation")(validate_assurance_validation_command)
app.command("validate-covariance-validation")(validate_covariance_validation_command)
app.command("assess-covariance-validation")(assess_covariance_validation_command)
app.command("verify-covariance-validation")(verify_covariance_validation_command)
app.command("run-empirical-covariance-campaign")(run_empirical_covariance_campaign_command)
app.command("inspect-assurance-calibration")(inspect_assurance_calibration_command)
app.command("run-assurance-validation")(run_assurance_validation_command)
app.command("verify-assurance-validation")(verify_assurance_validation_command)
app.command("validate-model-form-matrix")(validate_model_form_matrix_command)
app.command("run-model-form-matrix")(run_model_form_matrix_command)
app.command("verify-model-form-matrix")(verify_model_form_matrix_command)
app.command("review-assurance-validation")(review_assurance_validation_command)
app.command("compare-assurance-reviews")(compare_assurance_reviews_command)
app.command("verify-assurance-review-comparison")(verify_assurance_review_comparison_command)
app.command("verify-mission-lifecycle-result")(verify_mission_lifecycle_result_command)
app.command("review-mission-lifecycle")(review_mission_lifecycle_command)
app.command("verify-mission-lifecycle-review")(verify_mission_lifecycle_review_command)

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


def _load_reentry_scenario_or_exit(scenario_path: Path) -> ReentryScenario:
    try:
        return load_reentry_scenario(scenario_path)
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


def _ensure_parent_or_exit(output: Path, product_name: str) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f"could not create {product_name} parent {output.parent}: {exc}", err=True)
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
        station.model_dump(mode="json", exclude_none=True) for station in demo_added_ground_stations
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
    report_output: Annotated[
        Path | None,
        typer.Option("--report-output", help="Write a workflow report JSON to a file."),
    ] = None,
    report_summary_output: Annotated[
        Path | None,
        typer.Option(
            "--report-summary-output",
            help="Write a human-readable workflow report summary to a file.",
        ),
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
    report = (
        build_workflow_report(trace)
        if report_output is not None or report_summary_output is not None
        else None
    )
    if report_output is not None and report is not None:
        report_payload = report.model_dump_json(indent=2)
        try:
            report_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            typer.echo(f"could not write assistant report {report_output}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        _write_text_or_exit(report_output, report_payload, "assistant report")
    if report_summary_output is not None and report is not None:
        report_summary = format_workflow_report(report)
        try:
            report_summary_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            typer.echo(
                f"could not write assistant report summary {report_summary_output}: {exc}",
                err=True,
            )
            raise typer.Exit(code=2) from exc
        _write_text_or_exit(report_summary_output, report_summary, "assistant report summary")
    typer.echo(payload)
    if not trace.verification.passed:
        for diagnostic in trace.verification.diagnostics:
            typer.echo(diagnostic.message, err=True)
    if trace.warnings:
        for warning in trace.warnings:
            typer.echo(warning, err=True)
    if not trace.verification.passed or trace.warnings:
        raise typer.Exit(code=2)


@app.command("verify-assistant")
def verify_assistant(
    prompt: Annotated[str, typer.Argument(help="Natural language assistant request.")],
) -> None:
    classification = classify_local_od_support(prompt)
    if not classification.supported:
        verification = VerificationResult(
            passed=False,
            diagnostics=[
                VerificationDiagnostic(
                    code=classification.code,
                    message=classification.message,
                )
            ],
        )
        typer.echo(
            json.dumps(
                _assistant_verification_report(
                    classification=classification,
                    plan_id=None,
                    verification=verification,
                ),
                indent=2,
            )
        )
        raise typer.Exit(code=2)

    planner = DeterministicPlanner()
    plan = planner.plan(prompt)
    verification = verify_plan(plan)
    typer.echo(
        json.dumps(
            _assistant_verification_report(
                classification=classification,
                plan_id=plan.plan_id,
                verification=verification,
            ),
            indent=2,
        )
    )
    if not verification.passed:
        raise typer.Exit(code=2)


def _assistant_verification_report(
    *,
    classification: LocalODSupportReport,
    plan_id: str | None,
    verification: VerificationResult,
) -> dict[str, object]:
    return {
        "supported": classification.supported and verification.passed,
        "plan_id": plan_id,
        "scenario_path": classification.scenario_path,
        "scenario_id": classification.scenario_id,
        "artifact_dir": classification.artifact_dir,
        "classification": {
            "code": classification.code,
            "message": classification.message,
        },
        "verification": verification.model_dump(mode="json"),
    }


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


@app.command("run-twin")
def run_twin(
    scenario_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Digital twin scenario YAML path."),
    ],
    output: Annotated[Path, typer.Option("--output", help="Write digital twin JSON result.")],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise text summary."),
    ] = None,
) -> None:
    """Run the deterministic single-spacecraft digital twin workflow."""
    try:
        scenario = load_twin_scenario(scenario_path)
        result = run_digital_twin(scenario)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _ensure_parent_or_exit(output, "digital twin result")
    _write_text_or_exit(output, result.model_dump_json(indent=2), "digital twin result")
    typer.echo(f"wrote digital twin result: {output}")
    if summary_output is not None:
        _ensure_parent_or_exit(summary_output, "digital twin summary")
        _write_text_or_exit(
            summary_output,
            format_twin_summary(result),
            "digital twin summary",
        )
        typer.echo(f"wrote digital twin summary: {summary_output}")


@app.command("run-mission-lifecycle")
def run_mission_lifecycle_command(
    scenario_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Mission lifecycle scenario YAML path."),
    ],
    output: Annotated[Path, typer.Option("--output", help="Write lifecycle JSON result.")],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise text summary."),
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option("--artifacts-dir", help="Write the phase artifact bundle."),
    ] = None,
) -> None:
    """Run the checked launch-to-reentry mission lifecycle workflow."""
    try:
        scenario = load_mission_lifecycle_scenario(scenario_path)
        result = run_mission_lifecycle(scenario)
        _ensure_parent_or_exit(output, "mission lifecycle result")
        _write_text_or_exit(
            output,
            result.model_dump_json(indent=2),
            "mission lifecycle result",
        )
        if summary_output is not None:
            _ensure_parent_or_exit(summary_output, "mission lifecycle summary")
            _write_text_or_exit(
                summary_output,
                format_mission_lifecycle_summary(result),
                "mission lifecycle summary",
            )
        if artifacts_dir is not None:
            write_mission_artifact_bundle(artifacts_dir, result)
    except (InvalidScenarioError, MissionLifecycleError, UnsupportedBackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"wrote mission lifecycle result: {output}")
    if summary_output is not None:
        typer.echo(f"wrote mission lifecycle summary: {summary_output}")
    if artifacts_dir is not None:
        typer.echo(f"wrote mission artifact bundle: {artifacts_dir}")


@app.command("run-mission-evidence")
def run_mission_evidence_command(
    spec_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Mission evidence pack YAML path."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Publish the evidence pack to a new directory."),
    ],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Run the deterministic lifecycle, review, and uncertainty evidence pack."""
    try:
        manifest = run_mission_evidence_pack(spec_path, output_dir, workers=workers)
    except (InvalidScenarioError, MissionLifecycleError, ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(manifest.model_dump_json())


@app.command("run-mission-operator")
def run_mission_operator_command(
    spec_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Mission operator objective YAML path."),
    ],
    reasoner_replay: Annotated[
        Path,
        typer.Option(
            "--reasoner-replay",
            exists=True,
            readable=True,
            help="Checked typed action replay for the provider-neutral reasoner boundary.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Publish the operator run to a new directory."),
    ],
    command_store_path: Annotated[
        Path | None,
        typer.Option(
            "--command-store",
            help=(
                "Mission/grant-scoped durable SQLite command ledger. Required when the "
                "authority permits command execution; reuse it across output publications."
            ),
        ),
    ] = None,
    mission_design_context: Annotated[
        Path | None,
        typer.Option(
            "--mission-design-context",
            exists=True,
            file_okay=False,
            readable=True,
            help=(
                "Verified Director bundle used to resolve the exact run and baseline "
                "digests declared by the operator mission context."
            ),
        ),
    ] = None,
) -> None:
    """Run an adaptive, authority-scoped mission operator replay."""
    if output_dir.exists():
        typer.echo("operator output directory must not already exist", err=True)
        raise typer.Exit(code=2)
    partial_dir: Path | None = None
    retain_partial = False
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise OperatorError("operator output directory was created by another run")
        partial_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.partial-",
                dir=output_dir.parent,
            )
        )
        spec = load_mission_operator_spec(spec_path)
        mission_context = spec.mission_context
        if mission_design_context is not None:
            if mission_context is None:
                raise OperatorError("--mission-design-context requires a declared mission_context")
            mission_context = _resolve_operator_mission_context(
                mission_context,
                verify_mission_design_director(mission_design_context),
            )
        declared_scenario_path = Path(spec.base_scenario_path)
        scenario_path = (
            declared_scenario_path
            if declared_scenario_path.is_absolute()
            else spec_path.parent / declared_scenario_path
        ).resolve()
        scenario_evidence = capture_base_scenario_evidence(scenario_path, partial_dir)
        scenario = resolve_lifecycle_references(
            load_mission_lifecycle_scenario(partial_dir / scenario_evidence.path),
            scenario_path,
        )
        objective = type(spec.objective).model_validate(
            {
                **spec.objective.model_dump(mode="python"),
                "base_evidence": (
                    *spec.objective.base_evidence,
                    scenario_evidence,
                ),
            }
        )
        evaluator = LifecycleCandidateEvaluator(
            base_scenario=scenario,
            design_variables=objective.design_variables,
            output_root=partial_dir,
        )
        command_store: SQLiteCommandExecutionStore | None = None
        command_executor: CommandExecutionCoordinator | None = None
        if OperatorActionKind.EXECUTE_COMMAND in spec.authority.allowed_actions:
            if command_store_path is None:
                raise OperatorError(
                    "command-capable operator runs require a durable --command-store path"
                )
            retain_partial = True
            command_store = SQLiteCommandExecutionStore(command_store_path)
            initial_world_state = reduce_world_state(objective.base_assertions)
            command_executor = CommandExecutionCoordinator(
                CommandToolRegistry((SimulatedBurnTool(),)),
                command_store,
                authority_resolver=lambda _grant_id: spec.authority.model_copy(deep=True),
                world_state_resolver=lambda: initial_world_state.model_copy(deep=True),
            )
        try:
            run = run_operator(
                objective=objective,
                authority=spec.authority,
                mission_context=mission_context,
                reasoner=ConditionalReplayReasoner(load_operator_replay(reasoner_replay)),
                evaluator=evaluator,
                evidence_provider=build_operational_evidence_registry(
                    spec.evidence_sources,
                    source_root=spec_path.parent,
                    output_root=partial_dir,
                ),
                command_executor=command_executor,
            )
        finally:
            if command_store is not None:
                command_store.close()
        write_operator_run(partial_dir / "operator-run.json", run)
        partial_dir.replace(output_dir)
        partial_dir = None
    except (
        InvalidScenarioError,
        MissionLifecycleError,
        OperatorError,
        OSError,
        ValueError,
    ) as exc:
        if partial_dir is not None and partial_dir.exists():
            if retain_partial:
                typer.echo(
                    f"retained interrupted command journal for recovery: {partial_dir}",
                    err=True,
                )
            else:
                shutil.rmtree(partial_dir)
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"operator {run.status.value}: {len(run.steps)} steps, "
        f"selected={run.selected_candidate_id or 'none'}"
    )
    typer.echo(f"wrote mission operator run: {output_dir}")


def _resolve_operator_mission_context(
    declared: MissionBaselineContext,
    director_run: MissionDesignRun,
) -> MissionBaselineContext:
    baseline = director_run.baseline
    if (
        baseline is None
        or director_run.decision.disposition.value != "selected"
        or baseline.baseline_id != declared.baseline_id
        or baseline.version != declared.baseline_version
        or director_run.verification_plan.baseline_id != baseline.baseline_id
        or director_run.verification_plan.remaining_hard_requirement_ids
        or any(check.status != "passed" for check in director_run.verification_plan.checks)
    ):
        raise OperatorError(
            "mission design context does not resolve to the declared eligible baseline"
        )
    return declared.model_copy(
        update={
            "mission_design_run_sha256": director_run.run_sha256,
            "baseline_sha256": baseline.baseline_sha256,
        }
    )


@app.command("verify-mission-operator")
def verify_mission_operator_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify a mission operator journal and every local evidence digest."""
    try:
        run = verify_operator_run(output_dir)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"verified mission operator run: {run.status.value}, evidence={len(run.known_evidence)}"
    )


@app.command("run-mission-design-director")
def run_mission_design_director_command(
    spec_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Mission design director YAML path."),
    ],
    reasoner_replay: Annotated[
        Path,
        typer.Option(
            "--reasoner-replay",
            exists=True,
            readable=True,
            help="Checked adaptive candidate replay for the provider-neutral reasoner boundary.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Publish the design bundle to a new directory."),
    ],
) -> None:
    """Run the first evidence-backed, multi-domain Mission Design Director slice."""
    if output_dir.exists():
        typer.echo("mission design output directory must not already exist", err=True)
        raise typer.Exit(code=2)
    partial_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise OperatorError("mission design output directory was created by another run")
        partial_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.partial-",
                dir=output_dir.parent,
            )
        )
        spec = load_mission_design_director_spec(spec_path)
        captured_spec = partial_dir / "inputs" / "design-spec.yaml"
        captured_spec.parent.mkdir(parents=True)
        captured_spec.write_bytes(spec_path.read_bytes())
        operator_root = partial_dir / "operator"
        operator_root.mkdir()
        declared_scenario_path = Path(spec.base_scenario_path)
        scenario_path = (
            declared_scenario_path
            if declared_scenario_path.is_absolute()
            else spec_path.parent / declared_scenario_path
        ).resolve()
        scenario_evidence = capture_base_scenario_evidence(scenario_path, operator_root)
        scenario = resolve_lifecycle_references(
            load_mission_lifecycle_scenario(operator_root / scenario_evidence.path),
            scenario_path,
        )
        resolved_scenario_evidence = capture_resolved_base_scenario(scenario, operator_root)
        objective = spec.objective.model_copy(
            update={
                "base_evidence": (
                    *spec.objective.base_evidence,
                    scenario_evidence,
                    resolved_scenario_evidence,
                ),
            }
        )
        evaluator = LifecycleCandidateEvaluator(
            base_scenario=scenario,
            design_variables=objective.design_variables,
            output_root=operator_root,
        )
        operator_run = run_operator(
            objective=objective,
            authority=spec.authority,
            reasoner=ConditionalReplayReasoner(load_operator_replay(reasoner_replay)),
            evaluator=evaluator,
        )
        operator_path = operator_root / "operator-run.json"
        write_operator_run(operator_path, operator_run)
        design_run = build_mission_design_run(
            spec=spec,
            operator_run=operator_run,
            spec_sha256=sha256(captured_spec.read_bytes()).hexdigest(),
            operator_run_sha256=sha256(operator_path.read_bytes()).hexdigest(),
        )
        write_mission_design_run(partial_dir / "mission-design-run.json", design_run)
        write_mission_design_manifest(partial_dir)
        partial_dir.replace(output_dir)
        partial_dir = None
    except (
        InvalidScenarioError,
        MissionLifecycleError,
        OperatorError,
        OSError,
        ValueError,
    ) as exc:
        if partial_dir is not None and partial_dir.exists():
            shutil.rmtree(partial_dir)
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    recommended_count = sum(
        item.disposition.value == "recommended"
        for item in design_run.verification_plan.conditional_analyses
    )
    typer.echo(
        f"mission design {design_run.decision.disposition.value}: "
        f"candidate={design_run.decision.selected_candidate_id or 'none'}, "
        f"conditional_analyses_recommended={recommended_count}"
    )
    typer.echo(f"wrote mission design director bundle: {output_dir}")


@app.command("verify-mission-design-director")
def verify_mission_design_director_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify the exact design bundle and recompute its decision from operator evidence."""
    try:
        run = verify_mission_design_director(output_dir)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    recommended_count = sum(
        item.disposition.value == "recommended"
        for item in run.verification_plan.conditional_analyses
    )
    typer.echo(
        f"verified mission design director: {run.decision.disposition.value}, "
        f"candidate={run.decision.selected_candidate_id or 'none'}, "
        f"conditional_analyses_recommended={recommended_count}"
    )


@app.command("run-mission-design-conditional-campaign")
def run_mission_design_conditional_campaign_command(
    director_root: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    spec_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ],
    resume: Annotated[bool, typer.Option("--resume")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Execute a recommended Director campaign and reduce its checked evidence."""
    try:
        outcome = run_conditional_campaign(
            director_root=director_root,
            spec_path=spec_path,
            output_dir=output_dir,
            resume=resume,
            workers=workers,
        )
    except (
        CampaignIOError,
        InvalidScenarioError,
        MissionLifecycleError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"conditional mission-design campaign: {outcome.disposition.value}, "
        f"baseline={outcome.baseline_id}, "
        f"completed={outcome.completed_samples}/{outcome.requested_samples}"
    )
    typer.echo(f"wrote conditional campaign bundle: {output_dir}")


@app.command("verify-mission-design-conditional-campaign")
def verify_mission_design_conditional_campaign_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify a conditional campaign without rerunning physics or a provider."""
    try:
        outcome = verify_conditional_campaign(output_dir)
    except (
        CampaignIOError,
        InvalidScenarioError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"verified conditional mission-design campaign: "
        f"{outcome.disposition.value}, baseline={outcome.baseline_id}, "
        f"completed={outcome.completed_samples}/{outcome.requested_samples}"
    )


@app.command("build-mission-knowledge-graph")
def build_mission_knowledge_graph_command(
    spec_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Cross-run mission knowledge graph YAML path.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Publish the self-contained graph bundle."),
    ],
) -> None:
    """Capture verified mission runs and build their deterministic evidence graph."""
    try:
        graph = publish_mission_knowledge_graph(spec_path, output_dir)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"mission knowledge graph built: sources={len(graph.sources)}, "
        f"nodes={len(graph.nodes)}, edges={len(graph.edges)}"
    )
    typer.echo(f"wrote mission knowledge graph bundle: {output_dir}")


@app.command("verify-mission-knowledge-graph")
def verify_mission_knowledge_graph_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify source bundles and reconstruct their mission knowledge graph."""
    try:
        graph = verify_mission_knowledge_graph(output_dir)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"verified mission knowledge graph: sources={len(graph.sources)}, "
        f"nodes={len(graph.nodes)}, edges={len(graph.edges)}"
    )


@app.command("trace-mission-baseline")
def trace_mission_baseline_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    baseline_id: Annotated[
        str,
        typer.Option(
            "--baseline-id",
            help="Baseline record ID or fully namespaced graph node ID.",
        ),
    ],
) -> None:
    """Trace the requirements, evidence, and tool version behind a baseline."""
    try:
        graph = verify_mission_knowledge_graph(output_dir)
        trace = trace_baseline_justification(graph, baseline_id)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(trace.model_dump_json(indent=2))


@app.command("evaluate-mission-orchestration")
def evaluate_mission_orchestration_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    baseline_id: Annotated[str, typer.Option("--baseline-id")],
    operator_objective_id: Annotated[str, typer.Option("--operator-objective-id")],
    disposition_claim_id: Annotated[str, typer.Option("--claim-id")],
    manual_review_gate_predicate_id: Annotated[
        str,
        typer.Option("--manual-review-gate-predicate-id"),
    ],
) -> None:
    """Route a verified baseline-bound evidence package into manual review."""
    try:
        graph = verify_mission_knowledge_graph(output_dir)
        decision = evaluate_mission_orchestration(
            graph,
            MissionOrchestrationQuery(
                baseline_id=baseline_id,
                operator_objective_id=operator_objective_id,
                disposition_claim_id=disposition_claim_id,
                manual_review_gate_predicate_id=manual_review_gate_predicate_id,
            ),
        )
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(decision.model_dump_json(indent=2))


@app.command("score-mission-reasoner-corpus")
def score_mission_reasoner_corpus_command(
    corpus_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Offline adversarial corpus YAML path."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON score report path."),
    ] = None,
) -> None:
    """Score recorded reasoner actions without making provider calls."""
    try:
        score = score_adversarial_corpus(load_adversarial_corpus(corpus_path))
        payload = score.model_dump_json(indent=2) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(payload, nl=False)
    if not score.promoted:
        raise typer.Exit(code=1)


@app.command("score-mission-reasoner-behavior")
def score_mission_reasoner_behavior_command(
    corpus_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Offline behavior corpus YAML path."),
    ],
    reasoner_replay: Annotated[
        Path | None,
        typer.Option(
            "--reasoner-replay",
            exists=True,
            readable=True,
            help="Recorded provider-neutral decisions to score.",
        ),
    ] = None,
    reasoner_recording: Annotated[
        Path | None,
        typer.Option(
            "--reasoner-recording",
            exists=True,
            readable=True,
            help="Invocation-bound decisions to verify and score.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON behavior score path."),
    ] = None,
) -> None:
    """Score deterministic whole-run reasoner behavior without provider calls."""
    if (reasoner_replay is None) == (reasoner_recording is None):
        typer.echo(
            "provide exactly one of --reasoner-replay or --reasoner-recording",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        corpus = load_reasoner_behavior_corpus(corpus_path)
        if reasoner_recording is not None:
            score = score_recorded_reasoner_behavior_corpus(
                corpus, load_reasoner_behavior_recording(reasoner_recording)
            )
            passed = score.behavior_gate_passed
            payload = score.model_dump_json(indent=2) + "\n"
        else:
            assert reasoner_replay is not None
            replay_score = score_reasoner_behavior_corpus(
                corpus, load_reasoner_behavior_replay(reasoner_replay)
            )
            passed = replay_score.behavior_gate_passed
            payload = replay_score.model_dump_json(indent=2) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(payload, nl=False)
    if not passed:
        raise typer.Exit(code=1)


@app.command("bind-mission-reasoner-replay")
def bind_mission_reasoner_replay_command(
    corpus_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Behavior corpus YAML path."),
    ],
    reasoner_replay: Annotated[
        Path,
        typer.Option("--reasoner-replay", exists=True, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
    recording_id: Annotated[str, typer.Option("--recording-id")] = "bound-replay",
) -> None:
    """Convert a synthetic replay into an invocation-bound recording."""
    try:
        recording = record_reasoner_behavior_replay(
            load_reasoner_behavior_corpus(corpus_path),
            load_reasoner_behavior_replay(reasoner_replay),
            recording_id=recording_id,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_reasoner_behavior_recording(output, recording)
    except (InvalidScenarioError, OperatorError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"wrote invocation-bound reasoner recording: {output}")


@app.command("acquire-mission-reasoner-behavior")
def acquire_mission_reasoner_behavior_command(
    corpus_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Behavior corpus YAML path."),
    ],
    output: Annotated[Path, typer.Option("--output")],
    recording_id: Annotated[str, typer.Option("--recording-id")],
    provider: Annotated[str, typer.Option("--provider")] = "openrouter",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_OPENROUTER_MODEL,
    max_calls: Annotated[int, typer.Option("--max-calls", min=1, max=32)] = 8,
    timeout: Annotated[float, typer.Option("--timeout", min=1.0, max=300.0)] = 60.0,
    confirm_provider_calls: Annotated[
        bool,
        typer.Option(
            "--confirm-provider-calls",
            help="Explicitly authorize the bounded external provider calls.",
        ),
    ] = False,
) -> None:
    """Acquire a capped invocation-bound recording from an authorized provider."""
    if not confirm_provider_calls:
        typer.echo("provider calls require --confirm-provider-calls", err=True)
        raise typer.Exit(code=2)
    if provider != "openrouter":
        typer.echo(f"unsupported reasoner provider: {provider}", err=True)
        raise typer.Exit(code=2)
    try:
        corpus = load_reasoner_behavior_corpus(corpus_path)
        reserve_reasoner_behavior_recording(
            output,
            ReasonerBehaviorRecording(
                schema_version="1.0",
                recording_id=recording_id,
                corpus_id=corpus.corpus_id,
                call_cap=max_calls,
                calls_attempted=0,
                complete=False,
                cases=(),
            ),
        )
        recording = acquire_reasoner_behavior_recording(
            corpus,
            OpenRouterReasoner(model=model, timeout=timeout),
            recording_id=recording_id,
            max_calls=max_calls,
            checkpoint=lambda value: write_reasoner_behavior_recording(output, value),
        )
    except (InvalidScenarioError, OperatorError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"wrote reasoner recording: {output} calls={recording.calls_attempted}/{recording.call_cap}"
    )


@app.command("verify-mission-evidence")
def verify_mission_evidence_command(
    output_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify pack digests, deterministic review, and campaign integrity."""
    try:
        manifest = verify_mission_evidence_pack(output_dir)
    except (InvalidScenarioError, ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {"pack_id": manifest.pack_id, "verified": True, "workflow": manifest.workflow},
            sort_keys=True,
        )
    )


@app.command("run-mission-assurance")
def run_mission_assurance_command(
    scenario_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Post-launch assurance scenario YAML path.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", help="Write assurance case JSON result.")],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise text summary."),
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option("--artifacts-dir", help="Write the assurance artifact bundle."),
    ] = None,
) -> None:
    """Run deterministic post-launch acquisition and recovery screening."""
    try:
        scenario = load_post_launch_assurance_scenario(scenario_path)
        result = run_post_launch_assurance(scenario)
        _ensure_parent_or_exit(output, "mission assurance result")
        _write_text_or_exit(
            output,
            result.model_dump_json(indent=2),
            "mission assurance result",
        )
        if summary_output is not None:
            _ensure_parent_or_exit(summary_output, "mission assurance summary")
            _write_text_or_exit(
                summary_output,
                format_mission_assurance_summary(result),
                "mission assurance summary",
            )
        if artifacts_dir is not None:
            write_mission_assurance_artifact_bundle(artifacts_dir, result)
    except (
        InvalidScenarioError,
        MissionAssuranceError,
        NumericalConvergenceError,
        UnsupportedBackendError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"wrote mission assurance result: {output}")
    if summary_output is not None:
        typer.echo(f"wrote mission assurance summary: {summary_output}")
    if artifacts_dir is not None:
        typer.echo(f"wrote mission assurance artifact bundle: {artifacts_dir}")
    if not result.passed:
        raise typer.Exit(code=1)


@app.command("verify-mission-assurance")
def verify_mission_assurance_command(
    artifacts_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Mission assurance artifact bundle directory.",
        ),
    ],
) -> None:
    """Verify mission-assurance artifact and input digests."""
    try:
        manifest = verify_mission_assurance_artifact_bundle(artifacts_dir)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"verified mission assurance bundle: {manifest.scenario_id} "
        f"({len(manifest.entries)} artifacts, {len(manifest.inputs)} inputs)"
    )


@app.command("run-constellation-twin")
def run_constellation_twin_command(
    scenario_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Constellation twin scenario YAML path."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Write constellation digital twin JSON result."),
    ],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise constellation text summary."),
    ] = None,
) -> None:
    """Run the deterministic constellation digital twin workflow."""
    try:
        scenario = load_constellation_twin_scenario(scenario_path)
        result = run_constellation_twin(scenario)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _ensure_parent_or_exit(output, "constellation digital twin result")
    _write_text_or_exit(
        output,
        result.model_dump_json(indent=2),
        "constellation digital twin result",
    )
    typer.echo(f"wrote constellation digital twin result: {output}")

    if summary_output is not None:
        _ensure_parent_or_exit(summary_output, "constellation digital twin summary")
        _write_text_or_exit(
            summary_output,
            format_constellation_summary(result),
            "constellation digital twin summary",
        )
        typer.echo(f"wrote constellation digital twin summary: {summary_output}")


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


@app.command("simulate-reentry")
def simulate_reentry(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    backend: Annotated[str, typer.Option()] = "local",
    summary_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run a ballistic or lifting reentry scenario and write a result product."""
    scenario = _load_reentry_scenario_or_exit(scenario_path)
    try:
        result = simulate_reentry_with_backend(scenario, backend)
    except UnsupportedBackendError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _ensure_parent_or_exit(output, "reentry result")
    _write_text_or_exit(output, result.model_dump_json(indent=2), "reentry result")
    typer.echo(f"wrote reentry result: {output}")
    if summary_output is not None:
        _ensure_parent_or_exit(summary_output, "reentry summary")
        _write_text_or_exit(summary_output, format_reentry_summary(result), "reentry summary")
        typer.echo(f"wrote reentry summary: {summary_output}")


@app.command("optimize-reentry")
def optimize_reentry(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    tuned_scenario_output: Annotated[Path | None, typer.Option()] = None,
    summary_output: Annotated[Path | None, typer.Option()] = None,
    maximum_iterations: Annotated[int, typer.Option()] = 80,
    bank_angle_lower_deg: Annotated[float, typer.Option()] = 0.0,
    bank_angle_upper_deg: Annotated[float, typer.Option()] = 80.0,
    load_penalty_scale: Annotated[float, typer.Option()] = 1000.0,
) -> None:
    """Tune target-tracking bank magnitudes and write an optimization product."""
    scenario = _load_reentry_scenario_or_exit(scenario_path)
    try:
        config = ReentryOptimizationConfig(
            maximum_iterations=maximum_iterations,
            bank_angle_lower_deg=bank_angle_lower_deg,
            bank_angle_upper_deg=bank_angle_upper_deg,
            load_penalty_scale=load_penalty_scale,
        )
        result = optimize_reentry_guidance(scenario, config)
    except ValueError as exc:
        typer.echo(f"could not optimize reentry guidance: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _ensure_parent_or_exit(output, "reentry optimization")
    _write_text_or_exit(
        output,
        result.model_dump_json(indent=2),
        "reentry optimization",
    )
    typer.echo(f"wrote reentry optimization: {output}")
    if tuned_scenario_output is not None:
        payload = yaml.safe_dump(result.tuned_scenario.model_dump(mode="json"), sort_keys=False)
        _ensure_parent_or_exit(tuned_scenario_output, "tuned reentry scenario")
        _write_text_or_exit(
            tuned_scenario_output,
            payload.rstrip("\n"),
            "tuned reentry scenario",
        )
        typer.echo(f"wrote tuned reentry scenario: {tuned_scenario_output}")
    if summary_output is not None:
        _ensure_parent_or_exit(summary_output, "reentry optimization summary")
        _write_text_or_exit(
            summary_output,
            format_reentry_optimization_summary(result),
            "reentry optimization summary",
        )
        typer.echo(f"wrote reentry optimization summary: {summary_output}")


@app.command("handoff-reentry")
def handoff_reentry(
    trajectory_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    template_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    sample_index: Annotated[int, typer.Option()] = -1,
    scenario_id: Annotated[str | None, typer.Option()] = None,
    use_sample_mass: Annotated[bool, typer.Option()] = False,
) -> None:
    """Convert an EME2000 trajectory sample into a reentry scenario initial state."""
    trajectory = _load_trajectory_or_exit(trajectory_path)
    template = _load_reentry_scenario_or_exit(template_path)
    try:
        scenario = trajectory_to_reentry_scenario(
            trajectory,
            template,
            sample_index=sample_index,
            scenario_id=scenario_id,
            use_sample_mass=use_sample_mass,
        )
    except ValueError as exc:
        typer.echo(f"could not create reentry scenario: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    payload = yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False)
    _ensure_parent_or_exit(output, "reentry scenario")
    _write_text_or_exit(output, payload.rstrip("\n"), "reentry scenario")
    typer.echo(f"wrote reentry scenario: {output}")


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
            help=("Dymos optimization mode: phase, pitch-program, or multistage-pitch-program."),
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
