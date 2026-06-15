from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from astro_core.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.models import CartesianState, ForceModelName, Trajectory
from astro_dynamics.local import propagate_local
from astro_launch.handoff import launch_trajectory_to_orbit_scenario
from astro_launch.local import propagate_launch_local
from astro_launch.models import (
    LaunchReportAssessment,
    LaunchReportCheck,
    LaunchReportInsertionMetrics,
    LaunchReportMetricDelta,
    LaunchReportShortArcMetrics,
    LaunchScenario,
    LaunchTrajectory,
    TunedLaunchReport,
    TunedLaunchReportBatch,
    TunedLaunchReportBatchCase,
    TunedLaunchReportComparison,
)
from astro_launch.targeting import tune_pitch_program


def _circular_velocity_km_s(altitude_km: float) -> float:
    return sqrt(MU_EARTH_KM3_S2 / (R_EARTH_KM + altitude_km))


def _state_altitude_and_speed(state: CartesianState) -> tuple[float, float]:
    position_km = state.position_km
    velocity_km_s = state.velocity_km_s
    radius_km = sqrt(
        position_km[0] ** 2 + position_km[1] ** 2 + position_km[2] ** 2
    )
    speed_km_s = sqrt(
        velocity_km_s[0] ** 2 + velocity_km_s[1] ** 2 + velocity_km_s[2] ** 2
    )
    return radius_km - R_EARTH_KM, speed_km_s


def _insertion_metrics(
    scenario: LaunchScenario,
    trajectory: LaunchTrajectory,
) -> LaunchReportInsertionMetrics:
    insertion_sample = trajectory.samples[-1]
    target_altitude_km = scenario.target_orbit.altitude_km
    target_circular_velocity_km_s = _circular_velocity_km_s(target_altitude_km)
    return LaunchReportInsertionMetrics(
        target_altitude_km=target_altitude_km,
        target_circular_velocity_km_s=target_circular_velocity_km_s,
        altitude_km=insertion_sample.altitude_km,
        velocity_km_s=insertion_sample.velocity_km_s,
        radial_velocity_km_s=insertion_sample.radial_velocity_km_s,
        horizontal_velocity_km_s=insertion_sample.horizontal_velocity_km_s,
        altitude_miss_km=trajectory.target_miss["altitude_miss_km"],
        velocity_miss_km_s=trajectory.target_miss["velocity_miss_km_s"],
    )


def _short_arc_metrics(
    scenario: LaunchScenario,
    trajectory: Trajectory,
    *,
    duration_s: float,
    step_s: float,
) -> LaunchReportShortArcMetrics:
    target_altitude_km = scenario.target_orbit.altitude_km
    target_circular_velocity_km_s = _circular_velocity_km_s(target_altitude_km)
    altitudes_km: list[float] = []
    speeds_km_s: list[float] = []
    for sample in trajectory.samples:
        altitude_km, speed_km_s = _state_altitude_and_speed(sample.state)
        altitudes_km.append(altitude_km)
        speeds_km_s.append(speed_km_s)

    final_altitude_km = altitudes_km[-1]
    final_velocity_km_s = speeds_km_s[-1]
    return LaunchReportShortArcMetrics(
        duration_s=duration_s,
        step_s=step_s,
        sample_count=len(trajectory.samples),
        target_altitude_km=target_altitude_km,
        target_circular_velocity_km_s=target_circular_velocity_km_s,
        initial_altitude_km=altitudes_km[0],
        final_altitude_km=final_altitude_km,
        min_altitude_km=min(altitudes_km),
        max_altitude_km=max(altitudes_km),
        final_velocity_km_s=final_velocity_km_s,
        final_altitude_miss_km=final_altitude_km - target_altitude_km,
        final_velocity_miss_km_s=final_velocity_km_s - target_circular_velocity_km_s,
        altitudes_km=altitudes_km,
    )


def _check_against_tolerance(
    *,
    name: str,
    value: float,
    tolerance: float,
    units: str,
) -> LaunchReportCheck:
    return LaunchReportCheck(
        name=name,
        value=value,
        tolerance=tolerance,
        passed=abs(value) <= tolerance,
        units=units,
    )


def _assessment(checks: list[LaunchReportCheck]) -> LaunchReportAssessment:
    return LaunchReportAssessment(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def _insertion_assessment(
    scenario: LaunchScenario,
    metrics: LaunchReportInsertionMetrics,
) -> LaunchReportAssessment:
    return _assessment(
        [
            _check_against_tolerance(
                name="insertion_altitude_miss",
                value=metrics.altitude_miss_km,
                tolerance=scenario.target_orbit.altitude_tolerance_km,
                units="km",
            ),
            _check_against_tolerance(
                name="insertion_velocity_miss",
                value=metrics.velocity_miss_km_s,
                tolerance=scenario.target_orbit.velocity_tolerance_km_s,
                units="km/s",
            ),
        ]
    )


def _short_arc_assessment(
    scenario: LaunchScenario,
    metrics: LaunchReportShortArcMetrics,
) -> LaunchReportAssessment:
    return _assessment(
        [
            _check_against_tolerance(
                name="short_arc_final_altitude_miss",
                value=metrics.final_altitude_miss_km,
                tolerance=scenario.target_orbit.altitude_tolerance_km,
                units="km",
            ),
            _check_against_tolerance(
                name="short_arc_final_velocity_miss",
                value=metrics.final_velocity_miss_km_s,
                tolerance=scenario.target_orbit.velocity_tolerance_km_s,
                units="km/s",
            ),
        ]
    )


def _metric_delta(
    *,
    name: str,
    baseline_value: float,
    candidate_value: float,
    units: str,
) -> LaunchReportMetricDelta:
    baseline_abs_value = abs(baseline_value)
    candidate_abs_value = abs(candidate_value)
    return LaunchReportMetricDelta(
        name=name,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=candidate_value - baseline_value,
        baseline_abs_value=baseline_abs_value,
        candidate_abs_value=candidate_abs_value,
        improvement=baseline_abs_value - candidate_abs_value,
        improved=candidate_abs_value < baseline_abs_value,
        units=units,
    )


def compare_tuned_launch_reports(
    baseline: TunedLaunchReport,
    candidate: TunedLaunchReport,
) -> TunedLaunchReportComparison:
    """Compare two tuned launch report products without rerunning propagation."""
    metric_deltas = [
        _metric_delta(
            name="insertion_altitude_miss",
            baseline_value=baseline.insertion_metrics.altitude_miss_km,
            candidate_value=candidate.insertion_metrics.altitude_miss_km,
            units="km",
        ),
        _metric_delta(
            name="insertion_velocity_miss",
            baseline_value=baseline.insertion_metrics.velocity_miss_km_s,
            candidate_value=candidate.insertion_metrics.velocity_miss_km_s,
            units="km/s",
        ),
        _metric_delta(
            name="short_arc_final_altitude_miss",
            baseline_value=baseline.short_arc_metrics.final_altitude_miss_km,
            candidate_value=candidate.short_arc_metrics.final_altitude_miss_km,
            units="km",
        ),
        _metric_delta(
            name="short_arc_final_velocity_miss",
            baseline_value=baseline.short_arc_metrics.final_velocity_miss_km_s,
            candidate_value=candidate.short_arc_metrics.final_velocity_miss_km_s,
            units="km/s",
        ),
    ]
    return TunedLaunchReportComparison(
        baseline_scenario_id=baseline.scenario_id,
        candidate_scenario_id=candidate.scenario_id,
        baseline_passed=baseline.passed,
        candidate_passed=candidate.passed,
        passed_changed=baseline.passed != candidate.passed,
        baseline_insertion_passed=baseline.insertion_assessment.passed,
        candidate_insertion_passed=candidate.insertion_assessment.passed,
        baseline_short_arc_passed=baseline.short_arc_assessment.passed,
        candidate_short_arc_passed=candidate.short_arc_assessment.passed,
        metric_deltas=metric_deltas,
        backend="local",
        metadata={
            "workflow": "tuned_launch_report_comparison",
            "baseline_backend": baseline.backend,
            "candidate_backend": candidate.backend,
        },
    )


def _normalized_assessment_score(assessment: LaunchReportAssessment) -> float:
    return sum(abs(check.value) / check.tolerance for check in assessment.checks)


def _validate_iterations_values(iterations_values: Sequence[int]) -> list[int]:
    parsed_values = list(iterations_values)
    if not parsed_values:
        raise ValueError("iterations_values must contain at least one value")
    if any(iterations <= 0 for iterations in parsed_values):
        raise ValueError("iterations_values must contain positive integers")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("iterations_values must not contain duplicates")
    return parsed_values


def generate_tuned_launch_report_batch(
    scenario: LaunchScenario,
    *,
    point_indices: Sequence[int],
    iterations_values: Sequence[int],
    initial_span_deg: float = 10.0,
    refinement_factor: float = 0.5,
    altitude_weight: float = 1.0,
    velocity_weight: float = 1.0,
    orbit_duration_s: float = 600.0,
    orbit_step_s: float = 60.0,
    spacecraft_name: str = "launch-payload",
    spacecraft_mass_kg: float | None = None,
    area_m2: float = 2.5,
    drag_coefficient: float = 2.2,
    reflectivity_coefficient: float = 1.3,
    gravity: ForceModelName = ForceModelName.TWO_BODY,
) -> TunedLaunchReportBatch:
    """Generate and rank tuned launch reports for a small iteration-count batch."""
    parsed_iterations_values = _validate_iterations_values(iterations_values)
    unranked_cases: list[tuple[int, int, TunedLaunchReport, float, float, float]] = []
    for case_index, iterations in enumerate(parsed_iterations_values):
        report = generate_tuned_launch_report(
            scenario,
            point_indices=point_indices,
            initial_span_deg=initial_span_deg,
            iterations=iterations,
            refinement_factor=refinement_factor,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
            orbit_duration_s=orbit_duration_s,
            orbit_step_s=orbit_step_s,
            spacecraft_name=spacecraft_name,
            spacecraft_mass_kg=spacecraft_mass_kg,
            area_m2=area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
            gravity=gravity,
        )
        insertion_score = _normalized_assessment_score(report.insertion_assessment)
        short_arc_score = _normalized_assessment_score(report.short_arc_assessment)
        normalized_score = insertion_score + short_arc_score
        unranked_cases.append(
            (case_index, iterations, report, insertion_score, short_arc_score, normalized_score)
        )

    cases = [
        TunedLaunchReportBatchCase(
            case_index=case_index,
            rank=rank,
            label=f"iterations={iterations}",
            iterations=iterations,
            initial_span_deg=initial_span_deg,
            normalized_score=normalized_score,
            insertion_normalized_score=insertion_score,
            short_arc_normalized_score=short_arc_score,
            passed=report.passed,
            report=report,
        )
        for rank, (
            case_index,
            iterations,
            report,
            insertion_score,
            short_arc_score,
            normalized_score,
        ) in enumerate(
            sorted(unranked_cases, key=lambda item: (item[5], item[0])),
            start=1,
        )
    ]
    return TunedLaunchReportBatch(
        scenario_id=scenario.scenario_id,
        point_indices=list(point_indices),
        cases=cases,
        best_case=cases[0],
        backend="local",
        metadata={
            "workflow": "tuned_launch_report_batch",
            "ranking": "sum_abs_assessment_value_over_tolerance",
            "batch_axis": "iterations_values",
        },
    )


def generate_tuned_launch_report(
    scenario: LaunchScenario,
    *,
    point_indices: Sequence[int],
    initial_span_deg: float = 10.0,
    iterations: int = 2,
    refinement_factor: float = 0.5,
    altitude_weight: float = 1.0,
    velocity_weight: float = 1.0,
    orbit_duration_s: float = 600.0,
    orbit_step_s: float = 60.0,
    spacecraft_name: str = "launch-payload",
    spacecraft_mass_kg: float | None = None,
    area_m2: float = 2.5,
    drag_coefficient: float = 2.2,
    reflectivity_coefficient: float = 1.3,
    gravity: ForceModelName = ForceModelName.TWO_BODY,
) -> TunedLaunchReport:
    """Run tune -> launch -> orbit handoff -> orbit propagation and summarize metrics."""
    tuning_result = tune_pitch_program(
        scenario,
        point_indices=point_indices,
        initial_span_deg=initial_span_deg,
        iterations=iterations,
        refinement_factor=refinement_factor,
        altitude_weight=altitude_weight,
        velocity_weight=velocity_weight,
    )
    launch_trajectory = propagate_launch_local(tuning_result.tuned_scenario)
    orbit_scenario = launch_trajectory_to_orbit_scenario(
        launch_trajectory,
        duration_s=orbit_duration_s,
        step_s=orbit_step_s,
        spacecraft_name=spacecraft_name,
        spacecraft_mass_kg=spacecraft_mass_kg,
        area_m2=area_m2,
        drag_coefficient=drag_coefficient,
        reflectivity_coefficient=reflectivity_coefficient,
        gravity=gravity,
    )
    orbit_trajectory = propagate_local(orbit_scenario)
    insertion_metrics = _insertion_metrics(tuning_result.tuned_scenario, launch_trajectory)
    short_arc_metrics = _short_arc_metrics(
        tuning_result.tuned_scenario,
        orbit_trajectory,
        duration_s=orbit_duration_s,
        step_s=orbit_step_s,
    )
    insertion_assessment = _insertion_assessment(
        tuning_result.tuned_scenario,
        insertion_metrics,
    )
    short_arc_assessment = _short_arc_assessment(
        tuning_result.tuned_scenario,
        short_arc_metrics,
    )

    return TunedLaunchReport(
        scenario_id=scenario.scenario_id,
        tuning_result=tuning_result,
        launch_trajectory=launch_trajectory,
        orbit_scenario=orbit_scenario,
        orbit_trajectory=orbit_trajectory,
        insertion_metrics=insertion_metrics,
        short_arc_metrics=short_arc_metrics,
        insertion_assessment=insertion_assessment,
        short_arc_assessment=short_arc_assessment,
        passed=insertion_assessment.passed and short_arc_assessment.passed,
        backend="local",
        metadata={
            "workflow": "tuned_launch_report",
            "launch_backend": launch_trajectory.backend,
            "orbit_backend": orbit_trajectory.backend,
            "orbit_force_model": orbit_scenario.force_model.gravity,
        },
    )
