from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from astro_core.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.models import CartesianState, ForceModelName, Trajectory
from astro_dynamics.local import propagate_local
from astro_launch.handoff import launch_trajectory_to_orbit_scenario
from astro_launch.local import propagate_launch_local
from astro_launch.models import (
    LaunchReportInsertionMetrics,
    LaunchReportShortArcMetrics,
    LaunchScenario,
    LaunchTrajectory,
    TunedLaunchReport,
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

    return TunedLaunchReport(
        scenario_id=scenario.scenario_id,
        tuning_result=tuning_result,
        launch_trajectory=launch_trajectory,
        orbit_scenario=orbit_scenario,
        orbit_trajectory=orbit_trajectory,
        insertion_metrics=_insertion_metrics(tuning_result.tuned_scenario, launch_trajectory),
        short_arc_metrics=_short_arc_metrics(
            tuning_result.tuned_scenario,
            orbit_trajectory,
            duration_s=orbit_duration_s,
            step_s=orbit_step_s,
        ),
        backend="local",
        metadata={
            "workflow": "tuned_launch_report",
            "launch_backend": launch_trajectory.backend,
            "orbit_backend": orbit_trajectory.backend,
            "orbit_force_model": orbit_scenario.force_model.gravity,
        },
    )
