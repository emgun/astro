from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from math import sqrt
from typing import Any

from astro_backends.rocketpy.runtime import RocketPyRuntime, load_rocketpy_runtime
from astro_core.errors import UnsupportedBackendError
from astro_launch.local import (
    _circular_velocity_km_s,
    _dynamic_pressure_pa,
    _flight_path_angle_deg,
    _speed_m_s,
)
from astro_launch.models import (
    LaunchEvent,
    LaunchRocketPyConfig,
    LaunchScenario,
    LaunchStage,
    LaunchTrajectory,
    LaunchTrajectorySample,
)

RocketPyRuntimeLoader = Callable[[], RocketPyRuntime]
RocketPyFlightRunner = Callable[
    [LaunchScenario, RocketPyRuntime, LaunchRocketPyConfig],
    LaunchTrajectory,
]


def _with_rocketpy_provenance(
    trajectory: LaunchTrajectory,
    runtime: RocketPyRuntime,
    config: LaunchRocketPyConfig,
) -> LaunchTrajectory:
    metadata = {
        **trajectory.metadata,
        "adapter": "rocketpy",
        "source_backend": trajectory.backend,
        "rocketpy_version": runtime.package_version,
        "rocketpy_configured": True,
        "rocketpy_rail_length_m": config.rail_length_m,
        "rocketpy_inclination_deg": config.inclination_deg,
        "rocketpy_heading_deg": config.heading_deg,
    }
    return trajectory.model_copy(update={"backend": "rocketpy", "metadata": metadata})


def _build_environment(scenario: LaunchScenario, runtime: RocketPyRuntime) -> Any:
    return runtime.environment(
        date=scenario.epoch,
        latitude=scenario.launch_site.latitude_deg,
        longitude=scenario.launch_site.longitude_deg,
        elevation=scenario.launch_site.altitude_m,
    )


def _build_solid_motor(config: LaunchRocketPyConfig, runtime: RocketPyRuntime) -> Any:
    return runtime.solid_motor(
        thrust_source=[
            (float(time_s), float(thrust_n))
            for time_s, thrust_n in config.motor_thrust_source_n
        ],
        dry_mass=config.motor_dry_mass_kg,
        dry_inertia=config.motor_dry_inertia_kg_m2,
        nozzle_radius=config.motor_nozzle_radius_m,
        grain_number=config.motor_grain_number,
        grain_density=config.motor_grain_density_kg_m3,
        grain_outer_radius=config.motor_grain_outer_radius_m,
        grain_initial_inner_radius=config.motor_grain_initial_inner_radius_m,
        grain_initial_height=config.motor_grain_initial_height_m,
        grain_separation=config.motor_grain_separation_m,
        grains_center_of_mass_position=config.motor_grains_center_of_mass_position_m,
        center_of_dry_mass_position=config.motor_center_of_dry_mass_position_m,
        nozzle_position=config.motor_nozzle_position_m,
        burn_time=config.motor_burn_time_s,
    )


def _build_rocket(config: LaunchRocketPyConfig, runtime: RocketPyRuntime, motor: Any) -> Any:
    rocket = runtime.rocket(
        radius=config.rocket_radius_m,
        mass=config.rocket_mass_without_motor_kg,
        inertia=config.rocket_inertia_without_motor_kg_m2,
        power_off_drag=config.rocket_power_off_drag_coefficient,
        power_on_drag=config.rocket_power_on_drag_coefficient,
        center_of_mass_without_motor=config.rocket_center_of_mass_without_motor_m,
    )
    rocket.add_motor(motor, config.motor_position_m)
    rocket.set_rail_buttons(
        upper_button_position=config.rail_button_upper_position_m,
        lower_button_position=config.rail_button_lower_position_m,
        angular_position=config.rail_button_angular_position_deg,
    )
    return rocket


def _build_flight(
    scenario: LaunchScenario,
    config: LaunchRocketPyConfig,
    runtime: RocketPyRuntime,
    *,
    rocket: Any,
    environment: Any,
) -> Any:
    return runtime.flight(
        rocket=rocket,
        environment=environment,
        rail_length=config.rail_length_m,
        inclination=config.inclination_deg,
        heading=config.heading_deg,
        max_time=scenario.propagation.duration_s,
        max_time_step=scenario.propagation.step_s,
        terminate_on_apogee=False,
        verbose=False,
    )


def _rocketpy_mass_kg(flight: Any, time_s: float, fallback_kg: float) -> float:
    try:
        total_mass = flight.rocket.total_mass
        return max(float(total_mass.get_value_opt(time_s)), 1.0e-9)
    except AttributeError:
        return fallback_kg


def _rocketpy_solution_at_time(
    flight: Any,
    time_s: float,
    *,
    atol_s: float,
) -> tuple[float, float, float, float, float, float]:
    try:
        solution = flight.get_solution_at_time(time_s, atol=atol_s)
    except TypeError:
        solution = flight.get_solution_at_time(time_s)
    return (
        float(solution[1]),
        float(solution[2]),
        float(solution[3]),
        float(solution[4]),
        float(solution[5]),
        float(solution[6]),
    )


def _rocketpy_solution_end_time_s(flight: Any, requested_duration_s: float) -> float:
    try:
        solution_array = flight.solution_array
        end_time_s = float(solution_array[-1][0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return requested_duration_s
    return min(requested_duration_s, max(0.0, end_time_s))


def _rocketpy_sample_times_s(scenario: LaunchScenario, flight: Any) -> list[float]:
    end_time_s = _rocketpy_solution_end_time_s(flight, scenario.propagation.duration_s)
    times_s: list[float] = []
    sample_index = 0
    while True:
        time_s = sample_index * scenario.propagation.step_s
        if time_s >= end_time_s - 1.0e-9:
            break
        times_s.append(time_s)
        sample_index += 1
    if not times_s or abs(times_s[-1] - end_time_s) > 1.0e-9:
        times_s.append(end_time_s)
    return times_s


def _stage_windows(scenario: LaunchScenario) -> list[tuple[LaunchStage, float, float]]:
    windows: list[tuple[LaunchStage, float, float]] = []
    start_s = 0.0
    for stage in scenario.vehicle.stages:
        end_s = start_s + stage.burn_duration_s
        windows.append((stage, start_s, end_s))
        start_s = end_s
    return windows


def _stage_name_at_time(scenario: LaunchScenario, time_s: float) -> str:
    windows = _stage_windows(scenario)
    if len(windows) == 1:
        return windows[0][0].name
    for index, (_stage, _start_s, end_s) in enumerate(windows):
        if time_s < end_s - 1.0e-9:
            return windows[index][0].name
        if abs(time_s - end_s) <= 1.0e-9:
            if index + 1 < len(windows):
                return windows[index + 1][0].name
            return "payload"
    return "payload"


def _stage_events(
    scenario: LaunchScenario,
    config: LaunchRocketPyConfig,
    insertion_sample: LaunchTrajectorySample,
) -> list[LaunchEvent]:
    events: list[LaunchEvent] = []
    windows = _stage_windows(scenario)
    if len(windows) == 1:
        stage = windows[0][0]
        burnout_s = min(config.motor_burn_time_s, insertion_sample.time_s)
        events.extend(
            [
                LaunchEvent(
                    event_type="stage_ignition",
                    epoch=scenario.epoch,
                    time_s=0.0,
                    stage_name=stage.name,
                ),
                LaunchEvent(
                    event_type="stage_burnout",
                    epoch=scenario.epoch + timedelta(seconds=burnout_s),
                    time_s=burnout_s,
                    stage_name=stage.name,
                ),
            ]
        )
    insertion_time_s = insertion_sample.time_s
    for index, (stage, start_s, end_s) in enumerate(windows):
        if len(windows) == 1:
            break
        if start_s > insertion_time_s + 1.0e-9:
            break
        if index == 0 or len(windows) > 1:
            events.append(
                LaunchEvent(
                    event_type="stage_ignition",
                    epoch=scenario.epoch + timedelta(seconds=start_s),
                    time_s=start_s,
                    stage_name=stage.name,
                )
            )
        if end_s <= insertion_time_s + 1.0e-9:
            events.append(
                LaunchEvent(
                    event_type="stage_burnout",
                    epoch=scenario.epoch + timedelta(seconds=end_s),
                    time_s=end_s,
                    stage_name=stage.name,
                )
            )
            events.append(
                LaunchEvent(
                    event_type="stage_separation",
                    epoch=scenario.epoch + timedelta(seconds=end_s),
                    time_s=end_s,
                    stage_name=stage.name,
                )
            )
    events.append(
        LaunchEvent(
            event_type="insertion",
            epoch=insertion_sample.epoch,
            time_s=insertion_sample.time_s,
            stage_name="payload",
        )
    )
    return events


def _rocketpy_multistage_adapter_contract(
    scenario: LaunchScenario,
    *,
    stage_schedule_complete: bool,
    stage_schedule_duration_s: float,
) -> dict[str, Any] | None:
    if len(scenario.vehicle.stages) <= 1:
        return None
    return {
        "execution_scope": "single_configured_rocketpy_flight_with_suite_stage_annotations",
        "native_multistage_execution": False,
        "suite_stage_count": len(scenario.vehicle.stages),
        "suite_stage_names": [stage.name for stage in scenario.vehicle.stages],
        "rocketpy_config_count": 1,
        "annotated_stage_events": True,
        "stage_schedule_complete": stage_schedule_complete,
        "stage_schedule_duration_s": stage_schedule_duration_s,
        "native_multistage_gap": "runtime has no suite-validated staged multi-motor flight API",
    }


def _trajectory_from_rocketpy_flight(
    scenario: LaunchScenario,
    config: LaunchRocketPyConfig,
    flight: Any,
) -> LaunchTrajectory:
    samples: list[LaunchTrajectorySample] = []
    previous_speed_m_s: float | None = None
    previous_time_s: float | None = None
    sample_times_s = _rocketpy_sample_times_s(scenario, flight)
    sample_atol_s = max(scenario.propagation.step_s / 2.0, 1.0e-3)

    for time_s in sample_times_s:
        epoch = scenario.epoch + timedelta(seconds=time_s)
        x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s = _rocketpy_solution_at_time(
            flight,
            time_s,
            atol_s=sample_atol_s,
        )
        altitude_m = scenario.launch_site.altitude_m + z_m
        downrange_m = sqrt(x_m**2 + y_m**2)
        horizontal_velocity_m_s = sqrt(vx_m_s**2 + vy_m_s**2)
        speed_m_s = _speed_m_s(vz_m_s, horizontal_velocity_m_s)
        mass_kg = _rocketpy_mass_kg(flight, time_s, scenario.vehicle.initial_mass_kg)
        if previous_speed_m_s is None or previous_time_s is None:
            acceleration_m_s2 = 0.0
        else:
            elapsed_s = max(time_s - previous_time_s, 1.0e-9)
            acceleration_m_s2 = abs((speed_m_s - previous_speed_m_s) / elapsed_s)
        previous_speed_m_s = speed_m_s
        previous_time_s = time_s
        sample_state = scenario.insertion_state_from_local_state(
            epoch=epoch,
            altitude_km=altitude_m / 1000.0,
            radial_velocity_km_s=vz_m_s / 1000.0,
            horizontal_velocity_km_s=horizontal_velocity_m_s / 1000.0,
        ).cartesian
        samples.append(
            LaunchTrajectorySample(
                epoch=epoch,
                time_s=time_s,
                altitude_km=altitude_m / 1000.0,
                downrange_km=downrange_m / 1000.0,
                velocity_km_s=speed_m_s / 1000.0,
                radial_velocity_km_s=vz_m_s / 1000.0,
                horizontal_velocity_km_s=horizontal_velocity_m_s / 1000.0,
                mass_kg=mass_kg,
                stage_name=_stage_name_at_time(scenario, time_s),
                dynamic_pressure_pa=_dynamic_pressure_pa(scenario, altitude_m, speed_m_s),
                acceleration_m_s2=acceleration_m_s2,
                flight_path_angle_deg=_flight_path_angle_deg(vz_m_s, horizontal_velocity_m_s),
                state=sample_state,
            )
        )

    insertion_sample = samples[-1]
    insertion_state = scenario.insertion_state_from_local_state(
        epoch=insertion_sample.epoch,
        altitude_km=insertion_sample.altitude_km,
        radial_velocity_km_s=insertion_sample.radial_velocity_km_s,
        horizontal_velocity_km_s=insertion_sample.horizontal_velocity_km_s,
    )
    events = _stage_events(scenario, config, insertion_sample)
    target_miss = {
        "altitude_miss_km": insertion_sample.altitude_km - scenario.target_orbit.altitude_km,
        "velocity_miss_km_s": insertion_sample.velocity_km_s
        - _circular_velocity_km_s(scenario.target_orbit.altitude_km),
    }
    is_multistage = len(scenario.vehicle.stages) > 1
    stage_schedule_duration_s = _stage_windows(scenario)[-1][2]
    stage_schedule_complete = sample_times_s[-1] >= stage_schedule_duration_s - 1.0e-9
    metadata: dict[str, Any] = {
        "model": (
            "rocketpy_configured_multistage_composition"
            if is_multistage
            else "rocketpy_single_stage_solid"
        ),
        "integrator": "rocketpy",
        "sample_step_s": scenario.propagation.step_s,
        "rocketpy_solution_end_s": sample_times_s[-1],
        "rail_length_m": config.rail_length_m,
        "inclination_deg": config.inclination_deg,
        "heading_deg": config.heading_deg,
        "rocketpy_stage_count": len(scenario.vehicle.stages),
        "rocketpy_stage_schedule_duration_s": stage_schedule_duration_s,
        "rocketpy_stage_schedule_complete": stage_schedule_complete,
        "rocketpy_composition": (
            "single_flight_suite_stage_schedule"
            if is_multistage
            else "single_stage_direct_flight"
        ),
    }
    multistage_contract = _rocketpy_multistage_adapter_contract(
        scenario,
        stage_schedule_complete=stage_schedule_complete,
        stage_schedule_duration_s=stage_schedule_duration_s,
    )
    if multistage_contract is not None:
        metadata["rocketpy_multistage_adapter_contract"] = multistage_contract
    return LaunchTrajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        events=events,
        insertion_state=insertion_state,
        target_miss=target_miss,
        backend="rocketpy_direct",
        metadata=metadata,
    )


def run_rocketpy_flight(
    scenario: LaunchScenario,
    runtime: RocketPyRuntime,
    config: LaunchRocketPyConfig,
) -> LaunchTrajectory:
    environment = _build_environment(scenario, runtime)
    motor = _build_solid_motor(config, runtime)
    rocket = _build_rocket(config, runtime, motor)
    flight = _build_flight(
        scenario,
        config,
        runtime,
        rocket=rocket,
        environment=environment,
    )
    return _trajectory_from_rocketpy_flight(scenario, config, flight)


def propagate_launch_rocketpy(
    scenario: LaunchScenario,
    *,
    runtime_loader: RocketPyRuntimeLoader = load_rocketpy_runtime,
    flight_runner: RocketPyFlightRunner | None = None,
) -> LaunchTrajectory:
    runtime = runtime_loader()
    config = scenario.rocketpy
    if config is None:
        raise UnsupportedBackendError(
            "RocketPy launch simulation requires scenario.rocketpy backend-specific "
            "vehicle, motor, and flight configuration; use --backend local for aggregate "
            "launch scenarios."
        )
    if flight_runner is None:
        flight_runner = run_rocketpy_flight

    trajectory = flight_runner(scenario, runtime, config)
    return _with_rocketpy_provenance(trajectory, runtime, config)
