from __future__ import annotations

from datetime import timedelta
from math import atan2, cos, degrees, exp, radians, sin, sqrt

from astro_core.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from astro_launch.models import (
    LaunchEvent,
    LaunchEventType,
    LaunchScenario,
    LaunchStage,
    LaunchTrajectory,
    LaunchTrajectorySample,
)


def _atmospheric_density_kg_m3(scenario: LaunchScenario, altitude_m: float) -> float:
    if scenario.atmosphere.model == "none":
        return 0.0
    return scenario.atmosphere.sea_level_density_kg_m3 * exp(
        -max(0.0, altitude_m) / scenario.atmosphere.scale_height_m
    )


def _gravity_m_s2(altitude_m: float) -> float:
    radius_km = R_EARTH_KM + altitude_m / 1000.0
    return MU_EARTH_KM3_S2 / radius_km**2 * 1000.0


def _dynamic_pressure_pa(scenario: LaunchScenario, altitude_m: float, speed_m_s: float) -> float:
    density_kg_m3 = _atmospheric_density_kg_m3(scenario, altitude_m)
    return 0.5 * density_kg_m3 * speed_m_s**2


def _drag_acceleration_components_m_s2(
    *,
    dynamic_pressure_pa: float,
    radial_velocity_m_s: float,
    horizontal_velocity_m_s: float,
    mass_kg: float,
    stage: LaunchStage | None,
) -> tuple[float, float]:
    speed_m_s = _speed_m_s(radial_velocity_m_s, horizontal_velocity_m_s)
    if stage is None or speed_m_s == 0.0:
        return 0.0, 0.0
    magnitude_m_s2 = (
        dynamic_pressure_pa * stage.drag_coefficient * stage.reference_area_m2 / mass_kg
    )
    return (
        -magnitude_m_s2 * radial_velocity_m_s / speed_m_s,
        -magnitude_m_s2 * horizontal_velocity_m_s / speed_m_s,
    )


def _pitch_deg(scenario: LaunchScenario, time_s: float) -> float:
    if scenario.guidance.mode == "vertical":
        return 90.0

    pitch_program = scenario.guidance.pitch_program
    if time_s <= pitch_program[0].time_s:
        return pitch_program[0].pitch_deg
    for current_point, next_point in zip(pitch_program, pitch_program[1:], strict=False):
        if time_s <= next_point.time_s:
            segment_s = next_point.time_s - current_point.time_s
            fraction = (time_s - current_point.time_s) / segment_s
            return current_point.pitch_deg + fraction * (
                next_point.pitch_deg - current_point.pitch_deg
            )
    return pitch_program[-1].pitch_deg


def _speed_m_s(radial_velocity_m_s: float, horizontal_velocity_m_s: float) -> float:
    return sqrt(radial_velocity_m_s**2 + horizontal_velocity_m_s**2)


def _flight_path_angle_deg(radial_velocity_m_s: float, horizontal_velocity_m_s: float) -> float:
    if radial_velocity_m_s == 0.0 and horizontal_velocity_m_s == 0.0:
        return 90.0
    return degrees(atan2(radial_velocity_m_s, horizontal_velocity_m_s))


def _thrust_acceleration_components_m_s2(
    scenario: LaunchScenario,
    *,
    time_s: float,
    mass_kg: float,
    stage: LaunchStage | None,
) -> tuple[float, float]:
    if stage is None:
        return 0.0, 0.0
    thrust_acceleration_m_s2 = stage.engine.thrust_n / mass_kg
    if scenario.guidance.mode == "vertical":
        return thrust_acceleration_m_s2, 0.0

    pitch_rad = radians(_pitch_deg(scenario, time_s))
    return (
        thrust_acceleration_m_s2 * sin(pitch_rad),
        thrust_acceleration_m_s2 * cos(pitch_rad),
    )


def _acceleration_components_m_s2(
    *,
    scenario: LaunchScenario,
    time_s: float,
    altitude_m: float,
    radial_velocity_m_s: float,
    horizontal_velocity_m_s: float,
    mass_kg: float,
    stage: LaunchStage | None,
) -> tuple[float, float]:
    thrust_radial_m_s2, thrust_horizontal_m_s2 = _thrust_acceleration_components_m_s2(
        scenario,
        time_s=time_s,
        mass_kg=mass_kg,
        stage=stage,
    )
    dynamic_pressure_pa = _dynamic_pressure_pa(
        scenario,
        altitude_m,
        _speed_m_s(radial_velocity_m_s, horizontal_velocity_m_s),
    )
    drag_radial_m_s2, drag_horizontal_m_s2 = _drag_acceleration_components_m_s2(
        dynamic_pressure_pa=dynamic_pressure_pa,
        radial_velocity_m_s=radial_velocity_m_s,
        horizontal_velocity_m_s=horizontal_velocity_m_s,
        mass_kg=mass_kg,
        stage=stage,
    )
    return (
        thrust_radial_m_s2 - _gravity_m_s2(altitude_m) + drag_radial_m_s2,
        thrust_horizontal_m_s2 + drag_horizontal_m_s2,
    )


def _current_stage(scenario: LaunchScenario, stage_index: int) -> LaunchStage | None:
    if stage_index >= len(scenario.vehicle.stages):
        return None
    return scenario.vehicle.stages[stage_index]


def _append_event(
    events: list[LaunchEvent],
    scenario: LaunchScenario,
    *,
    event_type: LaunchEventType,
    time_s: float,
    stage_name: str,
) -> None:
    events.append(
        LaunchEvent(
            event_type=event_type,
            epoch=scenario.epoch + timedelta(seconds=time_s),
            time_s=time_s,
            stage_name=stage_name,
        )
    )


def _circular_velocity_km_s(altitude_km: float) -> float:
    return sqrt(MU_EARTH_KM3_S2 / (R_EARTH_KM + altitude_km))


def propagate_launch_local(scenario: LaunchScenario) -> LaunchTrajectory:
    stage_remaining_propellant_kg = [
        stage.propellant_mass_kg for stage in scenario.vehicle.stages
    ]
    stage_burn_elapsed_s = [0.0 for _stage in scenario.vehicle.stages]
    stage_index = 0
    time_s = 0.0
    altitude_m = scenario.launch_site.altitude_m
    downrange_m = 0.0
    radial_velocity_m_s = 0.0
    horizontal_velocity_m_s = 0.0
    mass_kg = scenario.vehicle.initial_mass_kg
    events: list[LaunchEvent] = []
    samples: list[LaunchTrajectorySample] = []

    first_stage = _current_stage(scenario, stage_index)
    if first_stage is not None:
        _append_event(
            events,
            scenario,
            event_type="stage_ignition",
            time_s=0.0,
            stage_name=first_stage.name,
        )

    for sample_index in range(scenario.propagation.sample_count):
        current_stage = _current_stage(scenario, stage_index)
        stage_name = current_stage.name if current_stage is not None else "payload"
        epoch = scenario.epoch + timedelta(seconds=time_s)
        sample_state = scenario.insertion_state_from_local_state(
            epoch=epoch,
            altitude_km=altitude_m / 1000.0,
            radial_velocity_km_s=radial_velocity_m_s / 1000.0,
            horizontal_velocity_km_s=horizontal_velocity_m_s / 1000.0,
        ).cartesian
        speed_m_s = _speed_m_s(radial_velocity_m_s, horizontal_velocity_m_s)
        dynamic_pressure_pa = _dynamic_pressure_pa(scenario, altitude_m, speed_m_s)
        radial_acceleration_m_s2, horizontal_acceleration_m_s2 = _acceleration_components_m_s2(
            scenario=scenario,
            time_s=time_s,
            altitude_m=altitude_m,
            radial_velocity_m_s=radial_velocity_m_s,
            horizontal_velocity_m_s=horizontal_velocity_m_s,
            mass_kg=mass_kg,
            stage=current_stage,
        )
        samples.append(
            LaunchTrajectorySample(
                epoch=epoch,
                time_s=time_s,
                altitude_km=altitude_m / 1000.0,
                downrange_km=downrange_m / 1000.0,
                velocity_km_s=speed_m_s / 1000.0,
                radial_velocity_km_s=radial_velocity_m_s / 1000.0,
                horizontal_velocity_km_s=horizontal_velocity_m_s / 1000.0,
                mass_kg=mass_kg,
                stage_name=stage_name,
                dynamic_pressure_pa=dynamic_pressure_pa,
                acceleration_m_s2=_speed_m_s(
                    radial_acceleration_m_s2,
                    horizontal_acceleration_m_s2,
                ),
                flight_path_angle_deg=_flight_path_angle_deg(
                    radial_velocity_m_s,
                    horizontal_velocity_m_s,
                ),
                state=sample_state,
            )
        )

        if sample_index == scenario.propagation.sample_count - 1:
            break

        step_remaining_s = scenario.propagation.step_s
        while step_remaining_s > 0.0:
            current_stage = _current_stage(scenario, stage_index)
            if current_stage is None:
                segment_s = step_remaining_s
            else:
                burn_remaining_s = current_stage.burn_duration_s - stage_burn_elapsed_s[stage_index]
                segment_s = min(step_remaining_s, burn_remaining_s)

            radial_acceleration_m_s2, horizontal_acceleration_m_s2 = _acceleration_components_m_s2(
                scenario=scenario,
                time_s=time_s,
                altitude_m=altitude_m,
                radial_velocity_m_s=radial_velocity_m_s,
                horizontal_velocity_m_s=horizontal_velocity_m_s,
                mass_kg=mass_kg,
                stage=current_stage,
            )
            altitude_m += (
                radial_velocity_m_s * segment_s
                + 0.5 * radial_acceleration_m_s2 * segment_s**2
            )
            downrange_m += (
                horizontal_velocity_m_s * segment_s
                + 0.5 * horizontal_acceleration_m_s2 * segment_s**2
            )
            radial_velocity_m_s += radial_acceleration_m_s2 * segment_s
            horizontal_velocity_m_s += horizontal_acceleration_m_s2 * segment_s
            if current_stage is not None:
                consumed_propellant_kg = min(
                    current_stage.engine.mass_flow_rate_kg_s * segment_s,
                    stage_remaining_propellant_kg[stage_index],
                )
                stage_remaining_propellant_kg[stage_index] -= consumed_propellant_kg
                stage_burn_elapsed_s[stage_index] += segment_s
                mass_kg -= consumed_propellant_kg

            time_s += segment_s
            step_remaining_s -= segment_s

            if (
                current_stage is not None
                and abs(stage_burn_elapsed_s[stage_index] - current_stage.burn_duration_s) <= 1.0e-9
            ):
                _append_event(
                    events,
                    scenario,
                    event_type="stage_burnout",
                    time_s=time_s,
                    stage_name=current_stage.name,
                )
                mass_kg -= current_stage.dry_mass_kg + stage_remaining_propellant_kg[stage_index]
                stage_remaining_propellant_kg[stage_index] = 0.0
                _append_event(
                    events,
                    scenario,
                    event_type="stage_separation",
                    time_s=time_s,
                    stage_name=current_stage.name,
                )
                stage_index += 1
                next_stage = _current_stage(scenario, stage_index)
                if next_stage is not None:
                    _append_event(
                        events,
                        scenario,
                        event_type="stage_ignition",
                        time_s=time_s,
                        stage_name=next_stage.name,
                    )

    insertion_sample = samples[-1]
    insertion_state = scenario.insertion_state_from_local_state(
        epoch=insertion_sample.epoch,
        altitude_km=insertion_sample.altitude_km,
        radial_velocity_km_s=insertion_sample.radial_velocity_km_s,
        horizontal_velocity_km_s=insertion_sample.horizontal_velocity_km_s,
    )
    _append_event(
        events,
        scenario,
        event_type="insertion",
        time_s=insertion_sample.time_s,
        stage_name="payload",
    )
    target_miss = {
        "altitude_miss_km": insertion_sample.altitude_km - scenario.target_orbit.altitude_km,
        "velocity_miss_km_s": insertion_sample.velocity_km_s
        - _circular_velocity_km_s(scenario.target_orbit.altitude_km),
        "radial_velocity_miss_km_s": insertion_sample.radial_velocity_km_s,
    }

    return LaunchTrajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        events=events,
        insertion_state=insertion_state,
        target_miss=target_miss,
        backend="local",
        metadata={
            "model": (
                "pitch_program_2d"
                if scenario.guidance.mode == "pitch_program"
                else "vertical_1d"
            ),
            "integrator": "constant_acceleration_step",
            "step_s": scenario.propagation.step_s,
            "sample_step_s": scenario.propagation.step_s,
            "atmosphere_model": scenario.atmosphere.model,
            "guidance_mode": scenario.guidance.mode,
            "pitch_program": [
                point.model_dump(mode="json") for point in scenario.guidance.pitch_program
            ],
        },
    )
