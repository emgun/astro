from __future__ import annotations

from datetime import timedelta
from math import copysign, exp, sqrt

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


def _dynamic_pressure_pa(scenario: LaunchScenario, altitude_m: float, velocity_m_s: float) -> float:
    density_kg_m3 = _atmospheric_density_kg_m3(scenario, altitude_m)
    return 0.5 * density_kg_m3 * velocity_m_s**2


def _drag_acceleration_m_s2(
    *,
    dynamic_pressure_pa: float,
    velocity_m_s: float,
    mass_kg: float,
    stage: LaunchStage | None,
) -> float:
    if stage is None or velocity_m_s == 0.0:
        return 0.0
    magnitude_m_s2 = (
        dynamic_pressure_pa * stage.drag_coefficient * stage.reference_area_m2 / mass_kg
    )
    return -copysign(magnitude_m_s2, velocity_m_s)


def _acceleration_m_s2(
    *,
    scenario: LaunchScenario,
    altitude_m: float,
    velocity_m_s: float,
    mass_kg: float,
    stage: LaunchStage | None,
) -> float:
    thrust_acceleration_m_s2 = 0.0 if stage is None else stage.engine.thrust_n / mass_kg
    dynamic_pressure_pa = _dynamic_pressure_pa(scenario, altitude_m, velocity_m_s)
    return (
        thrust_acceleration_m_s2
        - _gravity_m_s2(altitude_m)
        + _drag_acceleration_m_s2(
            dynamic_pressure_pa=dynamic_pressure_pa,
            velocity_m_s=velocity_m_s,
            mass_kg=mass_kg,
            stage=stage,
        )
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
    velocity_m_s = 0.0
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
        sample_state = scenario.insertion_state_from_vertical_state(
            epoch=epoch,
            altitude_km=altitude_m / 1000.0,
            velocity_km_s=velocity_m_s / 1000.0,
        ).cartesian
        dynamic_pressure_pa = _dynamic_pressure_pa(scenario, altitude_m, velocity_m_s)
        samples.append(
            LaunchTrajectorySample(
                epoch=epoch,
                time_s=time_s,
                altitude_km=altitude_m / 1000.0,
                downrange_km=0.0,
                velocity_km_s=velocity_m_s / 1000.0,
                mass_kg=mass_kg,
                stage_name=stage_name,
                dynamic_pressure_pa=dynamic_pressure_pa,
                acceleration_m_s2=_acceleration_m_s2(
                    scenario=scenario,
                    altitude_m=altitude_m,
                    velocity_m_s=velocity_m_s,
                    mass_kg=mass_kg,
                    stage=current_stage,
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

            acceleration_m_s2 = _acceleration_m_s2(
                scenario=scenario,
                altitude_m=altitude_m,
                velocity_m_s=velocity_m_s,
                mass_kg=mass_kg,
                stage=current_stage,
            )
            altitude_m += velocity_m_s * segment_s + 0.5 * acceleration_m_s2 * segment_s**2
            velocity_m_s += acceleration_m_s2 * segment_s
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
    insertion_state = scenario.insertion_state_from_vertical_state(
        epoch=insertion_sample.epoch,
        altitude_km=insertion_sample.altitude_km,
        velocity_km_s=insertion_sample.velocity_km_s,
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
    }

    return LaunchTrajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        events=events,
        insertion_state=insertion_state,
        target_miss=target_miss,
        backend="local",
        metadata={
            "model": "vertical_1d",
            "integrator": "constant_acceleration_step",
            "step_s": scenario.propagation.step_s,
            "sample_step_s": scenario.propagation.step_s,
            "atmosphere_model": scenario.atmosphere.model,
            "guidance_mode": scenario.guidance.mode,
        },
    )
