from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.models import (
    CartesianState,
    CovarianceSample,
    ForceModelConfig,
    ForceModelName,
    Maneuver,
    Scenario,
    Trajectory,
    TrajectoryEvent,
    TrajectorySample,
    Vector3,
)

FloatArray = NDArray[np.float64]
_SUPPORTED_LOCAL_FORCE_MODELS = {ForceModelName.TWO_BODY, ForceModelName.J2}
_MAX_RK4_INTERNAL_STEP_S = 30.0
_MANEUVER_TIME_TOLERANCE_S = 1.0e-9
_COVARIANCE_FD_REL_STEP = 1.0e-6
_COVARIANCE_FD_ABS_STEP = 1.0e-8
_STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class _ScheduledManeuver:
    index: int
    maneuver: Maneuver
    start_s: float
    end_s: float


def _validate_local_force_model(force_model: ForceModelName) -> ForceModelName:
    if not isinstance(force_model, ForceModelName):
        raise ValueError("force_model must be a ForceModelName")
    if force_model not in _SUPPORTED_LOCAL_FORCE_MODELS:
        raise ValueError("Local backend supports only two_body and j2 force models")
    return force_model


def _validate_local_force_config(force_model: ForceModelConfig) -> ForceModelName:
    enabled_flags = force_model.enabled_high_fidelity_flags()
    if enabled_flags:
        raise ValueError(
            "Local backend does not support high-fidelity force model flags: "
            f"{', '.join(enabled_flags)}"
        )
    return _validate_local_force_model(force_model.gravity)


def _radius_metrics(position_km: FloatArray) -> tuple[float, float]:
    radius2 = float(np.dot(position_km, position_km))
    if radius2 == 0.0:
        raise ValueError("Cannot compute acceleration for zero-radius position")
    return radius2, radius2**0.5


def two_body_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    _, radius = _radius_metrics(position_km)
    return -MU_EARTH_KM3_S2 * position_km / radius**3


def j2_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    x, y, z = position_km
    radius2, radius = _radius_metrics(position_km)
    z2_over_r2 = (z * z) / radius2
    factor = 1.5 * J2_EARTH * MU_EARTH_KM3_S2 * R_EARTH_KM**2 / radius**5
    return cast(
        FloatArray,
        factor
        * np.array(
            [
                x * (5.0 * z2_over_r2 - 1.0),
                y * (5.0 * z2_over_r2 - 1.0),
                z * (5.0 * z2_over_r2 - 3.0),
            ],
            dtype=np.float64,
        ),
    )


def _vector3_from_array(values: FloatArray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def acceleration_km_s2(position_km: FloatArray, force_model: ForceModelName) -> FloatArray:
    force_model = _validate_local_force_model(force_model)
    acceleration = two_body_acceleration_km_s2(position_km)
    if force_model is ForceModelName.J2:
        acceleration = acceleration + j2_acceleration_km_s2(position_km)
    return acceleration


def derivative(state: FloatArray, force_model: ForceModelName) -> FloatArray:
    force_model = _validate_local_force_model(force_model)
    position = state[:3]
    velocity = state[3:]
    return cast(FloatArray, np.concatenate([velocity, acceleration_km_s2(position, force_model)]))


def _internal_step_schedule(step_s: float) -> tuple[int, float]:
    substep_count = max(1, ceil(abs(step_s) / _MAX_RK4_INTERNAL_STEP_S))
    return substep_count, step_s / substep_count


def _rk4_step_once(state: FloatArray, step_s: float, force_model: ForceModelName) -> FloatArray:
    k1 = derivative(state, force_model)
    k2 = derivative(state + 0.5 * step_s * k1, force_model)
    k3 = derivative(state + 0.5 * step_s * k2, force_model)
    k4 = derivative(state + step_s * k3, force_model)
    return cast(FloatArray, state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def rk4_step(state: FloatArray, step_s: float, force_model: ForceModelName) -> FloatArray:
    """Advance a state over the requested interval using RK4 substeps capped at 30s."""
    force_model = _validate_local_force_model(force_model)
    substep_count, substep_s = _internal_step_schedule(step_s)
    next_state = state

    for _ in range(substep_count):
        next_state = _rk4_step_once(next_state, substep_s, force_model)

    return next_state


def _maneuver_offset_s(scenario: Scenario, maneuver: Maneuver) -> float:
    return (maneuver.epoch - scenario.initial_state.epoch).total_seconds()


def _scheduled_maneuvers(scenario: Scenario) -> list[_ScheduledManeuver]:
    duration_s = scenario.propagation.duration_s
    schedule: list[_ScheduledManeuver] = []

    for index, maneuver in enumerate(sorted(scenario.maneuvers, key=lambda item: item.epoch)):
        if maneuver.frame != scenario.initial_state.frame:
            raise ValueError("Local maneuver propagation requires maneuvers in the scenario frame")

        start_s = _maneuver_offset_s(scenario, maneuver)
        end_s = start_s + maneuver.duration_s
        if start_s < -_MANEUVER_TIME_TOLERANCE_S:
            raise ValueError("Local maneuver propagation does not support pre-epoch maneuvers")
        if start_s > duration_s + _MANEUVER_TIME_TOLERANCE_S:
            raise ValueError(
                "Local maneuver propagation does not support post-propagation maneuvers"
            )
        if end_s > duration_s + _MANEUVER_TIME_TOLERANCE_S:
            raise ValueError("Local finite-burn maneuvers must end within the propagation window")

        schedule.append(
            _ScheduledManeuver(
                index=index,
                maneuver=maneuver,
                start_s=max(0.0, start_s),
                end_s=min(duration_s, end_s),
            )
        )

    return schedule


def _finite_burn_acceleration_km_s2(
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
    mass_kg: float,
    position_km: FloatArray,
    velocity_km_s: FloatArray,
) -> FloatArray:
    acceleration = np.zeros(3, dtype=np.float64)
    for scheduled in schedule:
        maneuver = scheduled.maneuver
        if maneuver.duration_s <= 0.0:
            continue
        if (
            scheduled.start_s - _MANEUVER_TIME_TOLERANCE_S
            <= elapsed_s
            < scheduled.end_s - _MANEUVER_TIME_TOLERANCE_S
        ):
            if maneuver.thrust_vector_n is not None:
                thrust_vector_n = np.array(maneuver.thrust_vector_n, dtype=np.float64)
                if maneuver.thrust_direction_mode == "velocity_aligned":
                    velocity_norm = float(np.linalg.norm(velocity_km_s))
                    if velocity_norm == 0.0:
                        raise ValueError(
                            "velocity-aligned thrust requires nonzero spacecraft velocity"
                        )
                    thrust_vector_n = (
                        float(np.linalg.norm(thrust_vector_n)) * velocity_km_s / velocity_norm
                    )
                elif maneuver.thrust_direction_mode in {"radial_outward", "radial_inward"}:
                    position_norm = float(np.linalg.norm(position_km))
                    if position_norm == 0.0:
                        raise ValueError("radial thrust requires nonzero spacecraft position")
                    direction_sign = (
                        1.0 if maneuver.thrust_direction_mode == "radial_outward" else -1.0
                    )
                    thrust_vector_n = (
                        direction_sign
                        * float(np.linalg.norm(thrust_vector_n))
                        * position_km
                        / position_norm
                    )
                thrust_acceleration_m_s2 = thrust_vector_n / mass_kg
                acceleration = acceleration + thrust_acceleration_m_s2 / 1000.0
            else:
                acceleration = acceleration + (
                    np.array(maneuver.delta_v_km_s, dtype=np.float64)
                    / float(maneuver.duration_s)
                )
    return cast(FloatArray, acceleration)


def _finite_burn_mass_flow_kg_s(
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
) -> float:
    mass_flow_kg_s = 0.0
    for scheduled in schedule:
        maneuver = scheduled.maneuver
        if (
            maneuver.duration_s <= 0.0
            or maneuver.thrust_vector_n is None
            or maneuver.specific_impulse_s is None
        ):
            continue
        if (
            scheduled.start_s - _MANEUVER_TIME_TOLERANCE_S
            <= elapsed_s
            < scheduled.end_s - _MANEUVER_TIME_TOLERANCE_S
        ):
            thrust_n = float(np.linalg.norm(np.array(maneuver.thrust_vector_n, dtype=np.float64)))
            mass_flow_kg_s += thrust_n / (maneuver.specific_impulse_s * _STANDARD_GRAVITY_M_S2)
    return mass_flow_kg_s


def _derivative_with_maneuvers(
    state_with_mass: FloatArray,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
) -> FloatArray:
    position = state_with_mass[:3]
    velocity = state_with_mass[3:6]
    mass_kg = max(float(state_with_mass[6]), 1.0e-9)
    total_acceleration = acceleration_km_s2(
        position,
        force_model,
    ) + _finite_burn_acceleration_km_s2(
        schedule,
        elapsed_s,
        mass_kg,
        position,
        velocity,
    )
    return cast(
        FloatArray,
        np.concatenate(
            [
                velocity,
                total_acceleration,
                np.array([-_finite_burn_mass_flow_kg_s(schedule, elapsed_s)]),
            ]
        ),
    )


def _rk4_step_once_with_maneuvers(
    state_with_mass: FloatArray,
    step_s: float,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
) -> FloatArray:
    k1 = _derivative_with_maneuvers(state_with_mass, force_model, schedule, elapsed_s)
    k2 = _derivative_with_maneuvers(
        state_with_mass + 0.5 * step_s * k1,
        force_model,
        schedule,
        elapsed_s + 0.5 * step_s,
    )
    k3 = _derivative_with_maneuvers(
        state_with_mass + 0.5 * step_s * k2,
        force_model,
        schedule,
        elapsed_s + 0.5 * step_s,
    )
    k4 = _derivative_with_maneuvers(
        state_with_mass + step_s * k3,
        force_model,
        schedule,
        elapsed_s + step_s,
    )
    next_state = state_with_mass + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6] = max(float(next_state[6]), 1.0e-9)
    return cast(FloatArray, next_state)


def _next_maneuver_boundary_s(
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
    target_s: float,
) -> float | None:
    boundaries: list[float] = []
    for scheduled in schedule:
        boundaries.append(scheduled.start_s)
        if scheduled.maneuver.duration_s > 0.0:
            boundaries.append(scheduled.end_s)

    candidates = [
        boundary
        for boundary in boundaries
        if elapsed_s + _MANEUVER_TIME_TOLERANCE_S < boundary < target_s - _MANEUVER_TIME_TOLERANCE_S
    ]
    return min(candidates) if candidates else None


def _apply_impulses_at_elapsed_s(
    state: FloatArray,
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
    applied_impulses: set[int],
) -> FloatArray:
    next_state = state
    for scheduled in schedule:
        if scheduled.maneuver.duration_s > 0.0 or scheduled.index in applied_impulses:
            continue
        if abs(scheduled.start_s - elapsed_s) <= _MANEUVER_TIME_TOLERANCE_S:
            next_state = next_state.copy()
            next_state[3:] = next_state[3:] + np.array(
                scheduled.maneuver.delta_v_km_s,
                dtype=np.float64,
            )
            applied_impulses.add(scheduled.index)
    return next_state


def _advance_with_maneuvers(
    state: FloatArray,
    *,
    mass_kg: float,
    start_s: float,
    target_s: float,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    applied_impulses: set[int],
) -> tuple[FloatArray, float]:
    elapsed_s = start_s
    next_state = _apply_impulses_at_elapsed_s(state, schedule, elapsed_s, applied_impulses)
    state_with_mass = cast(
        FloatArray,
        np.concatenate([next_state, np.array([mass_kg], dtype=np.float64)]),
    )

    while elapsed_s < target_s - _MANEUVER_TIME_TOLERANCE_S:
        step_target_s = min(elapsed_s + _MAX_RK4_INTERNAL_STEP_S, target_s)
        boundary_s = _next_maneuver_boundary_s(schedule, elapsed_s, step_target_s)
        if boundary_s is not None:
            step_target_s = boundary_s

        step_s = step_target_s - elapsed_s
        state_with_mass = _rk4_step_once_with_maneuvers(
            state_with_mass,
            step_s,
            force_model,
            schedule,
            elapsed_s,
        )
        elapsed_s = step_target_s
        next_state = state_with_mass[:6]
        next_state = _apply_impulses_at_elapsed_s(
            next_state,
            schedule,
            elapsed_s,
            applied_impulses,
        )
        state_with_mass[:6] = next_state

    return cast(FloatArray, state_with_mass[:6].copy()), float(state_with_mass[6])


def _maneuver_events(
    schedule: list[_ScheduledManeuver],
    scenario: Scenario,
) -> list[TrajectoryEvent]:
    events: list[TrajectoryEvent] = []
    for scheduled in schedule:
        maneuver = scheduled.maneuver
        metadata: dict[str, Any] = {
            "maneuver": maneuver.name,
            "delta_v_km_s": maneuver.delta_v_km_s,
            "duration_s": maneuver.duration_s,
        }
        if maneuver.thrust_vector_n is not None:
            metadata["thrust_vector_n"] = maneuver.thrust_vector_n
            metadata["specific_impulse_s"] = maneuver.specific_impulse_s
            metadata["thrust_direction_mode"] = maneuver.thrust_direction_mode
        if maneuver.duration_s == 0.0:
            events.append(
                TrajectoryEvent(
                    event_type="maneuver_impulse",
                    epoch=maneuver.epoch,
                    description=f"Applied impulsive maneuver {maneuver.name}.",
                    metadata=metadata,
                )
            )
            continue

        events.append(
            TrajectoryEvent(
                event_type="maneuver_start",
                epoch=maneuver.epoch,
                description=f"Started finite-burn maneuver {maneuver.name}.",
                metadata=metadata,
            )
        )
        events.append(
            TrajectoryEvent(
                event_type="maneuver_end",
                epoch=scenario.initial_state.epoch + timedelta(seconds=scheduled.end_s),
                description=f"Completed finite-burn maneuver {maneuver.name}.",
                metadata=metadata,
            )
        )
    return events


def _maneuver_metadata(schedule: list[_ScheduledManeuver]) -> dict[str, Any]:
    if not schedule:
        return {}

    finite_burn_count = sum(1 for scheduled in schedule if scheduled.maneuver.duration_s > 0.0)
    impulse_count = len(schedule) - finite_burn_count
    thrust_vector_burn_count = sum(
        1 for scheduled in schedule if scheduled.maneuver.thrust_vector_n is not None
    )
    attitude_coupled_burn_count = sum(
        1
        for scheduled in schedule
        if scheduled.maneuver.thrust_vector_n is not None
        and scheduled.maneuver.thrust_direction_mode != "inertial"
    )
    thrust_direction_modes = sorted(
        {
            scheduled.maneuver.thrust_direction_mode
            for scheduled in schedule
            if scheduled.maneuver.thrust_vector_n is not None
        }
    )
    if attitude_coupled_burn_count:
        maneuver_model = "attitude_coupled_thrust_vector_mass_flow"
    elif thrust_vector_burn_count:
        maneuver_model = "thrust_vector_mass_flow"
    else:
        maneuver_model = "constant_inertial_acceleration"
    return {
        "maneuver_model": maneuver_model,
        "finite_burn_count": finite_burn_count,
        "impulsive_maneuver_count": impulse_count,
        "thrust_vector_burn_count": thrust_vector_burn_count,
        "attitude_coupled_burn_count": attitude_coupled_burn_count,
        "thrust_direction_modes": thrust_direction_modes,
    }


def _initial_covariance_matrix(scenario: Scenario) -> FloatArray | None:
    if scenario.initial_covariance is None:
        return None
    return cast(FloatArray, np.array(scenario.initial_covariance, dtype=np.float64))


def _matrix_to_nested_list(matrix: FloatArray) -> list[list[float]]:
    return [[float(component) for component in row] for row in matrix]


def _covariance_process_noise_model(acceleration_sigma_km_s2: float) -> str:
    return "white_acceleration" if acceleration_sigma_km_s2 > 0.0 else "none"


def _covariance_sample(
    epoch: datetime,
    covariance: FloatArray,
    *,
    state_transition_matrix: FloatArray,
    accumulated_state_transition_matrix: FloatArray,
    process_noise_covariance: FloatArray,
    metadata: dict[str, Any],
) -> CovarianceSample:
    return CovarianceSample(
        epoch=epoch,
        covariance=_matrix_to_nested_list(covariance),
        state_transition_matrix=_matrix_to_nested_list(state_transition_matrix),
        accumulated_state_transition_matrix=_matrix_to_nested_list(
            accumulated_state_transition_matrix
        ),
        process_noise_covariance=_matrix_to_nested_list(process_noise_covariance),
        metadata=metadata,
    )


def _finite_difference_state_transition(
    state: FloatArray,
    advance_state: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    transition = np.zeros((6, 6), dtype=np.float64)
    for column in range(6):
        step = max(abs(float(state[column])) * _COVARIANCE_FD_REL_STEP, _COVARIANCE_FD_ABS_STEP)
        perturbation = np.zeros(6, dtype=np.float64)
        perturbation[column] = step
        plus = advance_state(state + perturbation)
        minus = advance_state(state - perturbation)
        transition[:, column] = (plus - minus) / (2.0 * step)
    return cast(FloatArray, transition)


def two_body_acceleration_jacobian(position_km: FloatArray) -> FloatArray:
    radius2, radius = _radius_metrics(position_km)
    radius3 = radius2 * radius
    radius5 = radius3 * radius2
    identity = np.eye(3, dtype=np.float64)
    outer_position = np.outer(position_km, position_km)
    return cast(
        FloatArray,
        MU_EARTH_KM3_S2 * ((3.0 * outer_position / radius5) - (identity / radius3)),
    )


def _two_body_variational_derivative(augmented_state: FloatArray) -> FloatArray:
    state = augmented_state[:6]
    transition = augmented_state[6:].reshape((6, 6))
    dynamics_jacobian = np.zeros((6, 6), dtype=np.float64)
    dynamics_jacobian[:3, 3:] = np.eye(3, dtype=np.float64)
    dynamics_jacobian[3:, :3] = two_body_acceleration_jacobian(state[:3])
    transition_derivative = dynamics_jacobian @ transition
    return cast(
        FloatArray,
        np.concatenate(
            [
                derivative(state, ForceModelName.TWO_BODY),
                transition_derivative.reshape(36),
            ]
        ),
    )


def _rk4_variational_step_once(augmented_state: FloatArray, step_s: float) -> FloatArray:
    k1 = _two_body_variational_derivative(augmented_state)
    k2 = _two_body_variational_derivative(augmented_state + 0.5 * step_s * k1)
    k3 = _two_body_variational_derivative(augmented_state + 0.5 * step_s * k2)
    k4 = _two_body_variational_derivative(augmented_state + step_s * k3)
    return cast(
        FloatArray,
        augmented_state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
    )


def _two_body_variational_step(
    state: FloatArray,
    step_s: float,
) -> tuple[FloatArray, FloatArray]:
    substep_count, substep_s = _internal_step_schedule(step_s)
    augmented_state = cast(
        FloatArray,
        np.concatenate([state, np.eye(6, dtype=np.float64).reshape(36)]),
    )
    for _ in range(substep_count):
        augmented_state = _rk4_variational_step_once(augmented_state, substep_s)
    return augmented_state[:6], cast(FloatArray, augmented_state[6:].reshape((6, 6)))


def _process_noise_covariance(acceleration_sigma_km_s2: float, step_s: float) -> FloatArray:
    process_noise = np.zeros((6, 6), dtype=np.float64)
    if acceleration_sigma_km_s2 == 0.0:
        return cast(FloatArray, process_noise)

    acceleration_variance = acceleration_sigma_km_s2**2
    position_variance = 0.25 * acceleration_variance * step_s**4
    position_velocity_covariance = 0.5 * acceleration_variance * step_s**3
    velocity_variance = acceleration_variance * step_s**2
    for axis in range(3):
        velocity_axis = axis + 3
        process_noise[axis, axis] = position_variance
        process_noise[axis, velocity_axis] = position_velocity_covariance
        process_noise[velocity_axis, axis] = position_velocity_covariance
        process_noise[velocity_axis, velocity_axis] = velocity_variance
    return cast(FloatArray, process_noise)


def _propagate_covariance(
    covariance: FloatArray,
    transition: FloatArray,
    process_noise: FloatArray,
) -> FloatArray:
    propagated = transition @ covariance @ transition.T + process_noise
    return cast(FloatArray, 0.5 * (propagated + propagated.T))


def _covariance_transition_model(
    scenario: Scenario,
    *,
    force_model: ForceModelName,
    maneuver_schedule: list[_ScheduledManeuver],
    covariance: FloatArray | None,
) -> str:
    model = scenario.covariance_state_transition_model
    if covariance is None or model == "finite_difference":
        return model
    if force_model is not ForceModelName.TWO_BODY:
        raise ValueError(
            "two_body_variational covariance propagation requires two_body gravity"
        )
    if maneuver_schedule:
        raise ValueError(
            "two_body_variational covariance propagation does not support maneuvers"
        )
    return model


def _covariance_product_model(covariance_transition_model: str) -> str:
    if covariance_transition_model == "finite_difference":
        return "finite_difference_state_transition"
    return covariance_transition_model


def _covariance_metadata(
    scenario: Scenario,
    covariance: FloatArray | None,
    covariance_transition_model: str,
) -> dict[str, Any]:
    if covariance is None:
        return {}
    process_noise_acceleration = scenario.covariance_process_noise_acceleration_km_s2
    metadata: dict[str, Any] = {
        "covariance_model": _covariance_product_model(covariance_transition_model),
        "covariance_state_transition_storage": "per_sample_and_accumulated",
        "covariance_process_noise": _covariance_process_noise_model(process_noise_acceleration),
        "covariance_process_noise_acceleration_km_s2": process_noise_acceleration,
        "covariance_process_noise_storage": "per_sample_matrix",
    }
    if covariance_transition_model == "finite_difference":
        metadata |= {
            "covariance_finite_difference_relative_step": _COVARIANCE_FD_REL_STEP,
            "covariance_finite_difference_absolute_step": _COVARIANCE_FD_ABS_STEP,
        }
    else:
        metadata |= {
            "covariance_variational_dynamics": "two_body_acceleration_jacobian",
        }
    return metadata


def _covariance_sample_metadata(
    scenario: Scenario,
    *,
    step_index: int,
    transition_step_s: float,
    covariance_transition_model: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "covariance_sample_role": "initial" if step_index == 0 else "propagated",
        "covariance_model": _covariance_product_model(covariance_transition_model),
        "state_transition_model": (
            "identity" if step_index == 0 else covariance_transition_model
        ),
        "transition_step_s": transition_step_s,
        "process_noise_model": (
            "none"
            if step_index == 0
            else _covariance_process_noise_model(
                scenario.covariance_process_noise_acceleration_km_s2
            )
        ),
        "process_noise_acceleration_km_s2": (
            0.0 if step_index == 0 else scenario.covariance_process_noise_acceleration_km_s2
        ),
    }
    if covariance_transition_model == "finite_difference":
        metadata |= {
            "finite_difference_relative_step": _COVARIANCE_FD_REL_STEP,
            "finite_difference_absolute_step": _COVARIANCE_FD_ABS_STEP,
        }
    else:
        metadata |= {
            "variational_dynamics": "two_body_acceleration_jacobian",
        }
    return metadata


def propagate_local(scenario: Scenario) -> Trajectory:
    force_model = _validate_local_force_config(scenario.force_model)
    internal_substeps_per_sample, internal_step_s = _internal_step_schedule(
        scenario.propagation.step_s
    )
    maneuver_schedule = _scheduled_maneuvers(scenario)
    maneuver_events = _maneuver_events(maneuver_schedule, scenario)
    applied_impulses: set[int] = set()
    covariance_matrix = _initial_covariance_matrix(scenario)
    covariance_transition_model = _covariance_transition_model(
        scenario,
        force_model=force_model,
        maneuver_schedule=maneuver_schedule,
        covariance=covariance_matrix,
    )
    process_noise = _process_noise_covariance(
        scenario.covariance_process_noise_acceleration_km_s2,
        scenario.propagation.step_s,
    )
    zero_process_noise = np.zeros((6, 6), dtype=np.float64)
    identity_transition = np.eye(6, dtype=np.float64)
    accumulated_transition = identity_transition.copy()
    previous_transition = identity_transition.copy()
    previous_process_noise = zero_process_noise
    previous_transition_step_s = 0.0
    covariance_history: list[CovarianceSample] = []

    initial = scenario.initial_state.cartesian
    state = cast(FloatArray, np.concatenate([initial.position_array(), initial.velocity_array()]))
    mass_kg = float(scenario.spacecraft.mass_kg)
    samples: list[TrajectorySample] = []

    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(
                    position_km=_vector3_from_array(state[:3]),
                    velocity_km_s=_vector3_from_array(state[3:]),
                ),
                mass_kg=mass_kg,
            )
        )
        if covariance_matrix is not None:
            covariance_history.append(
                _covariance_sample(
                    epoch,
                    covariance_matrix,
                    state_transition_matrix=previous_transition,
                    accumulated_state_transition_matrix=accumulated_transition,
                    process_noise_covariance=previous_process_noise,
                    metadata=_covariance_sample_metadata(
                        scenario,
                        step_index=step_index,
                        transition_step_s=previous_transition_step_s,
                        covariance_transition_model=covariance_transition_model,
                    ),
                )
            )

        if step_index < scenario.propagation.sample_count - 1:
            transition: FloatArray | None = None
            step_start_s = step_index * scenario.propagation.step_s
            step_target_s = (step_index + 1) * scenario.propagation.step_s
            if maneuver_schedule:
                if covariance_matrix is not None:
                    def advance_trial_with_maneuvers(
                        trial_state: FloatArray,
                        *,
                        mass_kg: float = mass_kg,
                        step_start_s: float = step_start_s,
                        step_target_s: float = step_target_s,
                    ) -> FloatArray:
                        return _advance_with_maneuvers(
                            trial_state,
                            mass_kg=mass_kg,
                            start_s=step_start_s,
                            target_s=step_target_s,
                            force_model=force_model,
                            schedule=maneuver_schedule,
                            applied_impulses=set(applied_impulses),
                        )[0]

                    transition = _finite_difference_state_transition(
                        state,
                        advance_trial_with_maneuvers,
                    )
                state, mass_kg = _advance_with_maneuvers(
                    state,
                    mass_kg=mass_kg,
                    start_s=step_start_s,
                    target_s=step_target_s,
                    force_model=force_model,
                    schedule=maneuver_schedule,
                    applied_impulses=applied_impulses,
                )
            else:
                if covariance_matrix is not None:
                    if covariance_transition_model == "two_body_variational":
                        state, transition = _two_body_variational_step(
                            state,
                            scenario.propagation.step_s,
                        )
                    else:
                        transition = _finite_difference_state_transition(
                            state,
                            lambda trial_state: rk4_step(
                                trial_state,
                                scenario.propagation.step_s,
                                force_model,
                            ),
                        )
                        state = rk4_step(state, scenario.propagation.step_s, force_model)
                else:
                    state = rk4_step(state, scenario.propagation.step_s, force_model)

            if covariance_matrix is not None and transition is not None:
                covariance_matrix = _propagate_covariance(
                    covariance_matrix,
                    transition,
                    process_noise,
                )
                accumulated_transition = cast(FloatArray, transition @ accumulated_transition)
                previous_transition = transition
                previous_process_noise = process_noise
                previous_transition_step_s = scenario.propagation.step_s

    mass_metadata = (
        {
            "initial_mass_kg": scenario.spacecraft.mass_kg,
            "final_mass_kg": mass_kg,
        }
        if maneuver_schedule
        else {}
    )
    metadata = (
        {
            "integrator": "rk4",
            "step_s": scenario.propagation.step_s,
            "sample_step_s": scenario.propagation.step_s,
            "internal_max_step_s": _MAX_RK4_INTERNAL_STEP_S,
            "internal_substeps_per_sample": internal_substeps_per_sample,
            "internal_step_s": internal_step_s,
        }
        | mass_metadata
        | _maneuver_metadata(maneuver_schedule)
        | _covariance_metadata(scenario, covariance_matrix, covariance_transition_model)
    )

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="local",
        events=maneuver_events,
        maneuvers=scenario.maneuvers,
        covariance_history=covariance_history,
        metadata=metadata,
    )
