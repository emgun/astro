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
            acceleration = acceleration + (
                np.array(maneuver.delta_v_km_s, dtype=np.float64) / float(maneuver.duration_s)
            )
    return cast(FloatArray, acceleration)


def _derivative_with_maneuvers(
    state: FloatArray,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
) -> FloatArray:
    position = state[:3]
    velocity = state[3:]
    total_acceleration = acceleration_km_s2(
        position,
        force_model,
    ) + _finite_burn_acceleration_km_s2(
        schedule, elapsed_s
    )
    return cast(FloatArray, np.concatenate([velocity, total_acceleration]))


def _rk4_step_once_with_maneuvers(
    state: FloatArray,
    step_s: float,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    elapsed_s: float,
) -> FloatArray:
    k1 = _derivative_with_maneuvers(state, force_model, schedule, elapsed_s)
    k2 = _derivative_with_maneuvers(
        state + 0.5 * step_s * k1,
        force_model,
        schedule,
        elapsed_s + 0.5 * step_s,
    )
    k3 = _derivative_with_maneuvers(
        state + 0.5 * step_s * k2,
        force_model,
        schedule,
        elapsed_s + 0.5 * step_s,
    )
    k4 = _derivative_with_maneuvers(
        state + step_s * k3,
        force_model,
        schedule,
        elapsed_s + step_s,
    )
    return cast(FloatArray, state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


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
    start_s: float,
    target_s: float,
    force_model: ForceModelName,
    schedule: list[_ScheduledManeuver],
    applied_impulses: set[int],
) -> FloatArray:
    elapsed_s = start_s
    next_state = _apply_impulses_at_elapsed_s(state, schedule, elapsed_s, applied_impulses)

    while elapsed_s < target_s - _MANEUVER_TIME_TOLERANCE_S:
        step_target_s = min(elapsed_s + _MAX_RK4_INTERNAL_STEP_S, target_s)
        boundary_s = _next_maneuver_boundary_s(schedule, elapsed_s, step_target_s)
        if boundary_s is not None:
            step_target_s = boundary_s

        step_s = step_target_s - elapsed_s
        next_state = _rk4_step_once_with_maneuvers(
            next_state,
            step_s,
            force_model,
            schedule,
            elapsed_s,
        )
        elapsed_s = step_target_s
        next_state = _apply_impulses_at_elapsed_s(
            next_state,
            schedule,
            elapsed_s,
            applied_impulses,
        )

    return next_state


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
    return {
        "maneuver_model": "constant_inertial_acceleration",
        "finite_burn_count": finite_burn_count,
        "impulsive_maneuver_count": impulse_count,
    }


def _initial_covariance_matrix(scenario: Scenario) -> FloatArray | None:
    if scenario.initial_covariance is None:
        return None
    return cast(FloatArray, np.array(scenario.initial_covariance, dtype=np.float64))


def _covariance_sample(epoch: datetime, covariance: FloatArray) -> CovarianceSample:
    return CovarianceSample(
        epoch=epoch,
        covariance=[[float(component) for component in row] for row in covariance],
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


def _propagate_covariance(covariance: FloatArray, transition: FloatArray) -> FloatArray:
    propagated = transition @ covariance @ transition.T
    return cast(FloatArray, 0.5 * (propagated + propagated.T))


def _covariance_metadata(covariance: FloatArray | None) -> dict[str, Any]:
    if covariance is None:
        return {}
    return {
        "covariance_model": "finite_difference_state_transition",
        "covariance_process_noise": "none",
    }


def propagate_local(scenario: Scenario) -> Trajectory:
    force_model = _validate_local_force_model(scenario.force_model.gravity)
    internal_substeps_per_sample, internal_step_s = _internal_step_schedule(
        scenario.propagation.step_s
    )
    maneuver_schedule = _scheduled_maneuvers(scenario)
    maneuver_events = _maneuver_events(maneuver_schedule, scenario)
    applied_impulses: set[int] = set()
    covariance_matrix = _initial_covariance_matrix(scenario)
    covariance_history: list[CovarianceSample] = []

    initial = scenario.initial_state.cartesian
    state = cast(FloatArray, np.concatenate([initial.position_array(), initial.velocity_array()]))
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
            )
        )
        if covariance_matrix is not None:
            covariance_history.append(_covariance_sample(epoch, covariance_matrix))

        if step_index < scenario.propagation.sample_count - 1:
            transition: FloatArray | None = None
            step_start_s = step_index * scenario.propagation.step_s
            step_target_s = (step_index + 1) * scenario.propagation.step_s
            if maneuver_schedule:
                if covariance_matrix is not None:
                    def advance_trial_with_maneuvers(
                        trial_state: FloatArray,
                        *,
                        step_start_s: float = step_start_s,
                        step_target_s: float = step_target_s,
                    ) -> FloatArray:
                        return _advance_with_maneuvers(
                            trial_state,
                            start_s=step_start_s,
                            target_s=step_target_s,
                            force_model=force_model,
                            schedule=maneuver_schedule,
                            applied_impulses=set(applied_impulses),
                        )

                    transition = _finite_difference_state_transition(
                        state,
                        advance_trial_with_maneuvers,
                    )
                state = _advance_with_maneuvers(
                    state,
                    start_s=step_start_s,
                    target_s=step_target_s,
                    force_model=force_model,
                    schedule=maneuver_schedule,
                    applied_impulses=applied_impulses,
                )
            else:
                if covariance_matrix is not None:
                    transition = _finite_difference_state_transition(
                        state,
                        lambda trial_state: rk4_step(
                            trial_state,
                            scenario.propagation.step_s,
                            force_model,
                        ),
                    )
                state = rk4_step(state, scenario.propagation.step_s, force_model)

            if covariance_matrix is not None and transition is not None:
                covariance_matrix = _propagate_covariance(covariance_matrix, transition)

    metadata = {
        "integrator": "rk4",
        "step_s": scenario.propagation.step_s,
        "sample_step_s": scenario.propagation.step_s,
        "internal_max_step_s": _MAX_RK4_INTERNAL_STEP_S,
        "internal_substeps_per_sample": internal_substeps_per_sample,
        "internal_step_s": internal_step_s,
    } | _maneuver_metadata(maneuver_schedule) | _covariance_metadata(covariance_matrix)

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
