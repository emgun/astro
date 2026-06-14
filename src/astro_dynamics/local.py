from __future__ import annotations

from datetime import timedelta
from typing import cast

import numpy as np
from numpy.typing import NDArray

from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
    Vector3,
)

FloatArray = NDArray[np.float64]


def two_body_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    radius = float(np.linalg.norm(position_km))
    return -MU_EARTH_KM3_S2 * position_km / radius**3


def j2_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    x, y, z = position_km
    radius2 = float(np.dot(position_km, position_km))
    radius = radius2**0.5
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
    acceleration = two_body_acceleration_km_s2(position_km)
    if force_model is ForceModelName.J2:
        acceleration = acceleration + j2_acceleration_km_s2(position_km)
    return acceleration


def derivative(state: FloatArray, force_model: ForceModelName) -> FloatArray:
    position = state[:3]
    velocity = state[3:]
    return cast(FloatArray, np.concatenate([velocity, acceleration_km_s2(position, force_model)]))


def rk4_step(state: FloatArray, step_s: float, force_model: ForceModelName) -> FloatArray:
    k1 = derivative(state, force_model)
    k2 = derivative(state + 0.5 * step_s * k1, force_model)
    k3 = derivative(state + 0.5 * step_s * k2, force_model)
    k4 = derivative(state + step_s * k3, force_model)
    return cast(FloatArray, state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def propagate_local(scenario: Scenario) -> Trajectory:
    force_model = scenario.force_model.gravity
    if force_model is ForceModelName.OREKIT_HIGH_FIDELITY:
        raise ValueError("Local backend supports only two_body and j2 force models")

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
        if step_index < scenario.propagation.sample_count - 1:
            state = rk4_step(state, scenario.propagation.step_s, force_model)

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="local",
        metadata={"integrator": "rk4", "step_s": scenario.propagation.step_s},
    )
