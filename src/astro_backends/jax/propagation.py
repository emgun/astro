from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from math import ceil, isfinite
from typing import Any

import numpy as np

from astro_backends.jax.runtime import JaxRuntime, load_jax_runtime
from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
    Vector3,
)
from astro_dynamics.monte_carlo import MonteCarloCase, MonteCarloResult

JaxRuntimeLoader = Callable[[], JaxRuntime]
JaxResearchRunner = Callable[[Scenario, JaxRuntime, int, float, float, int], MonteCarloResult]
_MAX_RK4_INTERNAL_STEP_S = 30.0


def _validate_research_inputs(
    *,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
) -> None:
    if isinstance(cases, bool) or cases <= 0:
        raise ValueError("cases must be positive")
    if not isfinite(position_sigma_km) or not isfinite(velocity_sigma_km_s):
        raise ValueError("sigmas must be finite")
    if position_sigma_km < 0.0 or velocity_sigma_km_s < 0.0:
        raise ValueError("sigmas must be nonnegative")


def _tuple3(values: Any) -> Vector3:
    array = np.asarray(values, dtype=np.float64)
    return (float(array[0]), float(array[1]), float(array[2]))


def _internal_step_schedule(step_s: float) -> tuple[int, float]:
    substep_count = max(1, ceil(abs(step_s) / _MAX_RK4_INTERNAL_STEP_S))
    return substep_count, step_s / substep_count


_SUPPORTED_JAX_GRAVITY_MODELS = {ForceModelName.TWO_BODY, ForceModelName.J2}


def _two_body_acceleration(jnp: Any, position: Any) -> Any:
    radius = jnp.linalg.norm(position, axis=1)
    return -MU_EARTH_KM3_S2 * position / (radius[:, None] ** 3)


def _j2_acceleration(jnp: Any, position: Any) -> Any:
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    radius2 = jnp.sum(position * position, axis=1)
    radius = jnp.sqrt(radius2)
    z2_over_r2 = (z * z) / radius2
    factor = 1.5 * J2_EARTH * MU_EARTH_KM3_S2 * R_EARTH_KM**2 / (radius**5)
    return factor[:, None] * jnp.stack(
        (
            x * (5.0 * z2_over_r2 - 1.0),
            y * (5.0 * z2_over_r2 - 1.0),
            z * (5.0 * z2_over_r2 - 3.0),
        ),
        axis=1,
    )


def _acceleration(jnp: Any, position: Any, force_model: ForceModelName) -> Any:
    acceleration = _two_body_acceleration(jnp, position)
    if force_model is ForceModelName.J2:
        acceleration = acceleration + _j2_acceleration(jnp, position)
    return acceleration


def _derivative(jnp: Any, state: Any, force_model: ForceModelName) -> Any:
    position = state[:, :3]
    velocity = state[:, 3:]
    return jnp.concatenate((velocity, _acceleration(jnp, position, force_model)), axis=1)


def _rk4_step_once(
    jnp: Any,
    state: Any,
    step_s: float,
    force_model: ForceModelName,
) -> Any:
    k1 = _derivative(jnp, state, force_model)
    k2 = _derivative(jnp, state + 0.5 * step_s * k1, force_model)
    k3 = _derivative(jnp, state + 0.5 * step_s * k2, force_model)
    k4 = _derivative(jnp, state + step_s * k3, force_model)
    return state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_step(
    jnp: Any,
    state: Any,
    step_s: float,
    force_model: ForceModelName,
) -> Any:
    substep_count, substep_s = _internal_step_schedule(step_s)
    next_state = state
    for _ in range(substep_count):
        next_state = _rk4_step_once(jnp, next_state, substep_s, force_model)
    return next_state


def _validate_jax_force_model(scenario: Scenario) -> None:
    enabled_flags = scenario.force_model.enabled_high_fidelity_flags()
    if enabled_flags:
        raise UnsupportedBackendError(
            "JAX research propagation does not support high-fidelity force model flags: "
            f"{', '.join(enabled_flags)}"
        )
    if scenario.force_model.gravity not in _SUPPORTED_JAX_GRAVITY_MODELS:
        raise UnsupportedBackendError(
            "JAX research propagation currently supports only two_body and j2 force models"
        )


def _default_jax_two_body_runner(
    scenario: Scenario,
    runtime: JaxRuntime,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    seed: int,
) -> MonteCarloResult:
    _validate_research_inputs(
        cases=cases,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
    )
    _validate_jax_force_model(scenario)

    rng = np.random.default_rng(seed)
    position_delta = rng.normal(0.0, position_sigma_km, size=(cases, 3))
    velocity_delta = rng.normal(0.0, velocity_sigma_km_s, size=(cases, 3))
    base_position = np.asarray(scenario.initial_state.cartesian.position_km, dtype=np.float64)
    base_velocity = np.asarray(scenario.initial_state.cartesian.velocity_km_s, dtype=np.float64)
    jnp = runtime.jnp_module
    state = jnp.concatenate(
        (
            jnp.asarray(base_position + position_delta),
            jnp.asarray(base_velocity + velocity_delta),
        ),
        axis=1,
    )

    sample_states = []
    for sample_index in range(scenario.propagation.sample_count):
        sample_states.append(state)
        if sample_index < scenario.propagation.sample_count - 1:
            state = _rk4_step(
                jnp,
                state,
                scenario.propagation.step_s,
                scenario.force_model.gravity,
            )

    sample_array = np.asarray(jnp.stack(sample_states, axis=1), dtype=np.float64)
    internal_substeps_per_sample, internal_step_s = _internal_step_schedule(
        scenario.propagation.step_s
    )
    monte_carlo_cases: list[MonteCarloCase] = []

    for case_index in range(cases):
        initial_cartesian = CartesianState(
            position_km=_tuple3(sample_array[case_index, 0, :3]),
            velocity_km_s=_tuple3(sample_array[case_index, 0, 3:]),
        )
        initial_state = scenario.initial_state.model_copy(
            update={"cartesian": initial_cartesian}
        )
        trajectory_samples = [
            TrajectorySample(
                epoch=scenario.initial_state.epoch
                + timedelta(seconds=sample_index * scenario.propagation.step_s),
                state=CartesianState(
                    position_km=_tuple3(sample_array[case_index, sample_index, :3]),
                    velocity_km_s=_tuple3(sample_array[case_index, sample_index, 3:]),
                ),
            )
            for sample_index in range(scenario.propagation.sample_count)
        ]
        trajectory = Trajectory(
            scenario_id=scenario.scenario_id,
            samples=trajectory_samples,
            force_model=scenario.force_model,
            backend="jax",
            metadata={
                "integrator": "rk4",
                "step_s": scenario.propagation.step_s,
                "sample_step_s": scenario.propagation.step_s,
                "internal_max_step_s": _MAX_RK4_INTERNAL_STEP_S,
                "internal_substeps_per_sample": internal_substeps_per_sample,
                "internal_step_s": internal_step_s,
                "runner": f"jax_vectorized_{scenario.force_model.gravity.value}_rk4",
            },
        )
        monte_carlo_cases.append(
            MonteCarloCase(
                case_index=case_index,
                position_delta_km=_tuple3(position_delta[case_index]),
                velocity_delta_km_s=_tuple3(velocity_delta[case_index]),
                initial_state=initial_state,
                trajectory=trajectory,
            )
        )

    return MonteCarloResult(
        scenario_id=scenario.scenario_id,
        backend="jax",
        seed=seed,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
        cases=monte_carlo_cases,
        metadata={
            "runner": f"jax_vectorized_{scenario.force_model.gravity.value}_rk4",
            "perturbation_rng": "numpy.default_rng",
            "force_model": scenario.force_model.gravity.value,
            "case_count": cases,
        },
    )


def _with_jax_provenance(
    result: MonteCarloResult,
    runtime: JaxRuntime,
) -> MonteCarloResult:
    metadata = {
        **result.metadata,
        "adapter": "jax",
        "source_backend": result.backend,
        "jax_version": runtime.jax_version,
        "jaxlib_version": runtime.jaxlib_version,
    }
    return result.model_copy(update={"backend": "jax", "metadata": metadata})


def research_propagate_jax(
    scenario: Scenario,
    *,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    seed: int,
    runtime_loader: JaxRuntimeLoader = load_jax_runtime,
    research_runner: JaxResearchRunner | None = None,
) -> MonteCarloResult:
    runtime = runtime_loader()
    selected_runner = research_runner or _default_jax_two_body_runner

    result = selected_runner(
        scenario,
        runtime,
        cases,
        position_sigma_km,
        velocity_sigma_km_s,
        seed,
    )
    return _with_jax_provenance(result, runtime)
