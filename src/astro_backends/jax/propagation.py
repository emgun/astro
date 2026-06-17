from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from math import ceil, isfinite
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from astro_backends.jax.runtime import JaxRuntime, load_jax_runtime
from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    EstimateResult,
    ForceModelName,
    MeasurementRecord,
    MeasurementType,
    OdSensitivityResult,
    Scenario,
    Trajectory,
    TrajectorySample,
    Vector3,
)
from astro_dynamics.monte_carlo import MonteCarloCase, MonteCarloResult

JaxRuntimeLoader = Callable[[], JaxRuntime]
JaxResearchRunner = Callable[[Scenario, JaxRuntime, int, float, float, int], MonteCarloResult]
FloatArray = NDArray[np.float64]
_MAX_RK4_INTERNAL_STEP_S = 30.0
_STATE_DIMENSION = 6
_SUPPORTED_JAX_OD_MEASUREMENTS = {MeasurementType.RANGE, MeasurementType.RANGE_RATE}
_SUPPORTED_JAX_GRAVITY_MODELS = {
    ForceModelName.TWO_BODY,
    ForceModelName.J2,
    ForceModelName.OREKIT_HIGH_FIDELITY,
}
_EXPONENTIAL_ATMOSPHERE_REFERENCE_DENSITY_KG_M3 = 4.0e-13
_EXPONENTIAL_ATMOSPHERE_REFERENCE_ALTITUDE_M = 400_000.0
_EXPONENTIAL_ATMOSPHERE_SCALE_HEIGHT_M = 60_000.0
_SOLAR_RADIATION_PRESSURE_N_M2 = 4.56e-6
_RESEARCH_FORCE_MODEL_POLICY = "screening_only_not_operational_ephemeris"


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


def _research_force_model_names(scenario: Scenario) -> list[str]:
    force_models = ["J2" if _uses_j2_baseline(scenario.force_model.gravity) else "two_body"]
    if scenario.force_model.atmospheric_drag:
        force_models.append("exponential_atmospheric_drag")
    if scenario.force_model.solar_radiation_pressure:
        force_models.append("constant_direction_solar_radiation_pressure")
    return force_models


def _uses_j2_baseline(force_model: ForceModelName) -> bool:
    return force_model in {ForceModelName.J2, ForceModelName.OREKIT_HIGH_FIDELITY}


def _drag_acceleration(jnp: Any, position: Any, velocity: Any, scenario: Scenario) -> Any:
    radius_km = jnp.linalg.norm(position, axis=1)
    altitude_m = (radius_km - R_EARTH_KM) * 1000.0
    density_kg_m3 = _EXPONENTIAL_ATMOSPHERE_REFERENCE_DENSITY_KG_M3 * jnp.exp(
        -(altitude_m - _EXPONENTIAL_ATMOSPHERE_REFERENCE_ALTITUDE_M)
        / _EXPONENTIAL_ATMOSPHERE_SCALE_HEIGHT_M
    )
    velocity_m_s = velocity * 1000.0
    speed_m_s = jnp.linalg.norm(velocity_m_s, axis=1)
    coefficient = (
        0.5
        * density_kg_m3
        * scenario.spacecraft.drag_coefficient
        * scenario.spacecraft.area_m2
        / scenario.spacecraft.mass_kg
    )
    return -(coefficient * speed_m_s)[:, None] * velocity_m_s / 1000.0


def _solar_radiation_pressure_acceleration(jnp: Any, position: Any, scenario: Scenario) -> Any:
    acceleration_km_s2 = (
        _SOLAR_RADIATION_PRESSURE_N_M2
        * scenario.spacecraft.reflectivity_coefficient
        * scenario.spacecraft.area_m2
        / scenario.spacecraft.mass_kg
        / 1000.0
    )
    direction = jnp.asarray((1.0, 0.0, 0.0))
    return acceleration_km_s2 * jnp.broadcast_to(direction, position.shape)


def _acceleration(jnp: Any, position: Any, velocity: Any, scenario: Scenario) -> Any:
    acceleration = _two_body_acceleration(jnp, position)
    if _uses_j2_baseline(scenario.force_model.gravity):
        acceleration = acceleration + _j2_acceleration(jnp, position)
    if scenario.force_model.atmospheric_drag:
        acceleration = acceleration + _drag_acceleration(jnp, position, velocity, scenario)
    if scenario.force_model.solar_radiation_pressure:
        acceleration = acceleration + _solar_radiation_pressure_acceleration(
            jnp, position, scenario
        )
    return acceleration


def _derivative(jnp: Any, state: Any, scenario: Scenario) -> Any:
    position = state[:, :3]
    velocity = state[:, 3:]
    return jnp.concatenate((velocity, _acceleration(jnp, position, velocity, scenario)), axis=1)


def _rk4_step_once(
    jnp: Any,
    state: Any,
    step_s: float,
    scenario: Scenario,
) -> Any:
    k1 = _derivative(jnp, state, scenario)
    k2 = _derivative(jnp, state + 0.5 * step_s * k1, scenario)
    k3 = _derivative(jnp, state + 0.5 * step_s * k2, scenario)
    k4 = _derivative(jnp, state + step_s * k3, scenario)
    return state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_step(
    jnp: Any,
    state: Any,
    step_s: float,
    scenario: Scenario,
) -> Any:
    substep_count, substep_s = _internal_step_schedule(step_s)
    next_state = state
    for _ in range(substep_count):
        next_state = _rk4_step_once(jnp, next_state, substep_s, scenario)
    return next_state


def _validate_jax_force_model(scenario: Scenario) -> None:
    if scenario.force_model.third_body_gravity:
        raise UnsupportedBackendError(
            "JAX research propagation does not support third_body_gravity; "
            "validated Sun/Moon ephemerides remain an Orekit/Tudat integration boundary."
        )
    if scenario.force_model.gravity not in _SUPPORTED_JAX_GRAVITY_MODELS:
        raise UnsupportedBackendError(
            "JAX research propagation currently supports only two_body, j2, and "
            "orekit_high_fidelity screening force models"
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
                scenario,
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
                "research_force_models": _research_force_model_names(scenario),
                "research_force_model_policy": _RESEARCH_FORCE_MODEL_POLICY,
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
            "research_force_models": _research_force_model_names(scenario),
            "research_force_model_policy": _RESEARCH_FORCE_MODEL_POLICY,
            "case_count": cases,
        },
    )


def _propagate_nominal_final_state(
    jnp: Any,
    scenario: Scenario,
    state_vector: Any,
) -> Any:
    state = jnp.reshape(state_vector, (1, 6))
    for _sample_index in range(scenario.propagation.sample_count - 1):
        state = _rk4_step(
            jnp,
            state,
            scenario.propagation.step_s,
            scenario,
        )
    return state[0]


def _initial_state_vector(scenario: Scenario) -> FloatArray:
    return np.concatenate(
        (
            np.asarray(scenario.initial_state.cartesian.position_km, dtype=np.float64),
            np.asarray(scenario.initial_state.cartesian.velocity_km_s, dtype=np.float64),
        )
    )


def _jax_state_history(jnp: Any, scenario: Scenario, state_vector: Any) -> Any:
    state = jnp.reshape(state_vector, (1, _STATE_DIMENSION))
    samples = []
    for sample_index in range(scenario.propagation.sample_count):
        samples.append(state[0])
        if sample_index < scenario.propagation.sample_count - 1:
            state = _rk4_step(
                jnp,
                state,
                scenario.propagation.step_s,
                scenario,
            )
    return jnp.stack(samples, axis=0)


def _measurement_sample_index(scenario: Scenario, record: MeasurementRecord) -> int:
    elapsed_s = (record.epoch - scenario.initial_state.epoch).total_seconds()
    sample_index = round(elapsed_s / scenario.propagation.step_s)
    expected_elapsed_s = sample_index * scenario.propagation.step_s
    if (
        sample_index < 0
        or sample_index >= scenario.propagation.sample_count
        or abs(elapsed_s - expected_elapsed_s) > 1.0e-9
    ):
        raise UnsupportedBackendError(
            "JAX OD sensitivity requires measurement epochs to align with propagation samples"
        )
    return sample_index


def _jax_od_measurement_specs(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
) -> list[tuple[MeasurementType, int, FloatArray, float, float]]:
    if not measurements:
        raise UnsupportedBackendError("JAX OD sensitivity requires at least one measurement")
    stations = {station.name: station for station in scenario.ground_stations}
    specs: list[tuple[MeasurementType, int, FloatArray, float, float]] = []
    for record in measurements:
        if record.measurement_type not in _SUPPORTED_JAX_OD_MEASUREMENTS:
            raise UnsupportedBackendError(
                "JAX OD sensitivity currently supports only range and range_rate measurements; "
                f"unsupported measurement type: {record.measurement_type}"
            )
        if record.observed_object != scenario.spacecraft.name:
            raise UnsupportedBackendError(
                "JAX OD sensitivity measurement observed object "
                f"{record.observed_object!r} does not match scenario spacecraft "
                f"{scenario.spacecraft.name!r}"
            )
        try:
            station = stations[record.observer]
        except KeyError as exc:
            raise UnsupportedBackendError(
                "JAX OD sensitivity measurement observer "
                f"{record.observer!r} is not in the scenario"
            ) from exc
        sample_index = _measurement_sample_index(scenario, record)
        station_position = station.position_array(record.epoch, scenario.earth_orientation)
        specs.append(
            (
                record.measurement_type,
                sample_index,
                np.asarray(station_position, dtype=np.float64),
                float(record.value),
                float(record.sigma),
            )
        )
    return specs


def _jax_od_residual_vector(
    jnp: Any,
    scenario: Scenario,
    measurement_specs: list[tuple[MeasurementType, int, FloatArray, float, float]],
    state_vector: Any,
) -> Any:
    state_history = _jax_state_history(jnp, scenario, state_vector)
    residuals = []
    for measurement_type, sample_index, station_position, value, sigma in measurement_specs:
        sample_state = state_history[sample_index]
        position = sample_state[:3]
        velocity = sample_state[3:]
        station = jnp.asarray(station_position)
        relative_position = position - station
        distance = jnp.linalg.norm(relative_position)
        if measurement_type is MeasurementType.RANGE:
            predicted = distance
        else:
            line_of_sight = relative_position / distance
            predicted = jnp.dot(velocity, line_of_sight)
        residuals.append((predicted - value) / sigma)
    return jnp.stack(residuals)


def _unique_measurement_type_values(measurements: list[MeasurementRecord]) -> list[str]:
    values: list[str] = []
    for measurement in measurements:
        value = measurement.measurement_type.value
        if value not in values:
            values.append(value)
    return values


def _jax_od_residuals_and_jacobian(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    runtime: JaxRuntime,
    state_vector: FloatArray,
) -> tuple[FloatArray, FloatArray, list[tuple[MeasurementType, int, FloatArray, float, float]]]:
    jacfwd = getattr(runtime.jax_module, "jacfwd", None)
    if jacfwd is None:
        raise UnsupportedBackendError(
            "JAX OD sensitivity requires jax.jacfwd; install a complete "
            "astro-suite[research] JAX runtime."
        )

    jnp = runtime.jnp_module
    measurement_specs = _jax_od_measurement_specs(scenario, measurements)

    def residual_function(candidate_state_vector: Any) -> Any:
        return _jax_od_residual_vector(
            jnp,
            scenario,
            measurement_specs,
            candidate_state_vector,
        )

    residuals = np.asarray(residual_function(jnp.asarray(state_vector)), dtype=np.float64)
    jacobian = np.asarray(
        jacfwd(residual_function)(jnp.asarray(state_vector)),
        dtype=np.float64,
    )
    return residuals, jacobian, measurement_specs


def _final_state_transition_matrix(
    scenario: Scenario,
    runtime: JaxRuntime,
) -> list[list[float]]:
    _validate_jax_force_model(scenario)
    jacfwd = getattr(runtime.jax_module, "jacfwd", None)
    if jacfwd is None:
        raise UnsupportedBackendError(
            "JAX sensitivity propagation requires jax.jacfwd; install a complete "
            "astro-suite[research] JAX runtime."
        )

    jnp = runtime.jnp_module
    initial = _initial_state_vector(scenario)

    def final_state(state_vector: Any) -> Any:
        return _propagate_nominal_final_state(jnp, scenario, state_vector)

    transition_matrix = np.asarray(jacfwd(final_state)(jnp.asarray(initial)), dtype=np.float64)
    return [[float(component) for component in row] for row in transition_matrix]


def research_od_sensitivity_jax(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    *,
    runtime_loader: JaxRuntimeLoader = load_jax_runtime,
) -> OdSensitivityResult:
    _validate_jax_force_model(scenario)
    runtime = runtime_loader()
    initial = _initial_state_vector(scenario)
    residuals, jacobian, _measurement_specs = _jax_od_residuals_and_jacobian(
        scenario,
        measurements,
        runtime,
        initial,
    )
    return OdSensitivityResult(
        scenario_id=scenario.scenario_id,
        backend="jax",
        measurement_count=len(measurements),
        state_dimension=_STATE_DIMENSION,
        residuals=[float(value) for value in residuals],
        jacobian=[[float(component) for component in row] for row in jacobian],
        metadata={
            "adapter": "jax",
            "jax_version": runtime.jax_version,
            "jaxlib_version": runtime.jaxlib_version,
            "sensitivity_model": "jax_jacfwd_od_residuals",
            "force_model": scenario.force_model.gravity.value,
            "research_force_models": _research_force_model_names(scenario),
            "research_force_model_policy": _RESEARCH_FORCE_MODEL_POLICY,
            "measurement_types": _unique_measurement_type_values(measurements),
            "residual_convention": "(predicted - observed) / sigma",
        },
    )


def research_estimate_jax(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    *,
    runtime_loader: JaxRuntimeLoader = load_jax_runtime,
    max_iterations: int = 5,
    correction_tolerance: float = 1.0e-9,
    residual_tolerance: float = 1.0e-6,
) -> EstimateResult:
    if isinstance(max_iterations, bool) or max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if correction_tolerance <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("estimator tolerances must be positive")

    _validate_jax_force_model(scenario)
    runtime = runtime_loader()
    state_vector = _initial_state_vector(scenario)
    residuals: FloatArray | None = None
    jacobian: FloatArray | None = None
    measurement_specs: list[tuple[MeasurementType, int, FloatArray, float, float]] = []
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        residuals, jacobian, measurement_specs = _jax_od_residuals_and_jacobian(
            scenario,
            measurements,
            runtime,
            state_vector,
        )
        correction = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
        state_vector = cast(FloatArray, state_vector + correction)
        iterations = iteration
        correction_norm = float(np.linalg.norm(correction))

        residuals, jacobian, measurement_specs = _jax_od_residuals_and_jacobian(
            scenario,
            measurements,
            runtime,
            state_vector,
        )
        rms = float(np.sqrt(np.mean(residuals * residuals)))
        if correction_norm <= correction_tolerance or rms <= residual_tolerance:
            converged = True
            break

    if residuals is None or jacobian is None:
        raise UnsupportedBackendError("JAX research estimator did not evaluate residuals")

    normal_matrix = jacobian.T @ jacobian
    covariance_status = "available"
    try:
        covariance_matrix = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        covariance_matrix = np.linalg.pinv(normal_matrix)
        covariance_status = "pseudo_inverse"
    covariance_matrix = cast(FloatArray, 0.5 * (covariance_matrix + covariance_matrix.T))
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    estimated_cartesian = scenario.initial_state.cartesian.model_copy(
        update={
            "position_km": _tuple3(state_vector[:3]),
            "velocity_km_s": _tuple3(state_vector[3:]),
        }
    )
    estimated_state = scenario.initial_state.model_copy(
        update={"cartesian": estimated_cartesian}
    )

    return EstimateResult(
        estimated_state=estimated_state,
        residuals=[float(value) for value in residuals],
        covariance=[[float(component) for component in row] for row in covariance_matrix],
        rms=rms,
        iterations=iterations,
        converged=converged,
        metadata={
            "adapter": "jax",
            "backend": "jax_research_estimator",
            "estimator": "jax_research_gauss_newton",
            "jax_version": runtime.jax_version,
            "jaxlib_version": runtime.jaxlib_version,
            "sensitivity_model": "jax_jacfwd_od_residuals",
            "force_model": scenario.force_model.gravity.value,
            "research_force_models": _research_force_model_names(scenario),
            "research_force_model_policy": _RESEARCH_FORCE_MODEL_POLICY,
            "measurement_types": _unique_measurement_type_values(measurements),
            "measurement_count": len(measurement_specs),
            "residual_convention": "(predicted - observed) / sigma",
            "max_iterations": max_iterations,
            "correction_tolerance": correction_tolerance,
            "residual_tolerance": residual_tolerance,
            "covariance_status": covariance_status,
            "covariance_convention": "inverse_normal_matrix_of_normalized_residuals",
        },
    )


def _with_jax_sensitivities(
    result: MonteCarloResult,
    scenario: Scenario,
    runtime: JaxRuntime,
    *,
    include_sensitivities: bool,
) -> MonteCarloResult:
    if not include_sensitivities:
        return result
    metadata = {
        **result.metadata,
        "sensitivity_model": "jax_jacfwd_final_state_transition",
        "final_state_transition_matrix": _final_state_transition_matrix(scenario, runtime),
    }
    return result.model_copy(update={"metadata": metadata})


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
    include_sensitivities: bool = False,
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
    result_with_sensitivities = _with_jax_sensitivities(
        result,
        scenario,
        runtime,
        include_sensitivities=include_sensitivities,
    )
    return _with_jax_provenance(result_with_sensitivities, runtime)
