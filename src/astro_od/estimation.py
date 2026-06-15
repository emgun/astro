from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from astro_core.errors import NumericalConvergenceError
from astro_core.models import (
    CartesianState,
    EstimateResult,
    MeasurementRecord,
    MeasurementType,
    Scenario,
    Trajectory,
)
from astro_dynamics.backends import propagate_with_backend
from astro_od.measurements import declination_deg, range_km, range_rate_km_s, right_ascension_deg

FloatArray = NDArray[np.float64]
Propagator = Callable[[Scenario], Trajectory]
STATE_DIMENSION = 6
JACOBIAN_RANK_RTOL = 1.0e-7


def _vector3_from_state_vector(state_vector: FloatArray, start: int) -> tuple[float, float, float]:
    return (
        float(state_vector[start]),
        float(state_vector[start + 1]),
        float(state_vector[start + 2]),
    )


def scenario_with_state_vector(scenario: Scenario, state_vector: FloatArray) -> Scenario:
    cartesian = CartesianState(
        position_km=_vector3_from_state_vector(state_vector, 0),
        velocity_km_s=_vector3_from_state_vector(state_vector, 3),
    )
    initial_state = scenario.initial_state.model_copy(update={"cartesian": cartesian})
    return scenario.model_copy(update={"initial_state": initial_state})


def _station_position_for_observer(scenario: Scenario, observer: str) -> FloatArray:
    for station in scenario.ground_stations:
        if station.name == observer:
            return station.position_array()
    raise NumericalConvergenceError(f"Measurement observer {observer!r} is not in the scenario")


def _propagator_for_backend(backend: str) -> Propagator:
    def propagate(scenario: Scenario) -> Trajectory:
        return propagate_with_backend(scenario, backend)

    return propagate


def _validate_measurements(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    propagator: Propagator,
) -> None:
    if not measurements:
        raise NumericalConvergenceError("At least one measurement is required for estimation")
    if len(measurements) < STATE_DIMENSION:
        raise NumericalConvergenceError(
            f"At least {STATE_DIMENSION} measurements are required for 6-state estimation"
        )

    observer_names = {station.name for station in scenario.ground_stations}
    for measurement in measurements:
        if measurement.observed_object != scenario.spacecraft.name:
            raise NumericalConvergenceError(
                "Measurement observed object "
                f"{measurement.observed_object!r} does not match scenario spacecraft "
                f"{scenario.spacecraft.name!r}"
            )
        if measurement.observer not in observer_names:
            raise NumericalConvergenceError(
                f"Measurement observer {measurement.observer!r} is not in the scenario"
            )

    propagated_epochs = {sample.epoch for sample in propagator(scenario).samples}
    for measurement in measurements:
        if measurement.epoch not in propagated_epochs:
            epoch = measurement.epoch.isoformat()
            raise NumericalConvergenceError(
                f"No propagated sample is available for measurement epoch {epoch}"
            )


def _predicted_measurement(
    scenario: Scenario,
    measurement: MeasurementRecord,
    trajectory_index: dict[datetime, CartesianState],
) -> float:
    try:
        sample_state = trajectory_index[measurement.epoch]
    except KeyError as exc:
        epoch = measurement.epoch.isoformat()
        raise NumericalConvergenceError(
            f"No propagated sample is available for measurement epoch {epoch}"
        ) from exc

    spacecraft_position = sample_state.position_array()
    spacecraft_velocity = sample_state.velocity_array()
    station_position = _station_position_for_observer(scenario, measurement.observer)

    if measurement.measurement_type is MeasurementType.RANGE:
        return range_km(spacecraft_position, station_position)
    if measurement.measurement_type is MeasurementType.RANGE_RATE:
        return range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position)
    if measurement.measurement_type is MeasurementType.RIGHT_ASCENSION:
        return right_ascension_deg(spacecraft_position, station_position)
    if measurement.measurement_type is MeasurementType.DECLINATION:
        return declination_deg(spacecraft_position, station_position)
    raise NumericalConvergenceError(f"Unsupported measurement type: {measurement.measurement_type}")


def _angle_delta_deg(predicted_deg: float, observed_deg: float) -> float:
    return ((predicted_deg - observed_deg + 180.0) % 360.0) - 180.0


def _measurement_residual(predicted: float, measurement: MeasurementRecord) -> float:
    if measurement.measurement_type is MeasurementType.RIGHT_ASCENSION:
        return _angle_delta_deg(predicted, measurement.value) / measurement.sigma
    return (predicted - measurement.value) / measurement.sigma


def residual_vector(
    state_vector: FloatArray,
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    propagator: Propagator,
) -> FloatArray:
    trial_scenario = scenario_with_state_vector(scenario, state_vector)
    trajectory = propagator(trial_scenario)
    trajectory_index = {sample.epoch: sample.state for sample in trajectory.samples}

    residuals = [
        _measurement_residual(
            _predicted_measurement(trial_scenario, measurement, trajectory_index),
            measurement,
        )
        for measurement in measurements
    ]
    return np.array(residuals, dtype=np.float64)


def covariance_from_jacobian(jacobian: FloatArray, rms: float) -> list[list[float]]:
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * rms**2
    return [[float(component) for component in row] for row in covariance]


def _jacobian_diagnostics(jacobian: FloatArray) -> dict[str, Any]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    singular_value_list = [float(value) for value in singular_values]
    largest_singular_value = singular_value_list[0] if singular_value_list else 0.0
    smallest_singular_value = singular_value_list[-1] if singular_value_list else 0.0
    rank_tolerance = largest_singular_value * JACOBIAN_RANK_RTOL
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition_number = (
        float("inf")
        if smallest_singular_value <= 0.0
        else float(largest_singular_value / smallest_singular_value)
    )

    return {
        "jacobian_rank": rank,
        "jacobian_rank_tolerance": rank_tolerance,
        "singular_values": singular_value_list,
        "condition_number": condition_number,
    }


def _initial_state_vector(scenario: Scenario) -> FloatArray:
    initial = scenario.initial_state.cartesian
    return cast(
        FloatArray,
        np.concatenate([initial.position_array(), initial.velocity_array()]),
    )


def estimate_initial_state(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    *,
    backend: str = "local",
    propagator: Propagator | None = None,
) -> EstimateResult:
    selected_propagator = propagator or _propagator_for_backend(backend)
    _validate_measurements(scenario, measurements, selected_propagator)

    optimizer_result = least_squares(
        residual_vector,
        _initial_state_vector(scenario),
        args=(scenario, measurements, selected_propagator),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=80,
    )

    estimated_vector = cast(FloatArray, optimizer_result.x)
    residuals = residual_vector(estimated_vector, scenario, measurements, selected_propagator)
    rms = float(np.sqrt(np.mean(residuals**2)))
    estimated_scenario = scenario_with_state_vector(scenario, estimated_vector)
    diagnostics = _jacobian_diagnostics(cast(FloatArray, optimizer_result.jac))

    if diagnostics["jacobian_rank"] < STATE_DIMENSION:
        raise NumericalConvergenceError(
            "Orbit determination Jacobian is rank deficient: "
            f"rank {diagnostics['jacobian_rank']}/{STATE_DIMENSION}, "
            f"condition_number={diagnostics['condition_number']}, "
            f"singular_values={diagnostics['singular_values']}"
        )

    if not optimizer_result.success:
        raise NumericalConvergenceError(
            "Orbit determination optimizer failed to converge: "
            f"scipy_message={str(optimizer_result.message)!r}, "
            f"nfev={int(optimizer_result.nfev)}, "
            f"rms={rms}, "
            f"jacobian_rank={diagnostics['jacobian_rank']}, "
            f"condition_number={diagnostics['condition_number']}, "
            f"singular_values={diagnostics['singular_values']}"
        )

    return EstimateResult(
        estimated_state=estimated_scenario.initial_state,
        residuals=[float(value) for value in residuals],
        covariance=covariance_from_jacobian(cast(FloatArray, optimizer_result.jac), rms),
        rms=rms,
        iterations=int(optimizer_result.nfev),
        converged=bool(optimizer_result.success),
        metadata={
            "backend": f"{backend}_scipy_least_squares",
            "propagation_backend": backend,
            "message": str(optimizer_result.message),
            **diagnostics,
        },
    )
