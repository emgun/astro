from __future__ import annotations

from collections.abc import Callable
from math import radians
from typing import Any

from astro_backends.orekit.conversion import (
    absolute_date_from_datetime,
    km_s_to_m_s,
    km_to_m,
    m_s_to_km_s,
    m_to_km,
    validate_orekit_state_support,
)
from astro_backends.orekit.force_models import (
    build_atmospheric_drag_force_model,
    build_earth_shape,
    build_solar_radiation_pressure_force_model,
    build_third_body_gravity_force_models,
)
from astro_backends.orekit.propagation import (
    J2_POSITION_TOLERANCE_M,
    MU_EARTH_M3_S2,
    NUMERICAL_MIN_STEP_S,
    R_EARTH_M,
    _initial_orbit,
)
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime
from astro_core.constants import J2_EARTH
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    EstimateResult,
    ForceModelName,
    GroundStation,
    MeasurementRecord,
    MeasurementType,
    OrbitState,
    Scenario,
)

SUPPORTED_NATIVE_OREKIT_MEASUREMENTS = frozenset(
    {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    }
)
COVARIANCE_SINGULARITY_THRESHOLD = 1.0e-12
NATIVE_OD_MAX_ITERATIONS = 100
NATIVE_OD_MAX_EVALUATIONS = 100


def _station_by_name(scenario: Scenario) -> dict[str, GroundStation]:
    return {station.name: station for station in scenario.ground_stations}


def _validate_measurement_record(
    scenario: Scenario,
    record: MeasurementRecord,
    stations: dict[str, GroundStation],
) -> GroundStation:
    if record.measurement_type not in SUPPORTED_NATIVE_OREKIT_MEASUREMENTS:
        raise UnsupportedBackendError(
            "Native Orekit OD currently supports only range and range_rate measurements; "
            f"unsupported measurement type: {record.measurement_type}"
        )
    if record.observed_object != scenario.spacecraft.name:
        raise UnsupportedBackendError(
            "Native Orekit OD measurement observed object "
            f"{record.observed_object!r} does not match scenario spacecraft "
            f"{scenario.spacecraft.name!r}"
        )
    try:
        return stations[record.observer]
    except KeyError as exc:
        raise UnsupportedBackendError(
            f"Native Orekit OD measurement observer {record.observer!r} is not in the scenario"
        ) from exc


def _orekit_ground_station(
    station: GroundStation,
    runtime: OrekitRuntime,
    earth_shape: Any,
) -> Any:
    if (
        station.latitude_deg is None
        or station.longitude_deg is None
        or station.altitude_km is None
    ):
        raise UnsupportedBackendError(
            "Native Orekit OD ground-station mapping requires geodetic "
            "latitude_deg, longitude_deg, and altitude_km definitions"
        )

    geodetic_point = runtime.geodetic_point(
        radians(station.latitude_deg),
        radians(station.longitude_deg),
        km_to_m(station.altitude_km),
    )
    topocentric_frame = runtime.topocentric_frame(earth_shape, geodetic_point, station.name)
    return runtime.orekit_ground_station(topocentric_frame)


def build_orekit_observed_measurements(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    runtime: OrekitRuntime,
) -> list[Any]:
    """Map suite measurement records to Orekit observed measurement objects."""
    stations = _station_by_name(scenario)
    j2_frame = runtime.frames_factory.getITRF(runtime.iers_conventions.IERS_2010, True)
    earth_shape = build_earth_shape(runtime, j2_frame)
    satellite = runtime.observable_satellite(0)
    orekit_stations: dict[str, Any] = {}
    observed_measurements: list[Any] = []

    for record in measurements:
        station = _validate_measurement_record(scenario, record, stations)
        orekit_station = orekit_stations.setdefault(
            station.name,
            _orekit_ground_station(station, runtime, earth_shape),
        )
        measurement_date = absolute_date_from_datetime(runtime, record.epoch)
        if record.measurement_type is MeasurementType.RANGE:
            observed_measurements.append(
                runtime.range_measurement(
                    orekit_station,
                    False,
                    measurement_date,
                    km_to_m(record.value),
                    km_to_m(record.sigma),
                    1.0,
                    satellite,
                )
            )
        else:
            observed_measurements.append(
                runtime.range_rate_measurement(
                    orekit_station,
                    measurement_date,
                    km_s_to_m_s(record.value),
                    km_s_to_m_s(record.sigma),
                    1.0,
                    False,
                    satellite,
                )
            )

    return observed_measurements


def _orekit_propagator_builder(scenario: Scenario, runtime: OrekitRuntime) -> Any:
    validate_orekit_state_support(scenario.initial_state)
    frame = runtime.frames_factory.getEME2000()
    orbit = _initial_orbit(scenario, runtime, frame)
    max_step_s = float(max(scenario.propagation.step_s, NUMERICAL_MIN_STEP_S))
    integrator_builder = runtime.dormand_prince_853_integrator_builder(
        NUMERICAL_MIN_STEP_S,
        max_step_s,
        J2_POSITION_TOLERANCE_M,
    )
    builder = runtime.numerical_propagator_builder(
        orbit,
        integrator_builder,
        runtime.position_angle_type.TRUE,
        J2_POSITION_TOLERANCE_M,
    )
    if scenario.force_model.gravity is ForceModelName.TWO_BODY:
        return builder

    j2_frame = runtime.frames_factory.getITRF(runtime.iers_conventions.IERS_2010, True)
    earth_shape = build_earth_shape(runtime, j2_frame)
    builder.addForceModel(
        runtime.j2_only_perturbation(
            MU_EARTH_M3_S2,
            R_EARTH_M,
            J2_EARTH,
            j2_frame,
        )
    )
    if scenario.force_model.atmospheric_drag:
        builder.addForceModel(
            build_atmospheric_drag_force_model(scenario, runtime, earth_shape).model
        )
    if scenario.force_model.solar_radiation_pressure:
        builder.addForceModel(
            build_solar_radiation_pressure_force_model(scenario, runtime, earth_shape).model
        )
    if scenario.force_model.third_body_gravity:
        for third_body_force_model in build_third_body_gravity_force_models(runtime):
            builder.addForceModel(third_body_force_model.model)
    return builder


def build_orekit_batch_ls_estimator(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    runtime: OrekitRuntime,
) -> Any:
    """Build an Orekit BatchLSEstimator with suite measurements mapped to Orekit objects."""
    observed_measurements = build_orekit_observed_measurements(scenario, measurements, runtime)
    propagator_builder = _orekit_propagator_builder(scenario, runtime)
    estimator = runtime.batch_ls_estimator(
        runtime.levenberg_marquardt_optimizer(),
        propagator_builder,
    )
    estimator.setMaxIterations(NATIVE_OD_MAX_ITERATIONS)
    estimator.setMaxEvaluations(NATIVE_OD_MAX_EVALUATIONS)
    for observed_measurement in observed_measurements:
        estimator.addMeasurement(observed_measurement)
    return estimator


def _matrix_to_nested_list(matrix: Any, size: int = 6) -> list[list[float]]:
    return [
        [float(matrix.getEntry(row, column)) for column in range(size)]
        for row in range(size)
    ]


def _zero_covariance(size: int = 6) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _physical_covariance_with_status(estimator: Any) -> tuple[list[list[float]], dict[str, Any]]:
    try:
        covariance = _matrix_to_nested_list(
            estimator.getPhysicalCovariances(COVARIANCE_SINGULARITY_THRESHOLD)
        )
    except Exception as exc:
        return _zero_covariance(), {
            "covariance_status": "unavailable",
            "covariance_error": str(exc),
            "covariance_fallback": "zero_6x6",
        }
    return covariance, {"covariance_status": "available"}


def _vector_to_list(vector: Any) -> list[float]:
    if hasattr(vector, "toArray"):
        return [float(value) for value in vector.toArray()]
    return [float(value) for value in vector]


def _estimated_state_from_propagator(
    scenario: Scenario,
    runtime: OrekitRuntime,
    propagator: Any,
) -> OrbitState:
    frame = runtime.frames_factory.getEME2000()
    initial_date = absolute_date_from_datetime(runtime, scenario.initial_state.epoch)
    spacecraft_state = propagator.propagate(initial_date)
    pv_coordinates = spacecraft_state.getPVCoordinates(frame)
    position = pv_coordinates.getPosition()
    velocity = pv_coordinates.getVelocity()
    return scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=(
                    m_to_km(float(position.getX())),
                    m_to_km(float(position.getY())),
                    m_to_km(float(position.getZ())),
                ),
                velocity_km_s=(
                    m_s_to_km_s(float(velocity.getX())),
                    m_s_to_km_s(float(velocity.getY())),
                    m_s_to_km_s(float(velocity.getZ())),
                ),
            )
        }
    )


def estimate_orekit_native(
    scenario: Scenario,
    measurements: list[MeasurementRecord],
    *,
    runtime_loader: Callable[[], OrekitRuntime] | None = None,
) -> EstimateResult:
    """Run Orekit BatchLSEstimator and map the result into the suite OD product."""
    runtime = runtime_loader() if runtime_loader is not None else load_orekit_runtime()
    estimator = build_orekit_batch_ls_estimator(scenario, measurements, runtime)
    propagators = estimator.estimate()
    if len(propagators) == 0:
        raise UnsupportedBackendError("Native Orekit OD returned no estimated propagators")

    optimum = estimator.getOptimum()
    residuals = _vector_to_list(optimum.getResiduals())
    covariance, covariance_metadata = _physical_covariance_with_status(estimator)

    return EstimateResult(
        estimated_state=_estimated_state_from_propagator(scenario, runtime, propagators[0]),
        residuals=residuals,
        covariance=covariance,
        rms=float(optimum.getRMS()),
        iterations=int(estimator.getIterationsCount()),
        converged=True,
        metadata={
            "backend": "orekit_batch_ls_estimator",
            "propagation_backend": "orekit",
            "estimator": "Orekit BatchLSEstimator",
            "evaluations": int(estimator.getEvaluationsCount()),
            "max_iterations": NATIVE_OD_MAX_ITERATIONS,
            "max_evaluations": NATIVE_OD_MAX_EVALUATIONS,
            "measurement_count": len(measurements),
            "covariance_singularity_threshold": COVARIANCE_SINGULARITY_THRESHOLD,
            "wrapper": runtime.wrapper,
            "wrapper_version": runtime.wrapper_version,
            "data_path": runtime.data_path,
            **covariance_metadata,
        },
    )
