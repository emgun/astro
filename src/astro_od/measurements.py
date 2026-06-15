from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astro_core.models import MeasurementRecord, MeasurementType, Scenario, Trajectory

FloatArray = NDArray[np.float64]
MeasurementUnits = Literal["km", "km/s", "deg"]


def _as_float_array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("Measurement geometry vectors must have exactly 3 components")
    return cast(FloatArray, array)


def range_km(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    spacecraft_position = _as_float_array(spacecraft_position_km)
    station_position = _as_float_array(station_position_km)
    return float(np.linalg.norm(spacecraft_position - station_position))


def range_rate_km_s(
    spacecraft_position_km: ArrayLike,
    spacecraft_velocity_km_s: ArrayLike,
    station_position_km: ArrayLike,
) -> float:
    spacecraft_position = _as_float_array(spacecraft_position_km)
    spacecraft_velocity = _as_float_array(spacecraft_velocity_km_s)
    station_position = _as_float_array(station_position_km)

    relative_position = spacecraft_position - station_position
    distance = float(np.linalg.norm(relative_position))
    if distance == 0.0:
        raise ValueError("Cannot compute range-rate for zero range")

    line_of_sight = relative_position / distance
    return float(np.dot(spacecraft_velocity, line_of_sight))


def _relative_line_of_sight(
    spacecraft_position_km: ArrayLike,
    station_position_km: ArrayLike,
) -> FloatArray:
    spacecraft_position = _as_float_array(spacecraft_position_km)
    station_position = _as_float_array(station_position_km)
    relative_position = spacecraft_position - station_position
    distance = float(np.linalg.norm(relative_position))
    if distance == 0.0:
        raise ValueError("Cannot compute line-of-sight angles for zero range")
    return cast(FloatArray, relative_position / distance)


def right_ascension_deg(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    line_of_sight = _relative_line_of_sight(spacecraft_position_km, station_position_km)
    return float(np.degrees(np.arctan2(line_of_sight[1], line_of_sight[0])) % 360.0)


def declination_deg(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    line_of_sight = _relative_line_of_sight(spacecraft_position_km, station_position_km)
    return float(np.degrees(np.arcsin(np.clip(line_of_sight[2], -1.0, 1.0))))


def _is_on_cadence(elapsed_s: float, cadence_s: float) -> bool:
    cadence_index = elapsed_s / cadence_s
    return abs(cadence_index - round(cadence_index)) <= 1.0e-9


def generate_synthetic_measurements(
    scenario: Scenario,
    trajectory: Trajectory,
) -> list[MeasurementRecord]:
    if not scenario.ground_stations:
        return []

    rng = np.random.default_rng(scenario.measurements.noise.seed)
    records: list[MeasurementRecord] = []
    cadence_origin = scenario.initial_state.epoch

    for sample in trajectory.samples:
        elapsed_s = (sample.epoch - cadence_origin).total_seconds()
        if not _is_on_cadence(elapsed_s, scenario.measurements.cadence_s):
            continue

        spacecraft_position = sample.state.position_array()
        spacecraft_velocity = sample.state.velocity_array()

        for station in scenario.ground_stations:
            station_position = station.position_array()
            for measurement_type in scenario.measurements.types:
                truth, sigma, units = _measurement_geometry(
                    measurement_type,
                    spacecraft_position,
                    spacecraft_velocity,
                    station_position,
                    scenario,
                )
                records.append(
                    MeasurementRecord(
                        measurement_type=measurement_type,
                        epoch=sample.epoch,
                        observer=station.name,
                        observed_object=scenario.spacecraft.name,
                        value=float(truth + rng.normal(0.0, sigma)),
                        sigma=sigma,
                        units=units,
                        metadata={"truth": truth},
                    )
                )

    return records


def _measurement_geometry(
    measurement_type: MeasurementType,
    spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    station_position_km: FloatArray,
    scenario: Scenario,
) -> tuple[float, float, MeasurementUnits]:
    if measurement_type is MeasurementType.RANGE:
        return (
            range_km(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.range_sigma_km,
            "km",
        )
    if measurement_type is MeasurementType.RANGE_RATE:
        return (
            range_rate_km_s(
                spacecraft_position_km,
                spacecraft_velocity_km_s,
                station_position_km,
            ),
            scenario.measurements.noise.range_rate_sigma_km_s,
            "km/s",
        )
    if measurement_type is MeasurementType.RIGHT_ASCENSION:
        return (
            right_ascension_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
        )
    if measurement_type is MeasurementType.DECLINATION:
        return (
            declination_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
        )
    raise ValueError(f"Unsupported measurement type: {measurement_type}")
