from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astro_core.models import MeasurementRecord, MeasurementType, Scenario, Trajectory

FloatArray = NDArray[np.float64]
MeasurementUnits = Literal["km", "km/s", "Hz", "deg"]
SPEED_OF_LIGHT_KM_S = 299792.458
_ECI_NORTH_POLE = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_ECI_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


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


def doppler_hz(range_rate_truth_km_s: float, transmit_frequency_hz: float) -> float:
    return float(-range_rate_truth_km_s / SPEED_OF_LIGHT_KM_S * transmit_frequency_hz)


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


def _topocentric_basis(station_position_km: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    station_position = _as_float_array(station_position_km)
    station_radius = float(np.linalg.norm(station_position))
    if station_radius == 0.0:
        raise ValueError("Cannot compute topocentric angles for a station at Earth center")

    up = cast(FloatArray, station_position / station_radius)
    east = np.cross(_ECI_NORTH_POLE, up)
    east_norm = float(np.linalg.norm(east))
    if east_norm == 0.0:
        east = np.cross(_ECI_Y_AXIS, up)
        east_norm = float(np.linalg.norm(east))
    east = cast(FloatArray, east / east_norm)
    north = cast(FloatArray, np.cross(up, east))
    return east, north, up


def azimuth_deg(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    line_of_sight = _relative_line_of_sight(spacecraft_position_km, station_position_km)
    east, north, _up = _topocentric_basis(station_position_km)
    east_component = float(np.dot(line_of_sight, east))
    north_component = float(np.dot(line_of_sight, north))
    return float(np.degrees(np.arctan2(east_component, north_component)) % 360.0)


def elevation_deg(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    line_of_sight = _relative_line_of_sight(spacecraft_position_km, station_position_km)
    _east, _north, up = _topocentric_basis(station_position_km)
    return float(np.degrees(np.arcsin(np.clip(float(np.dot(line_of_sight, up)), -1.0, 1.0))))


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
            station_position = station.position_array(sample.epoch, scenario.earth_orientation)
            for measurement_type in scenario.measurements.types:
                truth, sigma, units, metadata = _measurement_geometry(
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
                        metadata={"truth": truth} | metadata,
                    )
                )

    return records


def _measurement_geometry(
    measurement_type: MeasurementType,
    spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    station_position_km: FloatArray,
    scenario: Scenario,
) -> tuple[float, float, MeasurementUnits, dict[str, Any]]:
    if measurement_type is MeasurementType.RANGE:
        return (
            range_km(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.range_sigma_km,
            "km",
            {},
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
            {},
        )
    if measurement_type is MeasurementType.DOPPLER:
        range_rate_truth = range_rate_km_s(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            station_position_km,
        )
        transmit_frequency_hz = scenario.measurements.doppler_transmit_frequency_hz
        return (
            doppler_hz(range_rate_truth, transmit_frequency_hz),
            scenario.measurements.noise.doppler_sigma_hz,
            "Hz",
            {
                "range_rate_truth_km_s": range_rate_truth,
                "doppler_transmit_frequency_hz": transmit_frequency_hz,
                "doppler_model": "one_way_range_rate",
            },
        )
    if measurement_type is MeasurementType.RIGHT_ASCENSION:
        return (
            right_ascension_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
            {},
        )
    if measurement_type is MeasurementType.DECLINATION:
        return (
            declination_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
            {},
        )
    if measurement_type is MeasurementType.AZIMUTH:
        return (
            azimuth_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
            {},
        )
    if measurement_type is MeasurementType.ELEVATION:
        return (
            elevation_deg(spacecraft_position_km, station_position_km),
            scenario.measurements.noise.angle_sigma_deg,
            "deg",
            {},
        )
    raise ValueError(f"Unsupported measurement type: {measurement_type}")
