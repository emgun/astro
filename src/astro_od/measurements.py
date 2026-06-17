from __future__ import annotations

from dataclasses import dataclass
from math import exp, radians, sin
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astro_core.models import (
    GroundStation,
    MeasurementRecord,
    MeasurementType,
    Scenario,
    Trajectory,
    TrajectorySample,
)

FloatArray = NDArray[np.float64]
MeasurementUnits = Literal["km", "km/s", "Hz", "deg"]
SPEED_OF_LIGHT_KM_S = 299792.458
_LIGHT_TIME_MAX_ITERATIONS = 8
_LIGHT_TIME_TOLERANCE_S = 1.0e-12
_ECI_NORTH_POLE = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_ECI_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_IONOSPHERE_RANGE_DELAY_CONSTANT_M3_S2 = 40.3
_TECU_ELECTRONS_PER_M2 = 1.0e16


@dataclass(frozen=True)
class RadiometricLightTimeSolution:
    light_time_model: str
    total_range_km: float
    total_light_time_s: float
    uplink_light_time_s: float
    downlink_light_time_s: float
    iterations: int
    transmit_time_offset_s: float
    reflection_time_offset_s: float
    receive_time_offset_s: float


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


def light_time_s(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    return float(range_km(spacecraft_position_km, station_position_km) / SPEED_OF_LIGHT_KM_S)


def two_way_range_km(spacecraft_position_km: ArrayLike, station_position_km: ArrayLike) -> float:
    return float(2.0 * range_km(spacecraft_position_km, station_position_km))


def two_way_light_time_s(
    spacecraft_position_km: ArrayLike,
    station_position_km: ArrayLike,
) -> float:
    return float(2.0 * light_time_s(spacecraft_position_km, station_position_km))


def two_way_range_rate_km_s(
    spacecraft_position_km: ArrayLike,
    spacecraft_velocity_km_s: ArrayLike,
    station_position_km: ArrayLike,
) -> float:
    return float(
        2.0
        * range_rate_km_s(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            station_position_km,
        )
    )


def three_way_range_km(
    spacecraft_position_km: ArrayLike,
    transmitter_position_km: ArrayLike,
    receiver_position_km: ArrayLike,
) -> float:
    return float(
        range_km(spacecraft_position_km, transmitter_position_km)
        + range_km(spacecraft_position_km, receiver_position_km)
    )


def three_way_light_time_s(
    spacecraft_position_km: ArrayLike,
    transmitter_position_km: ArrayLike,
    receiver_position_km: ArrayLike,
) -> float:
    return float(
        three_way_range_km(
            spacecraft_position_km,
            transmitter_position_km,
            receiver_position_km,
        )
        / SPEED_OF_LIGHT_KM_S
    )


def _linearized_spacecraft_position(
    receive_spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    receive_time_offset_s: float,
) -> FloatArray:
    return cast(
        FloatArray,
        receive_spacecraft_position_km + spacecraft_velocity_km_s * receive_time_offset_s,
    )


def iterative_two_way_light_time_solution(
    receive_spacecraft_position_km: ArrayLike,
    spacecraft_velocity_km_s: ArrayLike,
    station_position_km: ArrayLike,
    *,
    max_iterations: int = _LIGHT_TIME_MAX_ITERATIONS,
    tolerance_s: float = _LIGHT_TIME_TOLERANCE_S,
) -> RadiometricLightTimeSolution:
    return iterative_three_way_light_time_solution(
        receive_spacecraft_position_km,
        spacecraft_velocity_km_s,
        station_position_km,
        station_position_km,
        max_iterations=max_iterations,
        tolerance_s=tolerance_s,
    )


def iterative_three_way_light_time_solution(
    receive_spacecraft_position_km: ArrayLike,
    spacecraft_velocity_km_s: ArrayLike,
    transmitter_position_km: ArrayLike,
    receiver_position_km: ArrayLike,
    *,
    max_iterations: int = _LIGHT_TIME_MAX_ITERATIONS,
    tolerance_s: float = _LIGHT_TIME_TOLERANCE_S,
) -> RadiometricLightTimeSolution:
    receive_spacecraft_position = _as_float_array(receive_spacecraft_position_km)
    spacecraft_velocity = _as_float_array(spacecraft_velocity_km_s)
    transmitter_position = _as_float_array(transmitter_position_km)
    receiver_position = _as_float_array(receiver_position_km)

    downlink_light_time = light_time_s(receive_spacecraft_position, receiver_position)
    uplink_light_time = light_time_s(receive_spacecraft_position, transmitter_position)
    iteration_count = 0

    for iteration in range(1, max_iterations + 1):
        iteration_count = iteration
        reflection_time_offset = -downlink_light_time
        reflection_position = _linearized_spacecraft_position(
            receive_spacecraft_position,
            spacecraft_velocity,
            reflection_time_offset,
        )
        next_downlink_light_time = light_time_s(reflection_position, receiver_position)
        next_uplink_light_time = light_time_s(reflection_position, transmitter_position)
        converged = max(
            abs(next_downlink_light_time - downlink_light_time),
            abs(next_uplink_light_time - uplink_light_time),
        ) <= tolerance_s
        downlink_light_time = next_downlink_light_time
        uplink_light_time = next_uplink_light_time
        if converged:
            break

    total_light_time = uplink_light_time + downlink_light_time
    return RadiometricLightTimeSolution(
        light_time_model="vacuum_geometric_iterative_linearized",
        total_range_km=float(total_light_time * SPEED_OF_LIGHT_KM_S),
        total_light_time_s=float(total_light_time),
        uplink_light_time_s=float(uplink_light_time),
        downlink_light_time_s=float(downlink_light_time),
        iterations=iteration_count,
        transmit_time_offset_s=float(-total_light_time),
        reflection_time_offset_s=float(-downlink_light_time),
        receive_time_offset_s=0.0,
    )


def three_way_range_rate_km_s(
    spacecraft_position_km: ArrayLike,
    spacecraft_velocity_km_s: ArrayLike,
    transmitter_position_km: ArrayLike,
    receiver_position_km: ArrayLike,
) -> float:
    return float(
        range_rate_km_s(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            transmitter_position_km,
        )
        + range_rate_km_s(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            receiver_position_km,
        )
    )


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

        station_positions = [
            (
                station,
                station.position_array(sample.epoch, scenario.earth_orientation),
            )
            for station in scenario.ground_stations
        ]
        for measurement_type in scenario.measurements.types:
            if measurement_type in {
                MeasurementType.THREE_WAY_RANGE,
                MeasurementType.THREE_WAY_RANGE_RATE,
            }:
                records.extend(
                    _three_way_measurement_records(
                        measurement_type,
                        sample,
                        station_positions,
                        scenario,
                        rng,
                    )
                )
                continue

            for station, station_position in station_positions:
                truth, sigma, units, metadata = _measurement_geometry(
                    measurement_type,
                    spacecraft_position,
                    spacecraft_velocity,
                    station_position,
                    scenario,
                    observer_name=station.name,
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


def _three_way_measurement_records(
    measurement_type: MeasurementType,
    sample: TrajectorySample,
    station_positions: list[tuple[GroundStation, FloatArray]],
    scenario: Scenario,
    rng: np.random.Generator,
) -> list[MeasurementRecord]:
    if len(station_positions) < 2:
        return []

    spacecraft_position = sample.state.position_array()
    spacecraft_velocity = sample.state.velocity_array()
    transmitter, transmitter_position = station_positions[0]
    records: list[MeasurementRecord] = []

    for receiver, receiver_position in station_positions[1:]:
        truth, sigma, units, metadata = _three_way_measurement_geometry(
            measurement_type,
            spacecraft_position,
            spacecraft_velocity,
            transmitter_position,
            receiver_position,
            scenario,
        )
        path = f"{transmitter.name},{scenario.spacecraft.name},{receiver.name}"
        records.append(
            MeasurementRecord(
                measurement_type=measurement_type,
                epoch=sample.epoch,
                observer=receiver.name,
                observed_object=scenario.spacecraft.name,
                value=float(truth + rng.normal(0.0, sigma)),
                sigma=sigma,
                units=units,
                metadata={
                    "truth": truth,
                    "transmitter": transmitter.name,
                    "participant_path": path,
                }
                | metadata,
            )
        )
    return records


def _measurement_geometry(
    measurement_type: MeasurementType,
    spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    station_position_km: FloatArray,
    scenario: Scenario,
    *,
    observer_name: str | None = None,
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
    if measurement_type is MeasurementType.TWO_WAY_RANGE:
        observer = observer_name or scenario.ground_stations[0].name
        light_time_solution = iterative_two_way_light_time_solution(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            station_position_km,
        )
        media_metadata = radiometric_media_metadata(
            scenario,
            spacecraft_position_km,
            station_position_km,
            station_position_km,
        )
        return (
            light_time_solution.total_range_km + media_metadata["total_media_delay_km"],
            scenario.measurements.noise.range_sigma_km,
            "km",
            {
                "participant_path": f"{observer},{scenario.spacecraft.name},{observer}",
                "radiometric_model": "iterative_two_way_linearized",
                **_light_time_metadata(light_time_solution),
                "round_trip_light_time_s": light_time_solution.total_light_time_s,
                **media_metadata,
            },
        )
    if measurement_type is MeasurementType.TWO_WAY_RANGE_RATE:
        observer = observer_name or scenario.ground_stations[0].name
        light_time_solution = iterative_two_way_light_time_solution(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            station_position_km,
        )
        media_metadata = radiometric_media_metadata(
            scenario,
            spacecraft_position_km,
            station_position_km,
            station_position_km,
        )
        return (
            two_way_range_rate_km_s(
                spacecraft_position_km,
                spacecraft_velocity_km_s,
                station_position_km,
            ),
            scenario.measurements.noise.range_rate_sigma_km_s,
            "km/s",
            {
                "participant_path": f"{observer},{scenario.spacecraft.name},{observer}",
                "radiometric_model": "iterative_two_way_linearized",
                **_light_time_metadata(light_time_solution),
                "round_trip_light_time_s": light_time_solution.total_light_time_s,
                **media_metadata,
            },
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


def _three_way_measurement_geometry(
    measurement_type: MeasurementType,
    spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    transmitter_position_km: FloatArray,
    receiver_position_km: FloatArray,
    scenario: Scenario,
) -> tuple[float, float, MeasurementUnits, dict[str, Any]]:
    if measurement_type is MeasurementType.THREE_WAY_RANGE:
        light_time_solution = iterative_three_way_light_time_solution(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            transmitter_position_km,
            receiver_position_km,
        )
        media_metadata = radiometric_media_metadata(
            scenario,
            spacecraft_position_km,
            transmitter_position_km,
            receiver_position_km,
        )
        return (
            light_time_solution.total_range_km + media_metadata["total_media_delay_km"],
            scenario.measurements.noise.range_sigma_km,
            "km",
            {
                "radiometric_model": "iterative_three_way_linearized",
                **_light_time_metadata(light_time_solution),
                **media_metadata,
            },
        )
    if measurement_type is MeasurementType.THREE_WAY_RANGE_RATE:
        light_time_solution = iterative_three_way_light_time_solution(
            spacecraft_position_km,
            spacecraft_velocity_km_s,
            transmitter_position_km,
            receiver_position_km,
        )
        media_metadata = radiometric_media_metadata(
            scenario,
            spacecraft_position_km,
            transmitter_position_km,
            receiver_position_km,
        )
        return (
            three_way_range_rate_km_s(
                spacecraft_position_km,
                spacecraft_velocity_km_s,
                transmitter_position_km,
                receiver_position_km,
            ),
            scenario.measurements.noise.range_rate_sigma_km_s,
            "km/s",
            {
                "radiometric_model": "iterative_three_way_linearized",
                **_light_time_metadata(light_time_solution),
                **media_metadata,
            },
        )
    raise ValueError(f"Unsupported three-way measurement type: {measurement_type}")


def _light_time_metadata(solution: RadiometricLightTimeSolution) -> dict[str, Any]:
    return {
        "light_time_model": solution.light_time_model,
        "light_time_iterations": solution.iterations,
        "light_time_tolerance_s": _LIGHT_TIME_TOLERANCE_S,
        "uplink_light_time_s": solution.uplink_light_time_s,
        "downlink_light_time_s": solution.downlink_light_time_s,
        "total_light_time_s": solution.total_light_time_s,
        "transmit_time_offset_s": solution.transmit_time_offset_s,
        "reflection_time_offset_s": solution.reflection_time_offset_s,
        "receive_time_offset_s": solution.receive_time_offset_s,
    }


def _saturation_vapor_pressure_hpa(temperature_k: float) -> float:
    temperature_c = temperature_k - 273.15
    return 6.112 * exp((17.62 * temperature_c) / (243.12 + temperature_c))


def _zenith_troposphere_delay_km(scenario: Scenario) -> float:
    pressure_hpa = float(scenario.measurements.radiometric_weather_pressure_hpa)
    temperature_k = float(scenario.measurements.radiometric_weather_temperature_k)
    relative_humidity = float(scenario.measurements.radiometric_weather_relative_humidity)
    vapor_pressure_hpa = relative_humidity * _saturation_vapor_pressure_hpa(temperature_k)
    hydrostatic_delay_m = 0.0022768 * pressure_hpa
    wet_delay_m = 0.002277 * ((1255.0 / temperature_k) + 0.05) * vapor_pressure_hpa
    return float((hydrostatic_delay_m + wet_delay_m) / 1000.0)


def _zenith_ionosphere_delay_km(scenario: Scenario) -> float:
    tec_electrons_m2 = (
        float(scenario.measurements.radiometric_zenith_total_electron_content_tecu)
        * _TECU_ELECTRONS_PER_M2
    )
    frequency_hz = float(scenario.measurements.doppler_transmit_frequency_hz)
    delay_m = _IONOSPHERE_RANGE_DELAY_CONSTANT_M3_S2 * tec_electrons_m2 / frequency_hz**2
    return float(delay_m / 1000.0)


def _mapped_media_leg_delay(
    scenario: Scenario,
    spacecraft_position_km: ArrayLike,
    station_position_km: ArrayLike,
) -> tuple[float, float, float, float]:
    raw_elevation_deg = elevation_deg(spacecraft_position_km, station_position_km)
    mapped_elevation_deg = max(
        raw_elevation_deg,
        float(scenario.measurements.radiometric_media_min_elevation_deg),
    )
    mapping = 1.0 / sin(radians(mapped_elevation_deg))
    troposphere_delay_km = _zenith_troposphere_delay_km(scenario) * mapping
    ionosphere_delay_km = _zenith_ionosphere_delay_km(scenario) * mapping
    return (
        float(troposphere_delay_km + ionosphere_delay_km),
        float(troposphere_delay_km),
        float(ionosphere_delay_km),
        float(mapped_elevation_deg),
    )


def radiometric_media_metadata(
    scenario: Scenario,
    spacecraft_position_km: ArrayLike,
    uplink_station_position_km: ArrayLike,
    downlink_station_position_km: ArrayLike,
) -> dict[str, Any]:
    configured_uplink_delay_km = float(scenario.measurements.radiometric_media_uplink_delay_km)
    configured_downlink_delay_km = float(
        scenario.measurements.radiometric_media_downlink_delay_km
    )
    uplink_delay_km = configured_uplink_delay_km
    downlink_delay_km = configured_downlink_delay_km
    weather_metadata: dict[str, Any] = {}

    if scenario.measurements.radiometric_media_model == "weather_frequency":
        (
            uplink_weather_delay_km,
            uplink_troposphere_delay_km,
            uplink_ionosphere_delay_km,
            uplink_elevation_deg,
        ) = _mapped_media_leg_delay(scenario, spacecraft_position_km, uplink_station_position_km)
        (
            downlink_weather_delay_km,
            downlink_troposphere_delay_km,
            downlink_ionosphere_delay_km,
            downlink_elevation_deg,
        ) = _mapped_media_leg_delay(scenario, spacecraft_position_km, downlink_station_position_km)
        uplink_delay_km += uplink_weather_delay_km
        downlink_delay_km += downlink_weather_delay_km
        weather_metadata = {
            "configured_uplink_media_delay_km": configured_uplink_delay_km,
            "configured_downlink_media_delay_km": configured_downlink_delay_km,
            "uplink_troposphere_delay_km": uplink_troposphere_delay_km,
            "downlink_troposphere_delay_km": downlink_troposphere_delay_km,
            "uplink_ionosphere_delay_km": uplink_ionosphere_delay_km,
            "downlink_ionosphere_delay_km": downlink_ionosphere_delay_km,
            "media_frequency_hz": float(scenario.measurements.doppler_transmit_frequency_hz),
            "zenith_total_electron_content_tecu": float(
                scenario.measurements.radiometric_zenith_total_electron_content_tecu
            ),
            "weather_pressure_hpa": float(scenario.measurements.radiometric_weather_pressure_hpa),
            "weather_temperature_k": float(
                scenario.measurements.radiometric_weather_temperature_k
            ),
            "weather_relative_humidity": float(
                scenario.measurements.radiometric_weather_relative_humidity
            ),
            "uplink_media_elevation_deg": uplink_elevation_deg,
            "downlink_media_elevation_deg": downlink_elevation_deg,
            "media_min_elevation_deg": float(
                scenario.measurements.radiometric_media_min_elevation_deg
            ),
            "troposphere_model": "saastamoinen_surface_weather_simple_mapping",
            "ionosphere_model": "first_order_group_delay_tec_frequency",
        }

    total_delay_km = uplink_delay_km + downlink_delay_km
    if scenario.measurements.radiometric_media_model == "weather_frequency":
        media_model = "weather_frequency_range_delay"
    else:
        media_model = "configured_constant_range_delay" if total_delay_km > 0.0 else "none"
    return {
        "media_corrections_model": media_model,
        "media_corrections_source": scenario.measurements.radiometric_media_source,
        "uplink_media_delay_km": uplink_delay_km,
        "downlink_media_delay_km": downlink_delay_km,
        "total_media_delay_km": total_delay_km,
    } | weather_metadata
