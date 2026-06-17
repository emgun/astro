from __future__ import annotations

from datetime import datetime
from math import sqrt
from statistics import fmean
from typing import Any

from pydantic import Field, FiniteFloat

from astro_core.models import AstroModel, MeasurementRecord, Scenario, Trajectory
from astro_od.measurements import generate_synthetic_measurements


class DsnCalibrationSample(AstroModel):
    epoch: datetime
    measurement_type: str
    observer: str
    observed_object: str
    participant_path: str
    calibration_model: str
    media_source: str
    uplink_media_delay_km: FiniteFloat = Field(ge=0.0)
    downlink_media_delay_km: FiniteFloat = Field(ge=0.0)
    total_media_delay_km: FiniteFloat = Field(ge=0.0)
    uplink_media_elevation_deg: FiniteFloat | None = None
    downlink_media_elevation_deg: FiniteFloat | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DsnCalibrationProduct(AstroModel):
    scenario_id: str
    calibration_model: str
    media_source: str
    sample_count: int = Field(ge=1)
    station_count: int = Field(ge=1)
    measurement_types: tuple[str, ...]
    total_media_delay_km_min: FiniteFloat = Field(ge=0.0)
    total_media_delay_km_mean: FiniteFloat = Field(ge=0.0)
    total_media_delay_km_max: FiniteFloat = Field(ge=0.0)
    uplink_media_delay_km_mean: FiniteFloat = Field(ge=0.0)
    downlink_media_delay_km_mean: FiniteFloat = Field(ge=0.0)
    samples: tuple[DsnCalibrationSample, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StationCalibrationEntry(AstroModel):
    station: str
    measurement_type: str
    units: str
    sample_count: int = Field(ge=1)
    bias_mean: FiniteFloat
    bias_min: FiniteFloat
    bias_max: FiniteFloat
    bias_rms: FiniteFloat = Field(ge=0.0)
    sigma_mean: FiniteFloat = Field(gt=0.0)
    normalized_bias_mean: FiniteFloat


class StationCalibrationProduct(AstroModel):
    scenario_id: str
    calibration_model: str = "station_measurement_bias_from_truth_metadata"
    station_count: int = Field(ge=1)
    entry_count: int = Field(ge=1)
    measurement_count: int = Field(ge=1)
    entries: tuple[StationCalibrationEntry, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def generate_dsn_calibration_product(
    scenario: Scenario,
    trajectory: Trajectory,
) -> DsnCalibrationProduct:
    """Summarize generated radiometric media corrections into an audit product."""
    measurement_records = generate_synthetic_measurements(scenario, trajectory)
    return generate_dsn_calibration_product_from_measurements(
        scenario.scenario_id,
        measurement_records,
        station_count=len(scenario.ground_stations),
        measurement_types=tuple(
            measurement_type.value for measurement_type in scenario.measurements.types
        ),
        metadata=_scenario_calibration_metadata(scenario),
    )


def generate_station_calibration_product_from_measurements(
    scenario_id: str,
    measurement_records: list[MeasurementRecord],
    *,
    metadata: dict[str, Any] | None = None,
) -> StationCalibrationProduct:
    """Estimate station/type measurement biases from records carrying truth metadata."""
    grouped_residuals: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    calibrated_measurement_count = 0
    for record in measurement_records:
        if "truth" not in record.metadata:
            continue
        truth = _metadata_float(record.metadata, "truth")
        residual = float(record.value) - truth
        key = (record.observer, record.measurement_type.value, record.units)
        grouped_residuals.setdefault(key, []).append((residual, float(record.sigma)))
        calibrated_measurement_count += 1

    if not grouped_residuals:
        raise ValueError("station calibration requires measurement records with truth metadata")

    entries = tuple(
        _station_calibration_entry(station, measurement_type, units, residual_sigma_pairs)
        for (station, measurement_type, units), residual_sigma_pairs in sorted(
            grouped_residuals.items()
        )
    )
    product_metadata: dict[str, Any] = {
        "workflow": "station_calibration",
        "calibration_reference": "measurement_metadata_truth",
        "source_measurement_count": len(measurement_records),
        "calibrated_measurement_count": calibrated_measurement_count,
    }
    if metadata:
        product_metadata |= metadata
    return StationCalibrationProduct(
        scenario_id=scenario_id,
        station_count=len({entry.station for entry in entries}),
        entry_count=len(entries),
        measurement_count=calibrated_measurement_count,
        entries=entries,
        metadata=product_metadata,
    )


def generate_dsn_calibration_product_from_measurements(
    scenario_id: str,
    measurement_records: list[MeasurementRecord],
    *,
    station_count: int | None = None,
    measurement_types: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DsnCalibrationProduct:
    """Summarize radiometric media corrections carried by measurement records."""
    samples = tuple(
        _dsn_calibration_sample(record)
        for record in measurement_records
        if _has_applied_radiometric_media(record)
    )
    if not samples:
        raise ValueError(
            "scenario did not produce radiometric media correction records for DSN calibration"
        )

    total_delays_km = [sample.total_media_delay_km for sample in samples]
    uplink_delays_km = [sample.uplink_media_delay_km for sample in samples]
    downlink_delays_km = [sample.downlink_media_delay_km for sample in samples]
    calibration_models = tuple(dict.fromkeys(sample.calibration_model for sample in samples))
    media_sources = tuple(dict.fromkeys(sample.media_source for sample in samples))

    return DsnCalibrationProduct(
        scenario_id=scenario_id,
        calibration_model=calibration_models[0] if len(calibration_models) == 1 else "mixed",
        media_source=media_sources[0] if len(media_sources) == 1 else "mixed",
        sample_count=len(samples),
        station_count=station_count
        if station_count is not None
        else _derived_station_count(samples),
        measurement_types=measurement_types
        if measurement_types is not None
        else tuple(dict.fromkeys(sample.measurement_type for sample in samples)),
        total_media_delay_km_min=min(total_delays_km),
        total_media_delay_km_mean=fmean(total_delays_km),
        total_media_delay_km_max=max(total_delays_km),
        uplink_media_delay_km_mean=fmean(uplink_delays_km),
        downlink_media_delay_km_mean=fmean(downlink_delays_km),
        samples=samples,
        metadata=_dsn_calibration_metadata(
            samples,
            source_measurement_count=len(measurement_records),
            metadata=metadata,
        ),
    )


def _station_calibration_entry(
    station: str,
    measurement_type: str,
    units: str,
    residual_sigma_pairs: list[tuple[float, float]],
) -> StationCalibrationEntry:
    residuals = [residual for residual, _sigma in residual_sigma_pairs]
    sigmas = [sigma for _residual, sigma in residual_sigma_pairs]
    bias_mean = fmean(residuals)
    sigma_mean = fmean(sigmas)
    return StationCalibrationEntry(
        station=station,
        measurement_type=measurement_type,
        units=units,
        sample_count=len(residuals),
        bias_mean=bias_mean,
        bias_min=min(residuals),
        bias_max=max(residuals),
        bias_rms=sqrt(fmean([residual * residual for residual in residuals])),
        sigma_mean=sigma_mean,
        normalized_bias_mean=bias_mean / sigma_mean,
    )


def _has_applied_radiometric_media(record: MeasurementRecord) -> bool:
    return (
        "total_media_delay_km" in record.metadata
        and _metadata_float(record.metadata, "total_media_delay_km") > 0.0
        and record.metadata.get("media_corrections_model") != "none"
    )


def _dsn_calibration_sample(record: MeasurementRecord) -> DsnCalibrationSample:
    return DsnCalibrationSample(
        epoch=record.epoch,
        measurement_type=record.measurement_type.value,
        observer=record.observer,
        observed_object=record.observed_object,
        participant_path=str(record.metadata.get("participant_path", record.observer)),
        calibration_model=str(record.metadata.get("media_corrections_model", "unknown")),
        media_source=str(record.metadata.get("media_corrections_source", "unknown")),
        uplink_media_delay_km=_metadata_float(record.metadata, "uplink_media_delay_km"),
        downlink_media_delay_km=_metadata_float(record.metadata, "downlink_media_delay_km"),
        total_media_delay_km=_metadata_float(record.metadata, "total_media_delay_km"),
        uplink_media_elevation_deg=_optional_metadata_float(
            record.metadata,
            "uplink_media_elevation_deg",
        ),
        downlink_media_elevation_deg=_optional_metadata_float(
            record.metadata,
            "downlink_media_elevation_deg",
        ),
        metadata=_sample_metadata(record.metadata),
    )


def _dsn_calibration_metadata(
    samples: tuple[DsnCalibrationSample, ...],
    *,
    source_measurement_count: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    product_metadata: dict[str, Any] = {
        "workflow": "dsn_calibration_summary",
        "source_format": "measurement_records",
        "source_measurement_count": source_measurement_count,
    }
    if metadata:
        product_metadata |= metadata
    first_sample_metadata = samples[0].metadata
    for key in (
        "media_frequency_hz",
        "weather_pressure_hpa",
        "weather_temperature_k",
        "weather_relative_humidity",
        "zenith_total_electron_content_tecu",
        "troposphere_model",
        "ionosphere_model",
    ):
        if key in first_sample_metadata:
            product_metadata[key] = first_sample_metadata[key]
    return product_metadata


def _scenario_calibration_metadata(scenario: Scenario) -> dict[str, Any]:
    measurement_config = scenario.measurements
    return {
        "source_format": "generated_measurements",
        "spacecraft": scenario.spacecraft.name,
        "ground_stations": [station.name for station in scenario.ground_stations],
        "configured_media_model": measurement_config.radiometric_media_model,
        "configured_uplink_media_delay_km": float(
            measurement_config.radiometric_media_uplink_delay_km
        ),
        "configured_downlink_media_delay_km": float(
            measurement_config.radiometric_media_downlink_delay_km
        ),
        "media_min_elevation_deg": float(measurement_config.radiometric_media_min_elevation_deg),
    }


def _sample_metadata(record_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record_metadata.items()
        if key
        not in {
            "truth",
            "participant_path",
            "media_corrections_model",
            "media_corrections_source",
            "uplink_media_delay_km",
            "downlink_media_delay_km",
            "total_media_delay_km",
            "uplink_media_elevation_deg",
            "downlink_media_elevation_deg",
        }
    }


def _metadata_float(metadata: dict[str, Any], key: str) -> float:
    value = metadata[key]
    if isinstance(value, bool | str):
        raise ValueError(f"radiometric media metadata {key} must be numeric")
    return float(value)


def _optional_metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    if key not in metadata:
        return None
    return _metadata_float(metadata, key)


def _derived_station_count(samples: tuple[DsnCalibrationSample, ...]) -> int:
    station_names: set[str] = set()
    for sample in samples:
        station_names.add(sample.observer)
        participants = [participant.strip() for participant in sample.participant_path.split(",")]
        for participant in participants:
            if participant and participant != sample.observed_object:
                station_names.add(participant)
    return max(1, len(station_names))
