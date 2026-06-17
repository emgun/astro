from __future__ import annotations

from datetime import datetime
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


def generate_dsn_calibration_product(
    scenario: Scenario,
    trajectory: Trajectory,
) -> DsnCalibrationProduct:
    """Summarize generated radiometric media corrections into an audit product."""
    measurement_records = generate_synthetic_measurements(scenario, trajectory)
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
        scenario_id=scenario.scenario_id,
        calibration_model=calibration_models[0] if len(calibration_models) == 1 else "mixed",
        media_source=media_sources[0] if len(media_sources) == 1 else "mixed",
        sample_count=len(samples),
        station_count=len(scenario.ground_stations),
        measurement_types=tuple(
            measurement_type.value for measurement_type in scenario.measurements.types
        ),
        total_media_delay_km_min=min(total_delays_km),
        total_media_delay_km_mean=fmean(total_delays_km),
        total_media_delay_km_max=max(total_delays_km),
        uplink_media_delay_km_mean=fmean(uplink_delays_km),
        downlink_media_delay_km_mean=fmean(downlink_delays_km),
        samples=samples,
        metadata=_dsn_calibration_metadata(scenario, samples),
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
    scenario: Scenario,
    samples: tuple[DsnCalibrationSample, ...],
) -> dict[str, Any]:
    measurement_config = scenario.measurements
    metadata: dict[str, Any] = {
        "workflow": "dsn_calibration_summary",
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
            metadata[key] = first_sample_metadata[key]
    return metadata


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
