from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from astro_core.errors import InvalidMeasurementFileError
from astro_core.models import MeasurementRecord, MeasurementType

DEFAULT_TDM_RANGE_SIGMA_KM = 0.01
DEFAULT_TDM_RANGE_RATE_SIGMA_KM_S = 1.0e-5
DEFAULT_TDM_DOPPLER_SIGMA_HZ = 0.1
DEFAULT_TDM_ANGLE_SIGMA_DEG = 0.001
TDM_EXPORT_MEASUREMENT_TYPES = frozenset(
    {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
        MeasurementType.DOPPLER,
        MeasurementType.TWO_WAY_RANGE,
        MeasurementType.TWO_WAY_RANGE_RATE,
        MeasurementType.THREE_WAY_RANGE,
        MeasurementType.THREE_WAY_RANGE_RATE,
        MeasurementType.RIGHT_ASCENSION,
        MeasurementType.DECLINATION,
        MeasurementType.AZIMUTH,
        MeasurementType.ELEVATION,
    }
)

RANGE_MEASUREMENT_TYPES = frozenset(
    {
        MeasurementType.RANGE,
        MeasurementType.TWO_WAY_RANGE,
        MeasurementType.THREE_WAY_RANGE,
    }
)

RANGE_RATE_MEASUREMENT_TYPES = frozenset(
    {
        MeasurementType.RANGE_RATE,
        MeasurementType.TWO_WAY_RANGE_RATE,
        MeasurementType.THREE_WAY_RANGE_RATE,
    }
)

ANGLE_MEASUREMENT_TYPES = frozenset(
    {
        MeasurementType.RIGHT_ASCENSION,
        MeasurementType.DECLINATION,
        MeasurementType.AZIMUTH,
        MeasurementType.ELEVATION,
    }
)

TDM_ANGLE_TYPE_BY_MEASUREMENT_TYPE = {
    MeasurementType.RIGHT_ASCENSION: "RADEC",
    MeasurementType.DECLINATION: "RADEC",
    MeasurementType.AZIMUTH: "AZEL",
    MeasurementType.ELEVATION: "AZEL",
}

TDM_DATA_KEYWORD_BY_MEASUREMENT_TYPE = {
    MeasurementType.RANGE: "RANGE",
    MeasurementType.RANGE_RATE: "DOPPLER_INSTANTANEOUS",
    MeasurementType.DOPPLER: "DOPPLER_INSTANTANEOUS",
    MeasurementType.TWO_WAY_RANGE: "RANGE",
    MeasurementType.TWO_WAY_RANGE_RATE: "DOPPLER_INSTANTANEOUS",
    MeasurementType.THREE_WAY_RANGE: "RANGE",
    MeasurementType.THREE_WAY_RANGE_RATE: "DOPPLER_INSTANTANEOUS",
    MeasurementType.RIGHT_ASCENSION: "ANGLE_1",
    MeasurementType.DECLINATION: "ANGLE_2",
    MeasurementType.AZIMUTH: "ANGLE_1",
    MeasurementType.ELEVATION: "ANGLE_2",
}

TDM_ANGLE_MEASUREMENT_TYPE_BY_ANGLE_TYPE_AND_KEYWORD = {
    ("RADEC", "ANGLE_1"): MeasurementType.RIGHT_ASCENSION,
    ("RADEC", "ANGLE_2"): MeasurementType.DECLINATION,
    ("AZEL", "ANGLE_1"): MeasurementType.AZIMUTH,
    ("AZEL", "ANGLE_2"): MeasurementType.ELEVATION,
}

TDM_RADIOMETRIC_MEASUREMENT_TYPE_BY_LINK_AND_KEYWORD = {
    ("doppler_hz", "DOPPLER_INSTANTANEOUS"): MeasurementType.DOPPLER,
    ("doppler_hz", "DOPPLER_INTEGRATED"): MeasurementType.DOPPLER,
    ("two_way", "RANGE"): MeasurementType.TWO_WAY_RANGE,
    ("two_way", "DOPPLER_INSTANTANEOUS"): MeasurementType.TWO_WAY_RANGE_RATE,
    ("two_way", "DOPPLER_INTEGRATED"): MeasurementType.TWO_WAY_RANGE_RATE,
    ("three_way", "RANGE"): MeasurementType.THREE_WAY_RANGE,
    ("three_way", "DOPPLER_INSTANTANEOUS"): MeasurementType.THREE_WAY_RANGE_RATE,
    ("three_way", "DOPPLER_INTEGRATED"): MeasurementType.THREE_WAY_RANGE_RATE,
}

TDM_SUITE_METADATA_KEYS = (
    ("participant_path", "ASTRO_PARTICIPANT_PATH"),
    ("media_corrections_model", "ASTRO_MEDIA_CORRECTIONS_MODEL"),
    ("media_corrections_source", "ASTRO_MEDIA_CORRECTIONS_SOURCE"),
    ("uplink_media_delay_km", "ASTRO_UPLINK_MEDIA_DELAY_KM"),
    ("downlink_media_delay_km", "ASTRO_DOWNLINK_MEDIA_DELAY_KM"),
    ("total_media_delay_km", "ASTRO_TOTAL_MEDIA_DELAY_KM"),
    ("uplink_media_elevation_deg", "ASTRO_UPLINK_MEDIA_ELEVATION_DEG"),
    ("downlink_media_elevation_deg", "ASTRO_DOWNLINK_MEDIA_ELEVATION_DEG"),
    ("uplink_troposphere_delay_km", "ASTRO_UPLINK_TROPOSPHERE_DELAY_KM"),
    ("downlink_troposphere_delay_km", "ASTRO_DOWNLINK_TROPOSPHERE_DELAY_KM"),
    ("uplink_ionosphere_delay_km", "ASTRO_UPLINK_IONOSPHERE_DELAY_KM"),
    ("downlink_ionosphere_delay_km", "ASTRO_DOWNLINK_IONOSPHERE_DELAY_KM"),
    ("configured_uplink_media_delay_km", "ASTRO_CONFIGURED_UPLINK_MEDIA_DELAY_KM"),
    ("configured_downlink_media_delay_km", "ASTRO_CONFIGURED_DOWNLINK_MEDIA_DELAY_KM"),
    ("media_frequency_hz", "ASTRO_MEDIA_FREQUENCY_HZ"),
    ("zenith_total_electron_content_tecu", "ASTRO_ZENITH_TOTAL_ELECTRON_CONTENT_TECU"),
    ("weather_pressure_hpa", "ASTRO_WEATHER_PRESSURE_HPA"),
    ("weather_temperature_k", "ASTRO_WEATHER_TEMPERATURE_K"),
    ("weather_relative_humidity", "ASTRO_WEATHER_RELATIVE_HUMIDITY"),
    ("media_min_elevation_deg", "ASTRO_MEDIA_MIN_ELEVATION_DEG"),
    ("troposphere_model", "ASTRO_TROPOSPHERE_MODEL"),
    ("ionosphere_model", "ASTRO_IONOSPHERE_MODEL"),
)
TDM_RECORD_METADATA_BY_SUITE_KEY = {
    tdm_key: record_key for record_key, tdm_key in TDM_SUITE_METADATA_KEYS
}

CSV_REQUIRED_COLUMNS = frozenset(
    {
        "scenario_id",
        "measurement_type",
        "epoch",
        "observer",
        "observed_object",
        "value",
        "sigma",
        "units",
    }
)

SUPPORTED_TDM_DATA_KEYWORDS = frozenset(
    {
        "RANGE",
        "DOPPLER_INSTANTANEOUS",
        "DOPPLER_INTEGRATED",
        "ANGLE_1",
        "ANGLE_2",
    }
)


@dataclass(frozen=True)
class MeasurementProduct:
    scenario_id: str
    measurements: list[MeasurementRecord]


TdmMetadataSignature = tuple[tuple[str, str], ...]
TdmSegmentKey = tuple[str, str, str | None, str | None, str | None, TdmMetadataSignature]


def load_measurements(
    path: Path | str,
    *,
    expected_scenario_id: str | None = None,
    measurement_format: str = "auto",
) -> list[MeasurementRecord]:
    measurement_path = Path(path)
    resolved_format = resolve_measurement_format(measurement_path, measurement_format)
    if resolved_format == "tdm":
        return _load_tdm_measurements(
            measurement_path,
            expected_scenario_id=expected_scenario_id,
        )
    if resolved_format == "csv":
        return _load_csv_measurements(
            measurement_path,
            expected_scenario_id=expected_scenario_id,
        )
    return _load_json_measurements(
        measurement_path,
        expected_scenario_id=expected_scenario_id,
    )


def load_measurement_product(
    path: Path | str,
    *,
    expected_scenario_id: str | None = None,
) -> MeasurementProduct:
    return _load_json_measurement_product(
        Path(path),
        expected_scenario_id=expected_scenario_id,
    )


def dump_measurements_json(scenario_id: str, measurements: list[MeasurementRecord]) -> str:
    return json.dumps(
        {
            "scenario_id": scenario_id,
            "measurements": [record.model_dump(mode="json") for record in measurements],
        },
        indent=2,
    )


def dump_measurements_csv(scenario_id: str, measurements: list[MeasurementRecord]) -> str:
    fieldnames = [
        "scenario_id",
        "measurement_type",
        "epoch",
        "observer",
        "observed_object",
        "value",
        "sigma",
        "units",
        "metadata_json",
    ]
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in measurements:
        payload = record.model_dump(mode="json")
        writer.writerow(
            {
                "scenario_id": scenario_id,
                "measurement_type": payload["measurement_type"],
                "epoch": payload["epoch"],
                "observer": payload["observer"],
                "observed_object": payload["observed_object"],
                "value": payload["value"],
                "sigma": payload["sigma"],
                "units": payload["units"],
                "metadata_json": json.dumps(payload["metadata"], sort_keys=True),
            }
        )
    return output.getvalue().rstrip("\r\n")


def dump_measurements_tdm(
    scenario_id: str,
    measurements: list[MeasurementRecord],
    *,
    originator: str = "ASTRO_SUITE",
) -> str:
    if not measurements:
        raise InvalidMeasurementFileError("At least one measurement is required for TDM export")
    unsupported_types = {
        record.measurement_type
        for record in measurements
        if record.measurement_type not in TDM_EXPORT_MEASUREMENT_TYPES
    }
    if unsupported_types:
        unsupported_text = ", ".join(
            sorted(str(measurement_type) for measurement_type in unsupported_types)
        )
        raise InvalidMeasurementFileError(
            "TDM export supports range, range_rate, two_way_range, "
            "doppler, two_way_range_rate, three_way_range, three_way_range_rate, "
            "right_ascension, declination, azimuth, and elevation measurements; "
            f"unsupported measurement types: {unsupported_text}"
        )

    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE = {_format_tdm_epoch(measurements[0].epoch)}",
        f"ORIGINATOR = {originator}",
    ]
    for (
        observer,
        observed_object,
        angle_type,
        link_type,
        transmitter,
        suite_metadata,
    ), segment_records in _measurement_segments(measurements).items():
        lines.extend(["META_START", f"SCENARIO_ID = {scenario_id}", "TIME_SYSTEM = UTC"])
        lines.append("MODE = SEQUENTIAL")
        if link_type == "three_way":
            if transmitter is None:
                raise InvalidMeasurementFileError(
                    "TDM three-way measurements require metadata transmitter"
                )
            lines.extend(
                [
                    f"PARTICIPANT_1 = {transmitter}",
                    f"PARTICIPANT_2 = {observed_object}",
                    f"PARTICIPANT_3 = {observer}",
                    "PATH = 1,2,3",
                    "ASTRO_MEASUREMENT_TYPE = three_way",
                ]
            )
        elif link_type == "doppler_hz":
            lines.extend(
                [
                    f"PARTICIPANT_1 = {observer}",
                    f"PARTICIPANT_2 = {observed_object}",
                    "PATH = 1,2",
                    "ASTRO_MEASUREMENT_TYPE = doppler_hz",
                ]
            )
        else:
            lines.extend(
                [
                    f"PARTICIPANT_1 = {observer}",
                    f"PARTICIPANT_2 = {observed_object}",
                    "PATH = 1,2,1",
                ]
            )
            if link_type == "two_way":
                lines.append("ASTRO_MEASUREMENT_TYPE = two_way")

        if link_type == "doppler_hz":
            lines.append("DOPPLER_UNITS = Hz")
            doppler_sigma = _common_sigma_for_types(
                segment_records,
                frozenset({MeasurementType.DOPPLER}),
            )
            if doppler_sigma is not None:
                lines.append(f"DOPPLER_SIGMA_HZ = {_format_float(doppler_sigma)}")
        elif angle_type is None:
            lines.append("RANGE_UNITS = km")
            range_sigma = _common_sigma_for_types(segment_records, RANGE_MEASUREMENT_TYPES)
            if range_sigma is not None:
                lines.append(f"RANGE_SIGMA_KM = {_format_float(range_sigma)}")
            range_rate_sigma = _common_sigma_for_types(
                segment_records,
                RANGE_RATE_MEASUREMENT_TYPES,
            )
            if range_rate_sigma is not None:
                lines.append(f"RANGE_RATE_SIGMA_KM_S = {_format_float(range_rate_sigma)}")
        else:
            lines.extend([f"ANGLE_TYPE = {angle_type}", "ANGLE_UNITS = deg"])
            angle_sigma = _common_sigma_for_types(segment_records, ANGLE_MEASUREMENT_TYPES)
            if angle_sigma is not None:
                lines.append(f"ANGLE_SIGMA_DEG = {_format_float(angle_sigma)}")
        lines.extend(f"{key} = {value}" for key, value in suite_metadata)

        lines.extend(["META_STOP", "DATA_START"])
        for record in segment_records:
            keyword = TDM_DATA_KEYWORD_BY_MEASUREMENT_TYPE[record.measurement_type]
            epoch = _format_tdm_epoch(record.epoch)
            value = _format_float(record.value)
            lines.append(f"{keyword} = {epoch} {value}")
        lines.append("DATA_STOP")
    return "\n".join(lines)


def resolve_measurement_format(
    path: Path | str,
    measurement_format: str = "auto",
) -> Literal["json", "csv", "tdm"]:
    measurement_path = Path(path)
    normalized_format = measurement_format.lower()

    if normalized_format == "json":
        return "json"
    if normalized_format == "csv":
        return "csv"
    if normalized_format == "tdm":
        return "tdm"

    if normalized_format != "auto":
        raise InvalidMeasurementFileError(
            f"Unsupported measurement format {measurement_format!r}. "
            "Supported formats are auto, json, csv, and tdm."
        )

    suffix = measurement_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tdm":
        return "tdm"

    raise InvalidMeasurementFileError(
        f"Unsupported measurement format for {measurement_path}: "
        "could not infer from file extension; use json, csv, or tdm."
    )


def _load_json_measurements(
    measurement_path: Path,
    *,
    expected_scenario_id: str | None,
) -> list[MeasurementRecord]:
    return _load_json_measurement_product(
        measurement_path,
        expected_scenario_id=expected_scenario_id,
    ).measurements


def _load_json_measurement_product(
    measurement_path: Path,
    *,
    expected_scenario_id: str | None,
) -> MeasurementProduct:
    try:
        raw: Any = json.loads(measurement_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidMeasurementFileError(
            f"Could not read measurement file {measurement_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidMeasurementFileError(
            f"Could not parse measurement file {measurement_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} must contain a JSON object"
        )

    scenario_id = raw.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} must contain a non-empty scenario_id"
        )
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} scenario_id {scenario_id!r} "
            f"does not match expected scenario_id {expected_scenario_id!r}"
        )

    measurements = raw.get("measurements")
    if not isinstance(measurements, list):
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} must contain a measurements list"
        )

    try:
        records = [MeasurementRecord.model_validate(record) for record in measurements]
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} is invalid: {exc}"
        ) from exc

    return MeasurementProduct(scenario_id=scenario_id, measurements=records)


def _load_csv_measurements(
    measurement_path: Path,
    *,
    expected_scenario_id: str | None,
) -> list[MeasurementRecord]:
    try:
        with measurement_path.open("r", encoding="utf-8", newline="") as measurement_file:
            reader = csv.DictReader(measurement_file)
            _validate_csv_header(measurement_path, reader.fieldnames)
            return [
                _measurement_record_from_csv_row(
                    measurement_path,
                    row_number=row_number,
                    row=row,
                    expected_scenario_id=expected_scenario_id,
                )
                for row_number, row in enumerate(reader, start=2)
            ]
    except (OSError, UnicodeError) as exc:
        raise InvalidMeasurementFileError(
            f"Could not read measurement file {measurement_path}: {exc}"
        ) from exc
    except csv.Error as exc:
        raise InvalidMeasurementFileError(
            f"Could not parse CSV measurement file {measurement_path}: {exc}"
        ) from exc


def _validate_csv_header(measurement_path: Path, fieldnames: Sequence[str] | None) -> None:
    columns = set(fieldnames or [])
    missing_columns = sorted(CSV_REQUIRED_COLUMNS - columns)
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} is missing required columns: "
            f"{joined_columns}"
        )


def _measurement_record_from_csv_row(
    measurement_path: Path,
    *,
    row_number: int,
    row: dict[str, str | None],
    expected_scenario_id: str | None,
) -> MeasurementRecord:
    scenario_id = _required_csv_cell(measurement_path, row_number, row, "scenario_id")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} scenario_id "
            f"{scenario_id!r} does not match expected scenario_id {expected_scenario_id!r}"
        )

    payload: dict[str, Any] = {
        "measurement_type": _required_csv_cell(
            measurement_path, row_number, row, "measurement_type"
        ),
        "epoch": _required_csv_cell(measurement_path, row_number, row, "epoch"),
        "observer": _required_csv_cell(measurement_path, row_number, row, "observer"),
        "observed_object": _required_csv_cell(
            measurement_path, row_number, row, "observed_object"
        ),
        "value": _required_csv_float(measurement_path, row_number, row, "value"),
        "sigma": _required_csv_float(measurement_path, row_number, row, "sigma"),
        "units": _required_csv_cell(measurement_path, row_number, row, "units"),
    }

    metadata = _csv_metadata(measurement_path, row_number, row)
    if metadata:
        payload["metadata"] = metadata

    try:
        return MeasurementRecord.model_validate(payload)
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} is invalid: {exc}"
        ) from exc


def _required_csv_cell(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
    column: str,
) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} "
            f"has an empty {column} value"
        )
    return value.strip()


def _required_csv_float(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
    column: str,
) -> float:
    value = _required_csv_cell(measurement_path, row_number, row, column)
    try:
        return float(value)
    except ValueError as exc:
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} "
            f"has non-numeric {column} value {value!r}"
        ) from exc


def _csv_metadata(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
) -> dict[str, Any]:
    raw_metadata = row.get("metadata_json")
    if raw_metadata is None or raw_metadata.strip() == "":
        return {}

    try:
        metadata: Any = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} "
            f"metadata_json is invalid JSON: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise InvalidMeasurementFileError(
            f"CSV measurement file {measurement_path} row {row_number} "
            "metadata_json must contain a JSON object"
        )
    return metadata


def _measurement_segments(
    measurements: list[MeasurementRecord],
) -> dict[TdmSegmentKey, list[MeasurementRecord]]:
    segments: dict[TdmSegmentKey, list[MeasurementRecord]] = {}
    for record in measurements:
        link_type = _tdm_link_type(record)
        transmitter = _tdm_transmitter(record) if link_type == "three_way" else None
        key = (
            record.observer,
            record.observed_object,
            TDM_ANGLE_TYPE_BY_MEASUREMENT_TYPE.get(record.measurement_type),
            link_type,
            transmitter,
            _tdm_suite_metadata_signature(record),
        )
        segments.setdefault(key, []).append(record)
    return segments


def _tdm_link_type(record: MeasurementRecord) -> str | None:
    if record.measurement_type is MeasurementType.DOPPLER:
        return "doppler_hz"
    if record.measurement_type in {
        MeasurementType.TWO_WAY_RANGE,
        MeasurementType.TWO_WAY_RANGE_RATE,
    }:
        return "two_way"
    if record.measurement_type in {
        MeasurementType.THREE_WAY_RANGE,
        MeasurementType.THREE_WAY_RANGE_RATE,
    }:
        return "three_way"
    return None


def _tdm_transmitter(record: MeasurementRecord) -> str:
    transmitter = record.metadata.get("transmitter")
    if not isinstance(transmitter, str) or transmitter.strip() == "":
        raise InvalidMeasurementFileError(
            "TDM three-way measurements require metadata transmitter"
        )
    return transmitter.strip()


def _tdm_suite_metadata_signature(record: MeasurementRecord) -> TdmMetadataSignature:
    signature: list[tuple[str, str]] = []
    for record_key, tdm_key in TDM_SUITE_METADATA_KEYS:
        if record_key not in record.metadata:
            continue
        value = record.metadata[record_key]
        if isinstance(value, bool | list | dict | tuple):
            continue
        signature.append((tdm_key, _format_tdm_metadata_value(value)))
    return tuple(signature)


def _format_tdm_metadata_value(value: object) -> str:
    if isinstance(value, int | float):
        return _format_float(float(value))
    return str(value)


def _common_sigma(
    measurements: list[MeasurementRecord],
    measurement_type: MeasurementType,
) -> float | None:
    sigmas = {
        record.sigma
        for record in measurements
        if record.measurement_type is measurement_type
    }
    if not sigmas:
        return None
    if len(sigmas) == 1:
        return next(iter(sigmas))
    return None


def _common_sigma_for_types(
    measurements: list[MeasurementRecord],
    measurement_types: frozenset[MeasurementType],
) -> float | None:
    sigmas = {
        record.sigma
        for record in measurements
        if record.measurement_type in measurement_types
    }
    if not sigmas:
        return None
    if len(sigmas) == 1:
        return next(iter(sigmas))
    return None


def _format_tdm_epoch(epoch: datetime) -> str:
    return epoch.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_float(value: float) -> str:
    return str(value)


def _load_tdm_measurements(
    measurement_path: Path,
    *,
    expected_scenario_id: str | None,
) -> list[MeasurementRecord]:
    try:
        lines = measurement_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InvalidMeasurementFileError(
            f"Could not read measurement file {measurement_path}: {exc}"
        ) from exc

    header: dict[str, str] = {}
    pending_metadata: dict[str, str] | None = None
    segment_metadata: dict[str, str] | None = None
    active_metadata: dict[str, str] | None = None
    section = "header"
    records: list[MeasurementRecord] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        key, value = _tdm_key_value(measurement_path, line_number, line)
        if key == "COMMENT":
            continue
        if key == "META_START":
            pending_metadata = {}
            section = "metadata"
            continue
        if key == "META_STOP":
            if pending_metadata is None:
                raise InvalidMeasurementFileError(
                    f"TDM file {measurement_path} line {line_number} has META_STOP "
                    "without META_START"
                )
            segment_metadata = pending_metadata
            pending_metadata = None
            section = "header"
            continue
        if key == "DATA_START":
            if segment_metadata is None:
                raise InvalidMeasurementFileError(
                    f"TDM file {measurement_path} line {line_number} has DATA_START "
                    "without preceding metadata"
                )
            active_metadata = segment_metadata
            section = "data"
            continue
        if key == "DATA_STOP":
            active_metadata = None
            segment_metadata = None
            section = "header"
            continue

        if section == "metadata":
            if pending_metadata is None:
                raise InvalidMeasurementFileError(
                    f"TDM file {measurement_path} line {line_number} has metadata "
                    "outside META_START/META_STOP"
                )
            pending_metadata[key] = value
            continue

        if section == "data":
            if active_metadata is None:
                raise InvalidMeasurementFileError(
                    f"TDM file {measurement_path} line {line_number} has data "
                    "outside DATA_START/DATA_STOP"
                )
            record = _tdm_record_from_data_line(
                measurement_path,
                line_number=line_number,
                keyword=key,
                raw_value=value,
                metadata=active_metadata,
                header=header,
                expected_scenario_id=expected_scenario_id,
            )
            if record is not None:
                records.append(record)
            continue

        header[key] = value

    if pending_metadata is not None:
        raise InvalidMeasurementFileError(f"TDM file {measurement_path} ended inside META block")
    if active_metadata is not None:
        raise InvalidMeasurementFileError(f"TDM file {measurement_path} ended inside DATA block")
    if not records:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} contains no supported RANGE, Doppler, or angle "
            "measurements"
        )
    return records


def _tdm_key_value(
    measurement_path: Path,
    line_number: int,
    line: str,
) -> tuple[str, str]:
    if "=" in line:
        raw_key, raw_value = line.split("=", maxsplit=1)
        key = raw_key.strip().upper()
        value = raw_value.strip()
    else:
        key = line.strip().upper()
        value = ""

    if not key:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} has an empty keyword"
        )
    return key, value


def _tdm_record_from_data_line(
    measurement_path: Path,
    *,
    line_number: int,
    keyword: str,
    raw_value: str,
    metadata: dict[str, str],
    header: dict[str, str],
    expected_scenario_id: str | None,
) -> MeasurementRecord | None:
    if keyword not in SUPPORTED_TDM_DATA_KEYWORDS:
        return None

    scenario_id = metadata.get("SCENARIO_ID")
    if (
        expected_scenario_id is not None
        and scenario_id is not None
        and scenario_id != expected_scenario_id
    ):
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} scenario_id {scenario_id!r} "
            f"does not match expected scenario_id {expected_scenario_id!r}"
        )

    epoch_text, value_text = _tdm_observable_fields(measurement_path, line_number, raw_value)
    time_system = _required_tdm_metadata(measurement_path, line_number, metadata, "TIME_SYSTEM")
    epoch = _tdm_epoch(measurement_path, line_number, epoch_text, time_system)
    link_type = _tdm_link_type_from_metadata(measurement_path, line_number, metadata)
    observer, observed_object, participant_metadata = _tdm_observer_and_object(
        measurement_path,
        line_number,
        metadata,
        link_type=link_type,
    )

    try:
        value = float(value_text)
    except ValueError as exc:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} has non-numeric value "
            f"{value_text!r}"
        ) from exc

    if keyword == "RANGE":
        range_units = metadata.get("RANGE_UNITS", "km")
        if range_units.lower() != "km":
            raise InvalidMeasurementFileError(
                f"TDM file {measurement_path} line {line_number} uses unsupported "
                f"RANGE_UNITS {range_units!r}; only km is supported"
            )
        measurement_type = _tdm_radiometric_measurement_type(
            measurement_path,
            line_number=line_number,
            link_type=link_type,
            keyword=keyword,
            default=MeasurementType.RANGE,
        )
        units = "km"
        sigma = _tdm_sigma(
            measurement_path,
            line_number=line_number,
            metadata=metadata,
            key="RANGE_SIGMA_KM",
            default=DEFAULT_TDM_RANGE_SIGMA_KM,
        )
        angle_type = None
    elif keyword in {"DOPPLER_INSTANTANEOUS", "DOPPLER_INTEGRATED"}:
        measurement_type = _tdm_radiometric_measurement_type(
            measurement_path,
            line_number=line_number,
            link_type=link_type,
            keyword=keyword,
            default=MeasurementType.RANGE_RATE,
        )
        if measurement_type is MeasurementType.DOPPLER:
            doppler_units = metadata.get("DOPPLER_UNITS", "Hz")
            if doppler_units.lower() != "hz":
                raise InvalidMeasurementFileError(
                    f"TDM file {measurement_path} line {line_number} uses unsupported "
                    f"DOPPLER_UNITS {doppler_units!r}; only Hz is supported for "
                    "ASTRO_MEASUREMENT_TYPE = doppler_hz"
                )
            units = "Hz"
            sigma = _tdm_sigma(
                measurement_path,
                line_number=line_number,
                metadata=metadata,
                key="DOPPLER_SIGMA_HZ",
                default=DEFAULT_TDM_DOPPLER_SIGMA_HZ,
            )
        else:
            units = "km/s"
            sigma = _tdm_sigma(
                measurement_path,
                line_number=line_number,
                metadata=metadata,
                key="RANGE_RATE_SIGMA_KM_S",
                default=DEFAULT_TDM_RANGE_RATE_SIGMA_KM_S,
            )
        angle_type = None
    else:
        measurement_type, angle_type = _tdm_angle_measurement_type(
            measurement_path,
            line_number=line_number,
            metadata=metadata,
            keyword=keyword,
        )
        angle_units = metadata.get("ANGLE_UNITS", "deg")
        if angle_units.lower() != "deg":
            raise InvalidMeasurementFileError(
                f"TDM file {measurement_path} line {line_number} uses unsupported "
                f"ANGLE_UNITS {angle_units!r}; only deg is supported"
            )
        units = "deg"
        sigma = _tdm_sigma(
            measurement_path,
            line_number=line_number,
            metadata=metadata,
            key="ANGLE_SIGMA_DEG",
            default=DEFAULT_TDM_ANGLE_SIGMA_DEG,
        )

    record_metadata = {
        "source_format": "ccsds_tdm_kvn",
        "tdm_keyword": keyword,
        "tdm_time_system": time_system,
        "tdm_mode": metadata.get("MODE"),
        "tdm_path": metadata.get("PATH"),
        "tdm_originator": header.get("ORIGINATOR"),
    } | participant_metadata | _tdm_suite_record_metadata(metadata)
    if link_type is not None:
        record_metadata["astro_measurement_type"] = link_type
    if angle_type is not None:
        record_metadata["tdm_angle_type"] = angle_type

    try:
        return MeasurementRecord.model_validate(
            {
                "measurement_type": measurement_type,
                "epoch": epoch,
                "observer": observer,
                "observed_object": observed_object,
                "value": value,
                "sigma": sigma,
                "units": units,
                "metadata": record_metadata,
            }
        )
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} is invalid: {exc}"
        ) from exc


def _tdm_suite_record_metadata(metadata: dict[str, str]) -> dict[str, str | float]:
    record_metadata: dict[str, str | float] = {}
    for tdm_key, record_key in TDM_RECORD_METADATA_BY_SUITE_KEY.items():
        if tdm_key not in metadata:
            continue
        value = metadata[tdm_key]
        if record_key.endswith(("_km", "_deg", "_hz", "_hpa", "_k", "_tecu", "_humidity")):
            record_metadata[record_key] = float(value)
        else:
            record_metadata[record_key] = value
    return record_metadata


def _tdm_angle_measurement_type(
    measurement_path: Path,
    *,
    line_number: int,
    metadata: dict[str, str],
    keyword: str,
) -> tuple[MeasurementType, str]:
    angle_type = metadata.get("ANGLE_TYPE")
    if angle_type is None or angle_type.strip() == "":
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} is missing ANGLE_TYPE metadata"
        )
    normalized_angle_type = angle_type.strip().upper()
    measurement_type = TDM_ANGLE_MEASUREMENT_TYPE_BY_ANGLE_TYPE_AND_KEYWORD.get(
        (normalized_angle_type, keyword)
    )
    if measurement_type is None:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} uses unsupported "
            f"ANGLE_TYPE {angle_type!r} with {keyword}; supported angle types are RADEC and AZEL"
        )
    return measurement_type, normalized_angle_type


def _tdm_link_type_from_metadata(
    measurement_path: Path,
    line_number: int,
    metadata: dict[str, str],
) -> str | None:
    raw_link_type = metadata.get("ASTRO_MEASUREMENT_TYPE")
    if raw_link_type is None or raw_link_type.strip() == "":
        return None
    link_type = raw_link_type.strip().lower()
    if link_type not in {"two_way", "three_way"}:
        if link_type == "doppler_hz":
            return link_type
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} uses unsupported "
            f"ASTRO_MEASUREMENT_TYPE {raw_link_type!r}; supported values are two_way, "
            "three_way, and doppler_hz"
        )
    return link_type


def _tdm_radiometric_measurement_type(
    measurement_path: Path,
    *,
    line_number: int,
    link_type: str | None,
    keyword: str,
    default: MeasurementType,
) -> MeasurementType:
    if link_type is None:
        return default
    measurement_type = TDM_RADIOMETRIC_MEASUREMENT_TYPE_BY_LINK_AND_KEYWORD.get(
        (link_type, keyword)
    )
    if measurement_type is None:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} uses unsupported "
            f"ASTRO_MEASUREMENT_TYPE {link_type!r} with {keyword}"
        )
    return measurement_type


def _tdm_observable_fields(
    measurement_path: Path,
    line_number: int,
    raw_value: str,
) -> tuple[str, str]:
    fields = raw_value.split()
    if len(fields) < 2:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} must contain epoch and value"
        )
    return fields[0], fields[1]


def _required_tdm_metadata(
    measurement_path: Path,
    line_number: int,
    metadata: dict[str, str],
    key: str,
) -> str:
    value = metadata.get(key)
    if value is None or value.strip() == "":
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} is missing {key} metadata"
        )
    return value.strip()


def _tdm_epoch(
    measurement_path: Path,
    line_number: int,
    epoch_text: str,
    time_system: str,
) -> datetime:
    if time_system.upper() != "UTC":
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} uses unsupported "
            f"TIME_SYSTEM {time_system!r}; only UTC is supported"
        )

    normalized_epoch = epoch_text[:-1] + "+00:00" if epoch_text.endswith("Z") else epoch_text
    try:
        epoch = datetime.fromisoformat(normalized_epoch)
    except ValueError:
        epoch = _tdm_ordinal_epoch(measurement_path, line_number, epoch_text)

    if epoch.tzinfo is None or epoch.utcoffset() is None:
        epoch = epoch.replace(tzinfo=UTC)
    return epoch


def _tdm_ordinal_epoch(
    measurement_path: Path,
    line_number: int,
    epoch_text: str,
) -> datetime:
    for epoch_format in ("%Y-%jT%H:%M:%S.%f", "%Y-%jT%H:%M:%S"):
        try:
            return datetime.strptime(epoch_text, epoch_format).replace(tzinfo=UTC)
        except ValueError:
            continue

    raise InvalidMeasurementFileError(
        f"TDM file {measurement_path} line {line_number} has invalid UTC epoch "
        f"{epoch_text!r}"
    )


def _tdm_observer_and_object(
    measurement_path: Path,
    line_number: int,
    metadata: dict[str, str],
    *,
    link_type: str | None,
) -> tuple[str, str, dict[str, str]]:
    participants = _tdm_participants(metadata)
    path = _tdm_path(metadata.get("PATH"))

    if link_type == "three_way":
        if path and len(path) >= 3:
            transmitter_index = path[0]
            observed_index = path[1]
            observer_index = path[2]
        else:
            transmitter_index = 1
            observed_index = 2
            observer_index = 3
        try:
            participant_path = ",".join(
                (
                    participants[transmitter_index],
                    participants[observed_index],
                    participants[observer_index],
                )
            )
            return (
                participants[observer_index],
                participants[observed_index],
                {
                    "transmitter": participants[transmitter_index],
                    "participant_path": participant_path,
                },
            )
        except KeyError as exc:
            raise InvalidMeasurementFileError(
                f"TDM file {measurement_path} line {line_number} is missing PARTICIPANT "
                f"metadata for PATH-derived participant {exc.args[0]}"
            ) from exc

    if path and len(path) >= 2:
        observer_index = path[0]
        observed_index = path[1]
    else:
        observer_index = 1
        observed_index = 2

    try:
        observer = participants[observer_index]
        observed_object = participants[observed_index]
        participant_path = ",".join(
            participants[index] for index in path
        ) if path else f"{observer},{observed_object}"
        return observer, observed_object, {"participant_path": participant_path}
    except KeyError as exc:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} is missing PARTICIPANT "
            f"metadata for PATH-derived participant {exc.args[0]}"
        ) from exc


def _tdm_participants(metadata: dict[str, str]) -> dict[int, str]:
    participants: dict[int, str] = {}
    for key, value in metadata.items():
        if not key.startswith("PARTICIPANT_"):
            continue
        try:
            participant_index = int(key.removeprefix("PARTICIPANT_"))
        except ValueError:
            continue
        participants[participant_index] = value
    return participants


def _tdm_path(path: str | None) -> list[int]:
    if path is None:
        return []
    cleaned_path = path.replace("{", " ").replace("}", " ").replace(",", " ")
    return [int(component) for component in cleaned_path.split()]


def _tdm_sigma(
    measurement_path: Path,
    *,
    line_number: int,
    metadata: dict[str, str],
    key: str,
    default: float,
) -> float:
    raw_sigma = metadata.get(key)
    if raw_sigma is None:
        return default
    try:
        return float(raw_sigma)
    except ValueError as exc:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} has non-numeric "
            f"{key} value {raw_sigma!r}"
        ) from exc
