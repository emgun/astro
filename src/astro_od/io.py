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
TDM_EXPORT_MEASUREMENT_TYPES = frozenset(
    {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    }
)

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
    }
)


@dataclass(frozen=True)
class MeasurementProduct:
    scenario_id: str
    measurements: list[MeasurementRecord]


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
            "TDM export supports only range and range_rate measurements; "
            f"unsupported measurement types: {unsupported_text}"
        )

    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE = {_format_tdm_epoch(measurements[0].epoch)}",
        f"ORIGINATOR = {originator}",
    ]
    for (observer, observed_object), segment_records in _measurement_segments(measurements).items():
        lines.extend(
            [
                "META_START",
                f"SCENARIO_ID = {scenario_id}",
                "TIME_SYSTEM = UTC",
                "MODE = SEQUENTIAL",
                f"PARTICIPANT_1 = {observer}",
                f"PARTICIPANT_2 = {observed_object}",
                "PATH = 1,2,1",
                "RANGE_UNITS = km",
            ]
        )
        range_sigma = _common_sigma(segment_records, MeasurementType.RANGE)
        if range_sigma is not None:
            lines.append(f"RANGE_SIGMA_KM = {_format_float(range_sigma)}")
        range_rate_sigma = _common_sigma(segment_records, MeasurementType.RANGE_RATE)
        if range_rate_sigma is not None:
            lines.append(f"RANGE_RATE_SIGMA_KM_S = {_format_float(range_rate_sigma)}")

        lines.extend(["META_STOP", "DATA_START"])
        for record in segment_records:
            keyword = (
                "RANGE"
                if record.measurement_type is MeasurementType.RANGE
                else "DOPPLER_INSTANTANEOUS"
            )
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
) -> dict[tuple[str, str], list[MeasurementRecord]]:
    segments: dict[tuple[str, str], list[MeasurementRecord]] = {}
    for record in measurements:
        key = (record.observer, record.observed_object)
        segments.setdefault(key, []).append(record)
    return segments


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
            f"TDM file {measurement_path} contains no supported RANGE or Doppler measurements"
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
    observer, observed_object = _tdm_observer_and_object(measurement_path, line_number, metadata)

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
        measurement_type = "range"
        units = "km"
        sigma = _tdm_sigma(
            measurement_path,
            line_number=line_number,
            metadata=metadata,
            key="RANGE_SIGMA_KM",
            default=DEFAULT_TDM_RANGE_SIGMA_KM,
        )
    else:
        measurement_type = "range_rate"
        units = "km/s"
        sigma = _tdm_sigma(
            measurement_path,
            line_number=line_number,
            metadata=metadata,
            key="RANGE_RATE_SIGMA_KM_S",
            default=DEFAULT_TDM_RANGE_RATE_SIGMA_KM_S,
        )

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
                "metadata": {
                    "source_format": "ccsds_tdm_kvn",
                    "tdm_keyword": keyword,
                    "tdm_time_system": time_system,
                    "tdm_mode": metadata.get("MODE"),
                    "tdm_path": metadata.get("PATH"),
                    "tdm_originator": header.get("ORIGINATOR"),
                },
            }
        )
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"TDM file {measurement_path} line {line_number} is invalid: {exc}"
        ) from exc


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
) -> tuple[str, str]:
    participants = _tdm_participants(metadata)
    path = _tdm_path(metadata.get("PATH"))

    if path and len(path) >= 2:
        observer_index = path[0]
        observed_index = path[1]
    else:
        observer_index = 1
        observed_index = 2

    try:
        return participants[observer_index], participants[observed_index]
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
