from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from struct import calcsize, unpack_from
from typing import Any

from pydantic import ValidationError

from astro_core.errors import InvalidMeasurementFileError
from astro_core.models import MeasurementRecord, MeasurementType
from astro_od.io import MeasurementProduct

DSN_TRACKING_REQUIRED_COLUMNS = frozenset(
    {
        "scenario_id",
        "tracking_format",
        "observable",
        "epoch",
        "station",
        "spacecraft",
        "value",
        "sigma",
        "units",
    }
)

DSN_OBSERVABLE_TYPES = {
    "range": MeasurementType.RANGE,
    "range_rate": MeasurementType.RANGE_RATE,
    "two_way_range": MeasurementType.TWO_WAY_RANGE,
    "two_way_range_rate": MeasurementType.TWO_WAY_RANGE_RATE,
    "three_way_range": MeasurementType.THREE_WAY_RANGE,
    "three_way_range_rate": MeasurementType.THREE_WAY_RANGE_RATE,
    "doppler": MeasurementType.DOPPLER,
}

DSN_TRACKING_FORMATS = {"odf", "tnf"}
DSN_BINARY_MAGIC = b"ASTRODSN1"
DSN_BINARY_HEADER = "<BBqddB"
DSN_BINARY_HEADER_SIZE = calcsize(DSN_BINARY_HEADER)
DSN_BINARY_TRACKING_FORMATS = {1: "odf", 2: "tnf"}
DSN_BINARY_OBSERVABLE_TYPES = {
    1: ("range", MeasurementType.RANGE),
    2: ("range_rate", MeasurementType.RANGE_RATE),
    3: ("two_way_range", MeasurementType.TWO_WAY_RANGE),
    4: ("two_way_range_rate", MeasurementType.TWO_WAY_RANGE_RATE),
    5: ("three_way_range", MeasurementType.THREE_WAY_RANGE),
    6: ("three_way_range_rate", MeasurementType.THREE_WAY_RANGE_RATE),
    7: ("doppler", MeasurementType.DOPPLER),
}
DSN_BINARY_UNITS = {1: "km", 2: "km/s", 3: "Hz", 4: "deg"}


def load_dsn_tracking_measurements(path: Path | str) -> MeasurementProduct:
    """Load normalized DSN ODF/TNF-style tracking rows into suite measurements."""
    measurement_path = Path(path)
    try:
        with measurement_path.open("r", encoding="utf-8", newline="") as measurement_file:
            reader = csv.DictReader(measurement_file)
            if reader.fieldnames is None:
                raise InvalidMeasurementFileError(
                    f"DSN tracking file {measurement_path} has no header"
                )
            _validate_dsn_tracking_columns(measurement_path, reader.fieldnames)
            records = [
                _dsn_tracking_record(measurement_path, row_number, row)
                for row_number, row in enumerate(reader, start=2)
            ]
    except OSError as exc:
        raise InvalidMeasurementFileError(
            f"could not read DSN tracking file {measurement_path}: {exc}"
        ) from exc

    if not records:
        raise InvalidMeasurementFileError(f"DSN tracking file {measurement_path} contains no rows")
    scenario_ids = {record.metadata["dsn_scenario_id"] for record in records}
    if len(scenario_ids) != 1:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} must contain a single scenario_id"
        )
    tracking_formats = sorted(
        {str(record.metadata["dsn_tracking_format"]) for record in records}
    )
    return MeasurementProduct(
        scenario_id=str(next(iter(scenario_ids))),
        measurements=records,
        metadata={
            "source_format": "normalized_dsn_tracking_csv",
            "tracking_formats": tracking_formats,
            "measurement_count": len(records),
        },
    )


def load_dsn_binary_tracking_measurements(path: Path | str) -> MeasurementProduct:
    """Load suite-owned ASTRODSN1 binary tracking bridge records."""
    measurement_path = Path(path)
    try:
        payload = measurement_path.read_bytes()
    except OSError as exc:
        raise InvalidMeasurementFileError(
            f"could not read DSN binary tracking file {measurement_path}: {exc}"
        ) from exc

    records = _dsn_binary_tracking_records(measurement_path, payload)
    if not records:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} contains no rows"
        )
    scenario_ids = {record.metadata["dsn_scenario_id"] for record in records}
    if len(scenario_ids) != 1:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} must contain a single scenario_id"
        )
    tracking_formats = sorted(
        {str(record.metadata["dsn_tracking_format"]) for record in records}
    )
    return MeasurementProduct(
        scenario_id=str(next(iter(scenario_ids))),
        measurements=records,
        metadata={
            "source_format": "astro_dsn_binary_tracking",
            "binary_format": "ASTRODSN1",
            "tracking_formats": tracking_formats,
            "measurement_count": len(records),
        },
    )


def _validate_dsn_tracking_columns(measurement_path: Path, fieldnames: Sequence[str]) -> None:
    missing_columns = DSN_TRACKING_REQUIRED_COLUMNS - set(fieldnames)
    if missing_columns:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _dsn_tracking_record(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
) -> MeasurementRecord:
    tracking_format = _required_text(measurement_path, row_number, row, "tracking_format").lower()
    if tracking_format not in DSN_TRACKING_FORMATS:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} row {row_number} uses unsupported "
            f"tracking_format {tracking_format!r}; supported values are odf and tnf"
        )
    observable = _required_text(measurement_path, row_number, row, "observable").lower()
    measurement_type = DSN_OBSERVABLE_TYPES.get(observable)
    if measurement_type is None:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} row {row_number} uses unsupported "
            f"observable {observable!r}"
        )

    metadata: dict[str, Any] = {
        "source_format": "normalized_dsn_tracking_csv",
        "dsn_tracking_format": tracking_format,
        "dsn_observable": observable,
        "dsn_scenario_id": _required_text(measurement_path, row_number, row, "scenario_id"),
    }
    _copy_optional_metadata(row, metadata, "participant_path")
    _copy_optional_metadata(row, metadata, "transmitter")
    media_source = row.get("media_source")
    if media_source is not None and media_source.strip():
        metadata["media_corrections_source"] = media_source.strip()

    try:
        return MeasurementRecord.model_validate(
            {
                "measurement_type": measurement_type,
                "epoch": _required_text(measurement_path, row_number, row, "epoch"),
                "observer": _required_text(measurement_path, row_number, row, "station"),
                "observed_object": _required_text(
                    measurement_path,
                    row_number,
                    row,
                    "spacecraft",
                ),
                "value": _required_float(measurement_path, row_number, row, "value"),
                "sigma": _required_float(measurement_path, row_number, row, "sigma"),
                "units": _required_text(measurement_path, row_number, row, "units"),
                "metadata": metadata,
            }
        )
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} row {row_number} is invalid: {exc}"
        ) from exc


def _dsn_binary_tracking_records(
    measurement_path: Path,
    payload: bytes,
) -> list[MeasurementRecord]:
    if not payload.startswith(DSN_BINARY_MAGIC):
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} has invalid ASTRODSN1 magic"
        )
    offset = len(DSN_BINARY_MAGIC)
    if len(payload) < offset + 4:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} is missing record count"
        )
    record_count = unpack_from("<I", payload, offset)[0]
    offset += 4
    records: list[MeasurementRecord] = []
    for record_index in range(record_count):
        record, offset = _dsn_binary_tracking_record(
            measurement_path,
            record_index,
            payload,
            offset,
        )
        records.append(record)
    if offset != len(payload):
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} has trailing bytes"
        )
    return records


def _dsn_binary_tracking_record(
    measurement_path: Path,
    record_index: int,
    payload: bytes,
    offset: int,
) -> tuple[MeasurementRecord, int]:
    if len(payload) < offset + DSN_BINARY_HEADER_SIZE:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} is truncated"
        )
    (
        tracking_format_code,
        observable_code,
        epoch_unix_s,
        value,
        sigma,
        units_code,
    ) = unpack_from(DSN_BINARY_HEADER, payload, offset)
    offset += DSN_BINARY_HEADER_SIZE
    try:
        tracking_format = DSN_BINARY_TRACKING_FORMATS[tracking_format_code]
        observable, measurement_type = DSN_BINARY_OBSERVABLE_TYPES[observable_code]
        units = DSN_BINARY_UNITS[units_code]
    except KeyError as exc:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} "
            "uses an unsupported tracking, observable, or units code"
        ) from exc

    scenario_id, offset = _read_binary_text(
        measurement_path, record_index, payload, offset, "scenario_id"
    )
    station, offset = _read_binary_text(
        measurement_path, record_index, payload, offset, "station"
    )
    spacecraft, offset = _read_binary_text(
        measurement_path, record_index, payload, offset, "spacecraft"
    )
    participant_path, offset = _read_binary_text(
        measurement_path,
        record_index,
        payload,
        offset,
        "participant_path",
        required=False,
    )
    transmitter, offset = _read_binary_text(
        measurement_path,
        record_index,
        payload,
        offset,
        "transmitter",
        required=False,
    )
    media_source, offset = _read_binary_text(
        measurement_path,
        record_index,
        payload,
        offset,
        "media_source",
        required=False,
    )
    metadata: dict[str, Any] = {
        "source_format": "astro_dsn_binary_tracking",
        "binary_format": "ASTRODSN1",
        "binary_record_index": record_index,
        "dsn_tracking_format": tracking_format,
        "dsn_observable": observable,
        "dsn_scenario_id": scenario_id,
    }
    if participant_path:
        metadata["participant_path"] = participant_path
    if transmitter:
        metadata["transmitter"] = transmitter
    if media_source:
        metadata["media_corrections_source"] = media_source
    try:
        record = MeasurementRecord.model_validate(
            {
                "measurement_type": measurement_type,
                "epoch": datetime.fromtimestamp(epoch_unix_s, UTC),
                "observer": station,
                "observed_object": spacecraft,
                "value": value,
                "sigma": sigma,
                "units": units,
                "metadata": metadata,
            }
        )
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} is invalid: {exc}"
        ) from exc
    return record, offset


def _read_binary_text(
    measurement_path: Path,
    record_index: int,
    payload: bytes,
    offset: int,
    field_name: str,
    *,
    required: bool = True,
) -> tuple[str, int]:
    if len(payload) < offset + 2:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} "
            f"is missing {field_name} length"
        )
    field_length = unpack_from("<H", payload, offset)[0]
    offset += 2
    end_offset = offset + field_length
    if len(payload) < end_offset:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} "
            f"has truncated {field_name}"
        )
    value = payload[offset:end_offset].decode("utf-8")
    if required and not value:
        raise InvalidMeasurementFileError(
            f"DSN binary tracking file {measurement_path} record {record_index} "
            f"is missing {field_name}"
        )
    return value, end_offset


def _required_text(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
    column: str,
) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} row {row_number} is missing {column}"
        )
    return value.strip()


def _required_float(
    measurement_path: Path,
    row_number: int,
    row: dict[str, str | None],
    column: str,
) -> float:
    value = _required_text(measurement_path, row_number, row, column)
    try:
        return float(value)
    except ValueError as exc:
        raise InvalidMeasurementFileError(
            f"DSN tracking file {measurement_path} row {row_number} has non-numeric {column}"
        ) from exc


def _copy_optional_metadata(
    row: dict[str, str | None],
    metadata: dict[str, Any],
    column: str,
) -> None:
    value = row.get(column)
    if value is not None and value.strip():
        metadata[column] = value.strip()
