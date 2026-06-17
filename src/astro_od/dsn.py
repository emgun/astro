from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
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
