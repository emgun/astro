from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from astro_core.errors import InvalidMeasurementFileError
from astro_core.models import MeasurementRecord

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


def load_measurements(
    path: Path | str,
    *,
    expected_scenario_id: str | None = None,
    measurement_format: str = "auto",
) -> list[MeasurementRecord]:
    measurement_path = Path(path)
    resolved_format = resolve_measurement_format(measurement_path, measurement_format)
    if resolved_format == "csv":
        return _load_csv_measurements(
            measurement_path,
            expected_scenario_id=expected_scenario_id,
        )
    return _load_json_measurements(
        measurement_path,
        expected_scenario_id=expected_scenario_id,
    )


def resolve_measurement_format(
    path: Path | str,
    measurement_format: str = "auto",
) -> Literal["json", "csv"]:
    measurement_path = Path(path)
    normalized_format = measurement_format.lower()

    if normalized_format == "json":
        return "json"
    if normalized_format == "csv":
        return "csv"

    if normalized_format != "auto":
        raise InvalidMeasurementFileError(
            f"Unsupported measurement format {measurement_format!r}. "
            "Supported formats are auto, json, and csv."
        )

    suffix = measurement_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"

    raise InvalidMeasurementFileError(
        f"Unsupported measurement format for {measurement_path}: "
        "could not infer from file extension; use json or csv."
    )


def _load_json_measurements(
    measurement_path: Path,
    *,
    expected_scenario_id: str | None,
) -> list[MeasurementRecord]:
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
        return [MeasurementRecord.model_validate(record) for record in measurements]
    except ValidationError as exc:
        raise InvalidMeasurementFileError(
            f"Measurement file {measurement_path} is invalid: {exc}"
        ) from exc


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
