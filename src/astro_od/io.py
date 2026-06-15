from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from astro_core.errors import InvalidMeasurementFileError
from astro_core.models import MeasurementRecord


def load_measurements(
    path: Path | str,
    *,
    expected_scenario_id: str | None = None,
) -> list[MeasurementRecord]:
    measurement_path = Path(path)
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
