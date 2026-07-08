from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario


def load_twin_scenario(path: Path | str) -> DigitalTwinScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read twin scenario {scenario_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(f"Could not parse twin scenario {scenario_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Twin scenario file {scenario_path} must contain a mapping")
    try:
        return DigitalTwinScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Twin scenario file {scenario_path} is invalid: {exc}") from exc


def write_twin_result(path: Path | str, result: DigitalTwinResult) -> None:
    Path(path).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_twin_result(path: Path | str) -> DigitalTwinResult:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read twin result {result_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(f"Could not parse twin result {result_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Twin result file {result_path} must contain a JSON object")
    try:
        return DigitalTwinResult.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Twin result file {result_path} is invalid: {exc}") from exc
