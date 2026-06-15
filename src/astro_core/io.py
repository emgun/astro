from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_core.models import Scenario, Trajectory


def load_scenario(path: Path | str) -> Scenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read scenario file {scenario_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(f"Could not parse scenario file {scenario_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Scenario file {scenario_path} must contain a mapping")

    try:
        return Scenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Scenario file {scenario_path} is invalid: {exc}") from exc


def load_trajectory(path: Path | str) -> Trajectory:
    trajectory_path = Path(path)
    try:
        raw: Any = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read trajectory file {trajectory_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse trajectory file {trajectory_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Trajectory file {trajectory_path} must contain a JSON object")

    try:
        return Trajectory.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Trajectory file {trajectory_path} is invalid: {exc}") from exc
