from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_launch.models import LaunchScenario


def load_launch_scenario(path: Path | str) -> LaunchScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read launch scenario file {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse launch scenario file {scenario_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Launch scenario file {scenario_path} must contain a mapping")

    try:
        return LaunchScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Launch scenario file {scenario_path} is invalid: {exc}"
        ) from exc
