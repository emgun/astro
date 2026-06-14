from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_core.models import Scenario


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
