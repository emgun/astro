from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_launch.models import LaunchScenario, LaunchTrajectory, TunedLaunchReport


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


def load_launch_trajectory(path: Path | str) -> LaunchTrajectory:
    trajectory_path = Path(path)
    try:
        raw: Any = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read launch trajectory file {trajectory_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse launch trajectory file {trajectory_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Launch trajectory file {trajectory_path} must contain a JSON object"
        )

    try:
        return LaunchTrajectory.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Launch trajectory file {trajectory_path} is invalid: {exc}"
        ) from exc


def load_tuned_launch_report(path: Path | str) -> TunedLaunchReport:
    report_path = Path(path)
    try:
        raw: Any = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read tuned launch report file {report_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse tuned launch report file {report_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Tuned launch report file {report_path} must contain a JSON object"
        )

    try:
        return TunedLaunchReport.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Tuned launch report file {report_path} is invalid: {exc}"
        ) from exc
