from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_mission.models import MissionLifecycleResult, MissionLifecycleScenario


def load_mission_lifecycle_scenario(path: Path | str) -> MissionLifecycleScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read mission lifecycle scenario {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse mission lifecycle scenario {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Mission lifecycle scenario file {scenario_path} must contain a mapping"
        )
    try:
        return MissionLifecycleScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Mission lifecycle scenario file {scenario_path} is invalid: {exc}"
        ) from exc


def write_mission_lifecycle_result(path: Path | str, result: MissionLifecycleResult) -> None:
    _write_text(Path(path), result.model_dump_json(indent=2) + "\n")


def load_mission_lifecycle_result(path: Path | str) -> MissionLifecycleResult:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read mission lifecycle result {result_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse mission lifecycle result {result_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Mission lifecycle result file {result_path} must contain a JSON object"
        )
    try:
        return MissionLifecycleResult.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Mission lifecycle result file {result_path} is invalid: {exc}"
        ) from exc


def write_mission_artifact_bundle(
    directory: Path | str,
    result: MissionLifecycleResult,
) -> None:
    artifact_directory = Path(directory)
    try:
        artifact_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidScenarioError(
            f"Could not create mission artifact directory {artifact_directory}: {exc}"
        ) from exc
    json_artifacts = {
        "launch.json": result.launch_trajectory,
        "operations-trajectory.json": result.operations_trajectory,
        "digital-twin.json": result.digital_twin,
        "deorbit-trajectory.json": result.deorbit_trajectory,
        "reentry-result.json": result.reentry_result,
        "manifest.json": result.manifest,
    }
    yaml_artifacts = {
        "orbit-scenario.yaml": result.orbit_scenario,
        "deorbit-scenario.yaml": result.deorbit_scenario,
        "reentry-scenario.yaml": result.reentry_scenario,
    }
    for name, model in json_artifacts.items():
        _write_text(artifact_directory / name, model.model_dump_json(indent=2) + "\n")
    for name, model in yaml_artifacts.items():
        payload = model.model_dump(mode="json")
        _write_text(artifact_directory / name, yaml.safe_dump(payload, sort_keys=False))


def format_mission_lifecycle_summary(result: MissionLifecycleResult) -> str:
    limiting = result.margin_report.limiting_margin
    lines = [
        f"Mission lifecycle: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Status: {'pass' if result.passed else 'fail'}",
        f"Continuity checks: {len(result.continuity_report.checks)} passed",
        f"Operations samples: {len(result.operations_trajectory.samples)}",
        f"Entry interface altitude km: {result.reentry_scenario.initial_state.altitude_km:.3f}",
        "Reentry termination: "
        f"{result.reentry_result.metadata.get('termination_reason', 'unknown')}",
        (
            f"Limiting margin: {limiting.phase}/{limiting.name} = "
            f"{limiting.margin:.3f} {limiting.unit} ({limiting.status})"
        ),
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not write mission artifact {path}: {exc}") from exc
