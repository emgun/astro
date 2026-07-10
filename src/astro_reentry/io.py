from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_reentry.models import ReentryOptimizationResult, ReentryResult, ReentryScenario

ModelT = TypeVar("ModelT", bound=AstroModel)


def load_reentry_scenario(path: Path | str) -> ReentryScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read reentry scenario {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse reentry scenario {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Reentry scenario file {scenario_path} must contain a mapping")
    try:
        return ReentryScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Reentry scenario file {scenario_path} is invalid: {exc}"
        ) from exc


def load_reentry_result(path: Path | str) -> ReentryResult:
    return _load_json_model(path, ReentryResult, "reentry result")


def load_reentry_optimization_result(path: Path | str) -> ReentryOptimizationResult:
    return _load_json_model(path, ReentryOptimizationResult, "reentry optimization result")


def _load_json_model(  # noqa: UP047 - mypy 1.11 cannot parse PEP 695 syntax
    path: Path | str,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read {label} {result_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(f"Could not parse {label} {result_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"{label.capitalize()} file {result_path} must contain an object"
        )
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"{label.capitalize()} file {result_path} is invalid: {exc}"
        ) from exc


def format_reentry_summary(result: ReentryResult) -> str:
    lines = [
        f"Reentry: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Backend: {result.backend}",
        f"Samples: {len(result.samples)}",
        f"Termination: {result.metadata.get('termination_reason', 'unknown')}",
        f"Final altitude km: {result.samples[-1].altitude_km:.3f}",
        f"Final velocity km/s: {result.samples[-1].velocity_km_s:.3f}",
        f"Peak dynamic pressure Pa: {result.peaks.dynamic_pressure.value:.3f}",
        f"Peak deceleration g: {result.peaks.deceleration.value:.3f}",
        f"Peak heat rate W/m^2: {result.peaks.heat_rate.value:.3f}",
        f"Total heat load J/m^2: {result.peaks.total_heat_load_j_m2:.3f}",
    ]
    if result.target_miss is not None:
        lines.append(f"Target miss km: {result.target_miss.distance_km:.3f}")
    lines.append(
        "Limiting margin: "
        f"{result.margin_report.limiting_margin.name} = "
        f"{result.margin_report.limiting_margin.margin:.3f} "
        f"{result.margin_report.limiting_margin.unit} "
        f"({result.margin_report.limiting_margin.status})"
    )
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def format_reentry_optimization_summary(result: ReentryOptimizationResult) -> str:
    miss = result.reentry_result.target_miss
    lines = [
        f"Reentry optimization: {result.scenario_id}",
        f"Success: {result.success}",
        f"Iterations: {result.iterations}",
        f"Initial objective: {result.initial_objective:.6f}",
        f"Final objective: {result.final_objective:.6f}",
    ]
    if miss is not None:
        lines.append(f"Target miss km: {miss.distance_km:.3f}")
    lines.append(f"Message: {result.message}")
    return "\n".join(lines)
