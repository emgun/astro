from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_models import (
    ConstellationTwinResult,
    ConstellationTwinScenario,
)


def load_constellation_twin_scenario(
    path: Path | str,
) -> ConstellationTwinScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read constellation twin scenario {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse constellation twin scenario {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Constellation twin scenario file {scenario_path} must contain a mapping"
        )
    try:
        return ConstellationTwinScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Constellation twin scenario file {scenario_path} is invalid: {exc}"
        ) from exc


def write_constellation_twin_result(
    path: Path | str,
    result: ConstellationTwinResult,
) -> None:
    result_path = Path(path)
    try:
        result_path.write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not write constellation twin result {result_path}: {exc}"
        ) from exc


def load_constellation_twin_result(path: Path | str) -> ConstellationTwinResult:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read constellation twin result {result_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse constellation twin result {result_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Constellation twin result file {result_path} must contain a JSON object"
        )
    try:
        return ConstellationTwinResult.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Constellation twin result file {result_path} is invalid: {exc}"
        ) from exc


def format_constellation_summary(result: ConstellationTwinResult) -> str:
    analysis_window = result.metadata.get("analysis_window_s", {})
    if isinstance(analysis_window, dict):
        start_s = analysis_window.get("start_s", "unknown")
        end_s = analysis_window.get("end_s", "unknown")
    else:
        start_s = "unknown"
        end_s = "unknown"
    total_data_mbit = sum(
        summary.total_data_volume_mbit for summary in result.link_summaries
    )
    max_simultaneous = max(
        (
            summary.max_simultaneous_spacecraft
            for summary in result.access_summaries
        ),
        default=0,
    )
    lines = [
        f"Constellation twin: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Members: {len(result.members)}",
        f"Analysis window s: {start_s} to {end_s}",
        f"Fleet access summaries: {len(result.access_summaries)}",
        f"Total data volume Mbit: {total_data_mbit:.3f}",
        f"Max simultaneous spacecraft: {max_simultaneous}",
        (
            "Limiting fleet margin: "
            f"{result.fleet_margin_report.limiting_margin.name} = "
            f"{result.fleet_margin_report.limiting_margin.margin:.3f}"
        ),
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
