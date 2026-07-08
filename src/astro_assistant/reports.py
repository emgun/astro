from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from astro_assistant.models import ArtifactKind, WorkflowArtifact, WorkflowTrace


class WorkflowReportArtifact(BaseModel):
    step_id: str
    path: str
    kind: ArtifactKind
    required: bool
    validation_passed: bool | None = None


class WorkflowReport(BaseModel):
    workflow_id: str
    title: str
    scenario_path: str | None = None
    dry_run: bool
    verification_passed: bool
    step_count: int
    executed_step_count: int
    artifacts: list[WorkflowReportArtifact] = Field(default_factory=list)
    measurement_count: int | None = None
    tdm_line_count: int | None = None
    estimate_converged: bool | None = None
    estimate_iterations: int | None = None
    estimate_rms: float | None = None
    jacobian_rank: int | None = None
    residual_count: int | None = None
    max_abs_residual: float | None = None
    warnings: list[str] = Field(default_factory=list)


def build_workflow_report(trace: WorkflowTrace) -> WorkflowReport:
    warnings = list(trace.warnings)
    artifact_validation = _artifact_validation_by_step(trace)
    report = WorkflowReport(
        workflow_id=trace.plan.plan_id,
        title=trace.plan.title,
        scenario_path=_scenario_path(trace),
        dry_run=trace.dry_run,
        verification_passed=trace.verification.passed,
        step_count=len(trace.plan.steps),
        executed_step_count=len(trace.results),
        artifacts=[
            WorkflowReportArtifact(
                step_id=step.step_id,
                path=artifact.path,
                kind=artifact.kind,
                required=artifact.required,
                validation_passed=artifact_validation.get((step.step_id, artifact.path)),
            )
            for step in trace.plan.steps
            for artifact in step.outputs
        ],
        warnings=warnings,
    )

    if not trace.dry_run:
        for step in trace.plan.steps:
            for artifact in step.outputs:
                _add_artifact_metrics(report, artifact)

    return report


def format_workflow_report(report: WorkflowReport) -> str:
    validated_artifact_count = sum(
        1 for artifact in report.artifacts if artifact.validation_passed is True
    )
    mode = "dry-run" if report.dry_run else "executed"
    verification = "passed" if report.verification_passed else "failed"
    lines = [
        f"Workflow: {report.workflow_id}",
        f"Title: {report.title}",
        f"Scenario: {report.scenario_path or 'unavailable'}",
        f"Mode: {mode}",
        f"Verification: {verification}",
        f"Steps: {report.executed_step_count}/{report.step_count} executed",
        f"Artifacts: {len(report.artifacts)} declared, {validated_artifact_count} validated",
        f"Measurements: {_format_value(report.measurement_count)}",
        f"TDM lines: {_format_value(report.tdm_line_count)}",
        f"Estimate: {_format_estimate(report)}",
        f"Jacobian rank: {_format_value(report.jacobian_rank)}",
        f"Residuals: {_format_residuals(report)}",
    ]
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


def _artifact_validation_by_step(trace: WorkflowTrace) -> dict[tuple[str, str], bool]:
    validation: dict[tuple[str, str], bool] = {}
    for result in trace.results:
        for artifact in result.artifacts:
            validation[(result.step_id, artifact.path)] = result.validation_passed
    return validation


def _scenario_path(trace: WorkflowTrace) -> str | None:
    for step in trace.plan.steps:
        scenario_path = step.inputs.get("scenario_path")
        if isinstance(scenario_path, str) and scenario_path:
            return scenario_path
    return None


def _add_artifact_metrics(report: WorkflowReport, artifact: WorkflowArtifact) -> None:
    path = Path(artifact.path)
    if not path.exists():
        if artifact.required and not report.dry_run:
            report.warnings.append(f"missing artifact: {artifact.path}")
        return

    if artifact.kind == ArtifactKind.MEASUREMENTS_JSON:
        payload = _read_json(path, report)
        if isinstance(payload, dict) and isinstance(payload.get("measurements"), list):
            report.measurement_count = len(payload["measurements"])
    elif artifact.kind == ArtifactKind.MEASUREMENTS_TDM:
        report.tdm_line_count = _count_lines(path, report)
    elif artifact.kind == ArtifactKind.ESTIMATE_JSON:
        payload = _read_json(path, report)
        if isinstance(payload, dict):
            report.estimate_converged = _optional_bool(payload.get("converged"))
            report.estimate_iterations = _optional_int(payload.get("iterations"))
            report.estimate_rms = _optional_float(payload.get("rms"))
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                report.jacobian_rank = _optional_int(metadata.get("jacobian_rank"))
                report.residual_count = _optional_int(metadata.get("residual_count"))
                report.max_abs_residual = _optional_float(metadata.get("max_abs_residual"))


def _read_json(path: Path, report: WorkflowReport) -> object | None:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.warnings.append(f"could not parse artifact {path}: {exc}")
        return None


def _count_lines(path: Path, report: WorkflowReport) -> int | None:
    try:
        return sum(1 for _ in path.open(encoding="utf-8"))
    except OSError as exc:
        report.warnings.append(f"could not read artifact {path}: {exc}")
        return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _format_value(value: object | None) -> str:
    return "unavailable" if value is None else str(value)


def _format_estimate(report: WorkflowReport) -> str:
    if (
        report.estimate_converged is None
        and report.estimate_iterations is None
        and report.estimate_rms is None
    ):
        return "unavailable"
    status = "unknown"
    if report.estimate_converged is True:
        status = "converged"
    elif report.estimate_converged is False:
        status = "not converged"
    parts = [status]
    if report.estimate_iterations is not None:
        parts.append(f"iterations={report.estimate_iterations}")
    if report.estimate_rms is not None:
        parts.append(f"rms={report.estimate_rms}")
    return ", ".join(parts)


def _format_residuals(report: WorkflowReport) -> str:
    if report.residual_count is None and report.max_abs_residual is None:
        return "unavailable"
    parts = []
    if report.residual_count is not None:
        parts.append(f"count={report.residual_count}")
    if report.max_abs_residual is not None:
        parts.append(f"max_abs={report.max_abs_residual}")
    return ", ".join(parts)
