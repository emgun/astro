from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.lifecycle_review_models import (
    LifecycleReviewCategory,
    LifecycleReviewDisposition,
    LifecycleReviewFinding,
    LifecycleReviewInputReference,
    LifecycleReviewInputRole,
    LifecycleTriageAction,
    MissionLifecycleReview,
)
from astro_assurance.review_models import AssuranceReviewEvidence, AssuranceReviewSeverity
from astro_core.errors import InvalidScenarioError
from astro_mission.errors import MissionLifecycleError
from astro_mission.models import LifecycleStatus, MissionLifecycleResult, MissionLifecycleScenario
from astro_mission.runner import run_mission_lifecycle

_EXPECTED_PHASE_ORDER = ("launch", "operations", "digital_twin", "deorbit", "reentry")


@dataclass(frozen=True)
class _VerifiedLifecycleEvidence:
    result_path: Path
    result_bytes: bytes
    result: MissionLifecycleResult
    scenario_path: Path
    scenario_bytes: bytes
    scenario: MissionLifecycleScenario
    referenced_inputs: tuple[LifecycleReviewInputReference, ...]
    referenced_bytes: tuple[tuple[Path, bytes], ...]


def verify_mission_lifecycle_result(
    result_path: Path | str,
    scenario_path: Path | str,
) -> MissionLifecycleResult:
    return _verified_lifecycle_evidence(result_path, scenario_path).result


def review_mission_lifecycle(
    result_path: Path | str,
    scenario_path: Path | str,
) -> MissionLifecycleReview:
    evidence = _verified_lifecycle_evidence(result_path, scenario_path)
    return _derive_mission_lifecycle_review(evidence)


def verify_mission_lifecycle_review(review_path: Path | str) -> MissionLifecycleReview:
    from astro_assurance.lifecycle_review_io import load_mission_lifecycle_review

    review = load_mission_lifecycle_review(review_path)
    expected = review_mission_lifecycle(review.result_path, review.scenario_path)
    if review != expected:
        raise InvalidScenarioError(
            "mission lifecycle review does not match its verified lifecycle evidence"
        )
    return review


def _derive_mission_lifecycle_review(
    evidence: _VerifiedLifecycleEvidence,
) -> MissionLifecycleReview:
    result = evidence.result
    result_digest = sha256(evidence.result_bytes).hexdigest()
    findings = [
        _finding(
            "integrity_verified",
            AssuranceReviewSeverity.INFO,
            LifecycleReviewCategory.INTEGRITY,
            "The lifecycle result exactly matches a fresh local execution of its scenario.",
            ("result_digest", result_digest),
            "The review may use the verified lifecycle evidence.",
            "Preserve the scenario, result, and bound digests with downstream review.",
        )
    ]
    phase_order = tuple(entry.phase for entry in result.manifest.entries)
    manifest_complete = phase_order == _EXPECTED_PHASE_ORDER
    findings.append(
        _finding(
            "phase_manifest",
            (
                AssuranceReviewSeverity.INFO
                if manifest_complete
                else AssuranceReviewSeverity.BLOCKER
            ),
            LifecycleReviewCategory.MANIFEST,
            f"Lifecycle manifest phase order is {phase_order}.",
            ("phase_count", str(len(phase_order))),
            (
                "All required lifecycle phases are represented in order."
                if manifest_complete
                else "The lifecycle evidence is incomplete or phase-substituted."
            ),
            (
                "Retain the verified phase manifest."
                if manifest_complete
                else "Regenerate the lifecycle result with the fixed five-phase workflow."
            ),
        )
    )
    failed_checks = [check for check in result.continuity_report.checks if not check.passed]
    if not failed_checks:
        findings.append(
            _finding(
                "continuity_all_passed",
                AssuranceReviewSeverity.INFO,
                LifecycleReviewCategory.CONTINUITY,
                f"All {len(result.continuity_report.checks)} continuity checks passed.",
                ("check_count", str(len(result.continuity_report.checks))),
                "No lifecycle state, epoch, or mass discontinuity is reported.",
                "Retain continuity evidence with the review.",
            )
        )
    for check in failed_checks:
        findings.append(
            _finding(
                f"continuity_{_id_component(check.name)}",
                AssuranceReviewSeverity.BLOCKER,
                LifecycleReviewCategory.CONTINUITY,
                f"Continuity check {check.name} failed: "
                f"{check.error} > {check.tolerance} {check.unit}.",
                ("phase_pair", f"{check.upstream_phase}->{check.downstream_phase}"),
                "Downstream lifecycle evidence cannot be treated as continuous with its source.",
                "Resolve the handoff mismatch and regenerate all downstream phases.",
            )
        )
    unresolved_margins = [
        margin
        for margin in result.margin_report.margins
        if margin.status is not LifecycleStatus.PASS
    ]
    if not unresolved_margins:
        limiting = result.margin_report.limiting_margin
        findings.append(
            _finding(
                "margins_all_passed",
                AssuranceReviewSeverity.INFO,
                LifecycleReviewCategory.MARGIN,
                f"All {len(result.margin_report.margins)} lifecycle margins passed.",
                ("limiting_margin", f"{limiting.phase}/{limiting.name}"),
                "No typed lifecycle requirement is warning or failed in this deterministic case.",
                "Retain units, thresholds, and the limiting-margin identity in downstream review.",
            )
        )
    for margin in unresolved_margins:
        severity = (
            AssuranceReviewSeverity.BLOCKER
            if margin.status is LifecycleStatus.FAIL
            else AssuranceReviewSeverity.WARNING
        )
        findings.append(
            _finding(
                f"margin_{_id_component(margin.phase)}_{_id_component(margin.name)}",
                severity,
                LifecycleReviewCategory.MARGIN,
                f"{margin.phase}/{margin.name} is {margin.status.value} with margin "
                f"{margin.margin} {margin.unit}.",
                ("threshold", f"{margin.threshold} {margin.unit}"),
                "The typed lifecycle requirement needs engineering review.",
                "Review the authoritative phase product and requirement before changing "
                "the design.",
            )
        )
    for margin in result.margin_report.margins:
        if margin.unit != "native":
            continue
        findings.append(
            _finding(
                f"margin_unit_{_id_component(margin.phase)}_{_id_component(margin.name)}",
                AssuranceReviewSeverity.WARNING,
                LifecycleReviewCategory.MARGIN,
                f"{margin.phase}/{margin.name} uses the non-specific unit label 'native'.",
                ("margin_value", str(margin.margin)),
                "The margin cannot be interpreted or compared with a unit-specific requirement.",
                "Publish the authoritative physical unit before using this margin in design "
                "review.",
            )
        )
    for warning in sorted(set(result.warnings)):
        warning_digest = sha256(warning.encode("utf-8")).hexdigest()[:12]
        findings.append(
            _finding(
                f"evidence_boundary_{warning_digest}",
                AssuranceReviewSeverity.INFO,
                LifecycleReviewCategory.EVIDENCE_BOUNDARY,
                warning,
                ("source", "mission_lifecycle_result.warnings"),
                "This caveat limits interpretation but is not a diagnosed anomaly.",
                "Carry the caveat forward without inferring severity or root cause from prose.",
            )
        )
    findings.append(
        _finding(
            "claim_boundary",
            AssuranceReviewSeverity.INFO,
            LifecycleReviewCategory.CLAIM_BOUNDARY,
            "The lifecycle remains deterministic design-screening evidence.",
            ("source_workflow", result.workflow),
            "The review cannot establish probability, causality, certification, or flight "
            "authority.",
            "Keep every explanation and decision within the recorded evidence boundary.",
        )
    )
    unresolved = [
        finding
        for finding in findings
        if finding.severity in {AssuranceReviewSeverity.BLOCKER, AssuranceReviewSeverity.WARNING}
    ]
    triage = tuple(
        LifecycleTriageAction(
            action_id=f"triage_{finding.finding_id}",
            priority=finding.severity,
            source_finding_id=finding.finding_id,
            action=finding.required_action,
        )
        for finding in unresolved
    )
    return MissionLifecycleReview(
        review_id=f"{_id_component(result.scenario_id)}-lifecycle-review-v1",
        result_path=str(evidence.result_path),
        result_digest=result_digest,
        scenario_path=str(evidence.scenario_path),
        scenario_digest=sha256(evidence.scenario_bytes).hexdigest(),
        referenced_inputs=evidence.referenced_inputs,
        scenario_id=result.scenario_id,
        lifecycle_passed=result.passed,
        continuity_all_passed=result.continuity_report.all_passed,
        margin_status=result.margin_report.overall_status,
        phase_order=phase_order,
        findings=tuple(findings),
        triage_actions=triage,
        disposition=(
            LifecycleReviewDisposition.ADDITIONAL_REVIEW_REQUIRED
            if unresolved
            else LifecycleReviewDisposition.DESIGN_REVIEW_READY
        ),
    )


def _verified_lifecycle_evidence(
    result_path: Path | str,
    scenario_path: Path | str,
) -> _VerifiedLifecycleEvidence:
    result_file = Path(result_path).resolve()
    scenario_file = Path(scenario_path).resolve()
    if result_file == scenario_file:
        raise InvalidScenarioError("lifecycle result and scenario paths must differ")
    result_bytes = result_file.read_bytes()
    scenario_bytes = scenario_file.read_bytes()
    result = _parse_result_bytes(result_bytes, result_file)
    scenario = _parse_scenario_bytes(scenario_bytes, scenario_file)
    if scenario.launch_backend != "local" or scenario.reentry_backend != "local":
        raise InvalidScenarioError("lifecycle review v1 supports local launch and reentry only")
    referenced_bytes = _snapshot_referenced_inputs(scenario)
    referenced_inputs = tuple(
        LifecycleReviewInputReference(
            role=role,
            path=str(path),
            digest=sha256(payload).hexdigest(),
        )
        for role, path, payload in referenced_bytes
    )
    try:
        expected = _run_captured_lifecycle(scenario, referenced_bytes)
    except MissionLifecycleError as exc:
        phase = f" in phase {exc.lifecycle_phase}" if exc.lifecycle_phase else ""
        raise InvalidScenarioError(
            f"could not reproduce mission lifecycle result{phase}: {exc}"
        ) from exc
    changed = (
        result_file.read_bytes() != result_bytes
        or scenario_file.read_bytes() != scenario_bytes
        or any(path.read_bytes() != payload for _, path, payload in referenced_bytes)
    )
    if changed:
        raise InvalidScenarioError("lifecycle evidence changed during verification")
    if result.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise InvalidScenarioError("mission lifecycle result does not match fresh local execution")
    return _VerifiedLifecycleEvidence(
        result_path=result_file,
        result_bytes=result_bytes,
        result=result,
        scenario_path=scenario_file,
        scenario_bytes=scenario_bytes,
        scenario=scenario,
        referenced_inputs=referenced_inputs,
        referenced_bytes=tuple((path, payload) for _, path, payload in referenced_bytes),
    )


def _parse_result_bytes(payload: bytes, path: Path) -> MissionLifecycleResult:
    try:
        raw: Any = json.loads(payload.decode("utf-8"))
        return MissionLifecycleResult.model_validate(raw)
    except (UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not parse mission lifecycle result {path}: {exc}"
        ) from exc


def _parse_scenario_bytes(payload: bytes, path: Path) -> MissionLifecycleScenario:
    try:
        raw: Any = yaml.safe_load(payload.decode("utf-8"))
        return MissionLifecycleScenario.model_validate(raw)
    except (UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not parse mission lifecycle scenario {path}: {exc}"
        ) from exc


def _snapshot_referenced_inputs(
    scenario: MissionLifecycleScenario,
) -> tuple[tuple[LifecycleReviewInputRole, Path, bytes], ...]:
    paths: tuple[tuple[LifecycleReviewInputRole, Path], ...] = (
        ("launch_scenario", Path(scenario.launch_scenario).resolve()),
        ("twin_scenario", Path(scenario.twin_scenario).resolve()),
        ("reentry_scenario", Path(scenario.reentry_scenario).resolve()),
    )
    return tuple((role, path, path.read_bytes()) for role, path in paths)


def _run_captured_lifecycle(
    scenario: MissionLifecycleScenario,
    referenced_bytes: tuple[tuple[LifecycleReviewInputRole, Path, bytes], ...],
) -> MissionLifecycleResult:
    with tempfile.TemporaryDirectory(prefix="astro-lifecycle-review-") as directory:
        staging = Path(directory)
        replacements: dict[str, str] = {}
        for role, source_path, payload in referenced_bytes:
            staged = staging / f"{role}{source_path.suffix}"
            staged.write_bytes(payload)
            replacements[role] = str(staged)
        captured_scenario = scenario.model_copy(update=replacements)
        return run_mission_lifecycle(captured_scenario)


def _finding(
    finding_id: str,
    severity: AssuranceReviewSeverity,
    category: LifecycleReviewCategory,
    statement: str,
    evidence: tuple[str, str],
    implication: str,
    required_action: str,
) -> LifecycleReviewFinding:
    return LifecycleReviewFinding(
        finding_id=finding_id,
        severity=severity,
        category=category,
        statement=statement,
        evidence=(AssuranceReviewEvidence(key=evidence[0], value=evidence[1]),),
        implication=implication,
        required_action=required_action,
    )


def _id_component(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"value_{normalized}"
    return normalized
