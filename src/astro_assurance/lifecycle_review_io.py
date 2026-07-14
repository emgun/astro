from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from astro_assurance.lifecycle_review_models import MissionLifecycleReview
from astro_assurance.review_models import AssuranceReviewSeverity
from astro_core.errors import InvalidScenarioError


def write_mission_lifecycle_review(
    path: Path | str, review: MissionLifecycleReview
) -> None:
    _atomic_write(Path(path), review.model_dump_json(indent=2) + "\n")


def load_mission_lifecycle_review(path: Path | str) -> MissionLifecycleReview:
    review_path = Path(path)
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        return MissionLifecycleReview.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission lifecycle review {review_path}: {exc}"
        ) from exc


def write_mission_lifecycle_review_summary(path: Path | str, summary: str) -> None:
    _atomic_write(Path(path), summary)


def format_mission_lifecycle_review(review: MissionLifecycleReview) -> str:
    blockers = sum(
        finding.severity is AssuranceReviewSeverity.BLOCKER for finding in review.findings
    )
    warnings = sum(
        finding.severity is AssuranceReviewSeverity.WARNING for finding in review.findings
    )
    lines = [
        f"Lifecycle review: {review.review_id}",
        f"Scenario: {review.scenario_id}",
        f"Integrity verified: {str(review.integrity_verified).lower()}",
        f"Lifecycle passed: {str(review.lifecycle_passed).lower()}",
        f"Margin status: {review.margin_status.value}",
        f"Disposition: {review.disposition.value}",
        f"Findings: {len(review.findings)} ({blockers} blockers, {warnings} warnings)",
        f"Triage actions: {len(review.triage_actions)}",
        f"Claim boundary: {review.claim_boundary}",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write(output: Path, payload: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
