from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from astro_assurance.review_models import AssuranceValidationReview
from astro_core.errors import InvalidScenarioError


def write_assurance_validation_review(
    path: Path | str, review: AssuranceValidationReview
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = review.model_dump_json(indent=2) + "\n"
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


def load_assurance_validation_review(path: Path | str) -> AssuranceValidationReview:
    review_path = Path(path)
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        return AssuranceValidationReview.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load assurance validation review {review_path}: {exc}"
        ) from exc


def format_assurance_validation_review(review: AssuranceValidationReview) -> str:
    blockers = sum(finding.severity.value == "blocker" for finding in review.findings)
    warnings = sum(finding.severity.value == "warning" for finding in review.findings)
    lines = [
        f"Review: {review.review_id}",
        f"Protocol: {review.protocol_id}",
        f"Integrity verified: {str(review.integrity_verified).lower()}",
        f"Calibration: {review.calibration_promotion_status.value}",
        f"Disposition: {review.disposition.value}",
        f"Findings: {len(review.findings)} ({blockers} blockers, {warnings} warnings)",
        f"Claim boundary: {review.claim_boundary}",
    ]
    return "\n".join(lines) + "\n"
