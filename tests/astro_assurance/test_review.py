from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from astro_assurance.review import review_assurance_validation
from astro_assurance.review_io import (
    load_assurance_validation_review,
    write_assurance_validation_review,
)
from astro_assurance.review_models import (
    AssuranceReviewDisposition,
    AssuranceReviewSeverity,
)
from astro_assurance.validation_io import (
    load_paired_assurance_validation_protocol,
    write_paired_assurance_validation_result,
)
from astro_assurance.validation_runner import run_paired_assurance_validation
from astro_cli.main import app

PROTOCOL = Path("examples/assurance/paired_force_model_validation.yaml")


@pytest.fixture(scope="module")
def paired_result_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    result = run_paired_assurance_validation(protocol)
    path = tmp_path_factory.mktemp("assurance-review") / "paired.json"
    write_paired_assurance_validation_result(path, result)
    return path


def test_review_is_deterministic_and_preserves_claim_boundaries(
    paired_result_path: Path,
) -> None:
    first = review_assurance_validation(paired_result_path)
    second = review_assurance_validation(paired_result_path)

    assert first == second
    assert first.integrity_verified
    assert first.disposition is AssuranceReviewDisposition.ADDITIONAL_EVIDENCE_REQUIRED
    assert first.calibration_promotion_status.value == "illustrative"
    assert [shift.metric for shift in first.metric_shifts] == sorted(
        shift.metric for shift in first.metric_shifts
    )
    findings = {finding.finding_id: finding for finding in first.findings}
    assert findings["calibration_authority"].severity is AssuranceReviewSeverity.BLOCKER
    assert findings["model_form_disposition"].severity is AssuranceReviewSeverity.WARNING
    assert findings["metric_shift_truth_recovery_position_error_km"].evidence[0].value == (
        "truth_recovery_position_error_km"
    )
    assert findings["claim_boundary"].required_action.endswith(
        "human or model-generated explanation."
    )


def test_review_output_is_byte_stable_and_loadable(
    tmp_path: Path, paired_result_path: Path
) -> None:
    review = review_assurance_validation(paired_result_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_assurance_validation_review(first, review)
    write_assurance_validation_review(second, review)

    assert first.read_bytes() == second.read_bytes()
    assert load_assurance_validation_review(first) == review


def test_review_command_writes_verified_json_and_summary(
    tmp_path: Path, paired_result_path: Path
) -> None:
    output = tmp_path / "review.json"
    summary = tmp_path / "review.txt"
    result = CliRunner().invoke(
        app,
        [
            "review-assurance-validation",
            str(paired_result_path),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    assert "Disposition: additional_evidence_required" in result.stdout
    assert load_assurance_validation_review(output).source_path == str(
        paired_result_path.resolve()
    )
    assert summary.read_text(encoding="utf-8") == result.stdout


def test_review_command_rejects_tampering_without_output(
    tmp_path: Path, paired_result_path: Path
) -> None:
    payload = json.loads(paired_result_path.read_text(encoding="utf-8"))
    payload["summary"]["matched_passed"] = 0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "must-not-exist.json"

    result = CliRunner().invoke(
        app,
        ["review-assurance-validation", str(tampered), "--output", str(output)],
    )

    assert result.exit_code == 2
    assert not output.exists()


@pytest.mark.parametrize("collision", ["output", "summary"])
def test_review_command_rejects_path_collisions(
    paired_result_path: Path, collision: str
) -> None:
    args = [
        "review-assurance-validation",
        str(paired_result_path),
        "--output",
        (
            str(paired_result_path)
            if collision == "output"
            else str(paired_result_path.parent / "r.json")
        ),
    ]
    if collision == "summary":
        args.extend(["--summary-output", str(paired_result_path)])
    result = CliRunner().invoke(
        app,
        args,
    )

    assert result.exit_code == 2
    assert "must be different paths" in result.output
