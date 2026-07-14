from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assistant.validators import validate_artifact
from astro_assurance.review import review_assurance_validation
from astro_assurance.review_comparison import (
    compare_assurance_validation_reviews,
    derive_assurance_review_comparison,
    verify_assurance_review_comparison,
)
from astro_assurance.review_io import (
    load_assurance_review_comparison,
    write_assurance_validation_review,
)
from astro_assurance.review_models import (
    AssuranceReviewDisposition,
    AssuranceReviewSeverity,
    AssuranceReviewTrend,
)
from astro_assurance.validation_io import (
    load_paired_assurance_validation_protocol,
    write_paired_assurance_validation_result,
)
from astro_assurance.validation_models import AssuranceCalibrationPromotionStatus
from astro_assurance.validation_runner import run_paired_assurance_validation
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError

PROTOCOL = Path("examples/assurance/paired_force_model_validation.yaml")


@pytest.fixture(scope="module")
def review_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    result = run_paired_assurance_validation(protocol)
    directory = tmp_path_factory.mktemp("assurance-comparison")
    result_path = directory / "paired.json"
    write_paired_assurance_validation_result(result_path, result)
    review = review_assurance_validation(result_path)
    baseline = directory / "baseline.json"
    candidate = directory / "candidate.json"
    write_assurance_validation_review(baseline, review)
    write_assurance_validation_review(candidate, review)
    return baseline, candidate


def test_identical_verified_reviews_produce_unchanged_comparison(
    review_paths: tuple[Path, Path],
) -> None:
    comparison = compare_assurance_validation_reviews(*review_paths)

    assert comparison.trend is AssuranceReviewTrend.UNCHANGED
    assert comparison.finding_changes == ()
    assert comparison.metric_changes == ()
    assert [item.source_finding_id for item in comparison.recommendations] == [
        "calibration_authority",
        "model_form_disposition",
        "claim_boundary",
    ]


def test_risk_vector_reports_improvement_without_scalar_score(
    review_paths: tuple[Path, Path],
) -> None:
    baseline = review_assurance_validation(
        json.loads(review_paths[0].read_text(encoding="utf-8"))["source_path"]
    )
    findings = tuple(
        finding.model_copy(update={"severity": AssuranceReviewSeverity.INFO})
        if finding.finding_id == "calibration_authority"
        else finding
        for finding in baseline.findings
    )
    candidate = baseline.model_copy(
        update={
            "calibration_promotion_status": (
                AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED
            ),
            "disposition": AssuranceReviewDisposition.DESIGN_REVIEW_READY,
            "findings": findings,
        }
    )
    comparison = derive_assurance_review_comparison(
        baseline,
        candidate,
        baseline_review_path="baseline.json",
        baseline_review_digest="0" * 64,
        candidate_review_path="candidate.json",
        candidate_review_digest="1" * 64,
    )

    assert comparison.trend is AssuranceReviewTrend.IMPROVED
    assert comparison.candidate_risk_vector < comparison.baseline_risk_vector


def test_removed_metric_is_a_mixed_change_when_risk_is_equal(
    review_paths: tuple[Path, Path],
) -> None:
    baseline = review_assurance_validation(
        json.loads(review_paths[0].read_text(encoding="utf-8"))["source_path"]
    )
    candidate = baseline.model_copy(update={"metric_shifts": baseline.metric_shifts[:-1]})

    comparison = derive_assurance_review_comparison(
        baseline,
        candidate,
        baseline_review_path="baseline.json",
        baseline_review_digest="0" * 64,
        candidate_review_path="candidate.json",
        candidate_review_digest="1" * 64,
    )

    assert comparison.trend is AssuranceReviewTrend.MIXED
    assert len(comparison.metric_changes) == 1
    assert comparison.metric_changes[0].candidate_median is None


def test_comparison_rejects_tampered_review(
    tmp_path: Path, review_paths: tuple[Path, Path]
) -> None:
    baseline, candidate = review_paths
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["disposition"] = "design_review_ready"
    tampered = tmp_path / "tampered-review.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="does not match"):
        compare_assurance_validation_reviews(baseline, tampered)


def test_compare_command_writes_loadable_artifact(
    tmp_path: Path, review_paths: tuple[Path, Path]
) -> None:
    baseline, candidate = review_paths
    output = tmp_path / "comparison.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-assurance-reviews",
            str(baseline),
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Trend: unchanged" in result.stdout
    assert load_assurance_review_comparison(output).trend is AssuranceReviewTrend.UNCHANGED
    assert verify_assurance_review_comparison(output).trend is AssuranceReviewTrend.UNCHANGED


def test_comparison_verifier_rejects_tampered_artifact(
    tmp_path: Path, review_paths: tuple[Path, Path]
) -> None:
    comparison = compare_assurance_validation_reviews(*review_paths)
    output = tmp_path / "comparison.json"
    payload = comparison.model_dump(mode="json")
    payload["trend"] = "regressed"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="does not match"):
        verify_assurance_review_comparison(output)
    artifact = WorkflowArtifact(
        path=str(output), kind=ArtifactKind.ASSURANCE_REVIEW_COMPARISON
    )
    assert not validate_artifact(artifact)


def test_compare_command_rejects_path_collisions(review_paths: tuple[Path, Path]) -> None:
    baseline, candidate = review_paths
    result = CliRunner().invoke(
        app,
        [
            "compare-assurance-reviews",
            str(baseline),
            str(candidate),
            "--output",
            str(baseline),
        ],
    )

    assert result.exit_code == 2
    assert "must all be different" in result.output


def test_comparison_api_rejects_same_path(review_paths: tuple[Path, Path]) -> None:
    with pytest.raises(InvalidScenarioError, match="must differ"):
        compare_assurance_validation_reviews(review_paths[0], review_paths[0])


def test_compare_command_rejects_hard_link_alias(
    tmp_path: Path, review_paths: tuple[Path, Path]
) -> None:
    baseline, candidate = review_paths
    summary_alias = tmp_path / "summary-alias.json"
    summary_alias.hardlink_to(baseline)
    original = baseline.read_bytes()

    result = CliRunner().invoke(
        app,
        [
            "compare-assurance-reviews",
            str(baseline),
            str(candidate),
            "--output",
            str(tmp_path / "comparison.json"),
            "--summary-output",
            str(summary_alias),
        ],
    )

    assert result.exit_code == 2
    assert baseline.read_bytes() == original
