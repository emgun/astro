from __future__ import annotations

import numpy as np
import pytest

from astro_uq.models import (
    CaseObservation,
    EvaluationTiming,
    OutcomeStatus,
    RequirementOutcome,
)
from astro_uq.profiling import summarize_case_timings
from astro_uq.statistics import effective_sample_size, summarize_campaign, wilson_interval


def _observation(sample_id: str, value: float, passed: bool) -> CaseObservation:
    return CaseObservation(
        sample_id=sample_id,
        outcome_status=OutcomeStatus.SUCCESS,
        metric_values={"margin": value},
        requirements=(RequirementOutcome(requirement_id="positive", passed=passed),),
        evaluator_id="fixture",
        claim_boundary="test_fixture",
    )


def test_weighted_statistics_match_independent_fixture() -> None:
    observations = (
        _observation("a", 0.0, False),
        _observation("b", 10.0, True),
    )
    statistics = summarize_campaign(
        requested_samples=2,
        observations=observations,
        weights={"a": 0.25, "b": 0.75},
        quantiles=(0.5,),
    )

    summary = statistics.metrics[0]
    assert summary.mean == pytest.approx(7.5)
    assert summary.variance == pytest.approx(18.75)
    assert summary.effective_sample_size == pytest.approx(1.6)
    assert summary.quantiles["q0.5"] == 10.0
    assert statistics.requirement_probabilities["positive"] == 0.75


def test_failures_remain_in_outcome_denominator_evidence() -> None:
    observations = (
        _observation("a", 1.0, True),
        CaseObservation(
            sample_id="b",
            outcome_status=OutcomeStatus.EXECUTION_FAILURE,
            evaluator_id="fixture",
            claim_boundary="test_fixture",
        ),
    )

    statistics = summarize_campaign(
        requested_samples=2,
        observations=observations,
        weights={"a": 0.5, "b": 0.5},
        requirement_ids=("positive",),
    )

    assert statistics.completed_samples == 2
    assert statistics.outcome_counts == {"success": 1, "execution_failure": 1}
    assert statistics.requirement_probabilities == {"positive": 0.5}
    assert statistics.requirement_denominator_policy == "all_completed_cases"


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(80, 100)

    assert lower < 0.8 < upper


def test_effective_sample_size_for_equal_weights() -> None:
    assert effective_sample_size(np.ones(4)) == 4.0


def test_timing_profile_accounts_for_every_instrumented_phase() -> None:
    observations = tuple(
        CaseObservation(
            sample_id=sample_id,
            outcome_status=OutcomeStatus.SUCCESS,
            evaluator_id="fixture",
            claim_boundary="machine_scoped_test",
            evaluation_timing=EvaluationTiming(
                setup_s=setup,
                evaluation_s=evaluation,
                metric_extraction_s=extraction,
                serialization_s=serialization,
                total_s=total,
            ),
        )
        for sample_id, setup, evaluation, extraction, serialization, total in (
            ("a", 1.0, 4.0, 1.0, 1.0, 8.0),
            ("b", 1.0, 6.0, 1.0, 1.0, 10.0),
        )
    )

    profile = summarize_case_timings(observations)

    assert profile.case_count == 2
    assert profile.fully_instrumented_case_count == 2
    assert profile.evaluation.median_s == 5.0
    assert profile.evaluation.median_absolute_deviation_s == 1.0
    assert profile.unattributed.total_s == 2.0
    assert profile.evaluation_share_of_instrumented_time == pytest.approx(10.0 / 18.0)
    assert profile.accounted_share_of_instrumented_time == pytest.approx(16.0 / 18.0)


def test_legacy_timing_remains_explicitly_uninstrumented() -> None:
    observation = CaseObservation.model_validate(
        {
            "sample_id": "legacy",
            "outcome_status": "success",
            "evaluator_id": "fixture",
            "claim_boundary": "legacy_test",
            "evaluation_timing": {
                "setup_s": 0.0,
                "evaluation_s": 1.0,
                "serialization_s": 0.0,
                "total_s": 1.0,
            },
        }
    )

    profile = summarize_case_timings((observation,))

    assert profile.fully_instrumented_case_count == 0
    assert profile.metric_extraction is None
    assert profile.evaluation_share_of_instrumented_time is None
    assert profile.accounted_share_of_instrumented_time is None


def test_untimed_case_remains_in_profile_completeness_denominator() -> None:
    timed = CaseObservation(
        sample_id="timed",
        outcome_status=OutcomeStatus.SUCCESS,
        evaluator_id="fixture",
        claim_boundary="machine_scoped_test",
        evaluation_timing=EvaluationTiming(
            evaluation_s=1.0,
            metric_extraction_s=0.1,
            total_s=1.1,
        ),
    )
    untimed = CaseObservation(
        sample_id="untimed",
        outcome_status=OutcomeStatus.SUCCESS,
        evaluator_id="fixture",
        claim_boundary="machine_scoped_test",
    )

    profile = summarize_case_timings((timed, untimed))

    assert profile.case_count == 2
    assert profile.fully_instrumented_case_count == 1
    assert profile.evaluation.count == 1
    assert profile.evaluation_share_of_instrumented_time is None
    assert profile.accounted_share_of_instrumented_time is None


def test_timing_profile_rejects_failed_cases() -> None:
    failed = CaseObservation(
        sample_id="failed",
        outcome_status=OutcomeStatus.EXECUTION_FAILURE,
        evaluator_id="fixture",
        claim_boundary="machine_scoped_test",
        evaluation_timing=EvaluationTiming(total_s=0.1, metric_extraction_s=0.0),
    )

    with pytest.raises(ValueError, match="every campaign case to succeed"):
        summarize_case_timings((failed,))
