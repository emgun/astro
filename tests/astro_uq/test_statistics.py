from __future__ import annotations

import numpy as np
import pytest

from astro_uq.models import (
    CaseObservation,
    OutcomeStatus,
    RequirementOutcome,
)
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
