from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from astro_uq.models import (
    ConfidenceIntervalStopping,
    FixedCountStopping,
    MetricStabilityStopping,
    StoppingRule,
)
from astro_uq.stopping import (
    ConfidenceIntervalRule,
    MetricStabilityRule,
    confidence_interval_decision,
    fixed_count_decision,
    metric_stability_decision,
)


def test_fixed_count_stops_at_requested_count() -> None:
    assert fixed_count_decision(8, 8).stop is True
    assert fixed_count_decision(7, 8).stop is False


def test_fixed_count_contract_remains_the_default_shape() -> None:
    assert FixedCountStopping().model_dump(mode="json") == {"kind": "fixed_count"}


def test_confidence_interval_requires_minimum_effective_sample_size() -> None:
    rule = ConfidenceIntervalRule(
        target_half_width=0.1,
        minimum_samples=20,
        maximum_samples=100,
    )

    decision = confidence_interval_decision(
        successes=19,
        completed=20,
        effective_sample_size=5.0,
        rule=rule,
    )

    assert decision.stop is False
    assert decision.reason == "effective_sample_size_insufficient"


def test_confidence_interval_matches_analytical_wilson_half_width() -> None:
    rule = ConfidenceIntervalRule(
        target_half_width=0.097,
        minimum_samples=20,
        maximum_samples=101,
    )

    decision = confidence_interval_decision(
        successes=50,
        completed=100,
        effective_sample_size=100.0,
        rule=rule,
    )

    assert decision.stop is True
    assert decision.reason == "confidence_interval_converged"


def test_confidence_interval_only_checks_at_batch_boundaries() -> None:
    rule = ConfidenceIntervalRule(
        target_half_width=0.5,
        minimum_samples=10,
        maximum_samples=30,
        batch_size=5,
    )

    decision = confidence_interval_decision(
        successes=5,
        completed=11,
        effective_sample_size=11.0,
        rule=rule,
    )

    assert decision == decision.__class__(False, "batch_boundary_not_reached")


def test_metric_stability_uses_recent_window() -> None:
    rule = MetricStabilityRule(
        absolute_tolerance=0.01,
        minimum_samples=20,
        maximum_samples=100,
        window=3,
    )

    decision = metric_stability_decision((1.0, 1.004, 1.006), completed=20, rule=rule)

    assert decision.stop is True
    assert decision.reason == "metric_stability_converged"


def test_metric_stability_requires_effective_sample_size() -> None:
    rule = MetricStabilityRule(
        absolute_tolerance=0.01,
        minimum_samples=20,
        maximum_samples=100,
        minimum_effective_sample_size=15.0,
    )

    decision = metric_stability_decision(
        (1.0, 1.001, 1.002),
        completed=20,
        effective_sample_size=14.9,
        rule=rule,
    )

    assert decision.stop is False
    assert decision.reason == "effective_sample_size_insufficient"


@pytest.mark.parametrize(
    "rule",
    [
        ConfidenceIntervalRule(0.01, 20, 20),
        MetricStabilityRule(0.01, 20, 20),
    ],
)
def test_maximum_samples_forces_a_non_convergence_stop(
    rule: ConfidenceIntervalRule | MetricStabilityRule,
) -> None:
    if isinstance(rule, ConfidenceIntervalRule):
        decision = confidence_interval_decision(
            successes=10,
            completed=20,
            effective_sample_size=1.0,
            rule=rule,
        )
    else:
        decision = metric_stability_decision(
            (1.0,),
            completed=20,
            effective_sample_size=1.0,
            rule=rule,
        )

    assert decision.stop is True
    assert decision.reason == "maximum_samples_reached"


def test_stopping_models_validate_bounds_and_targets() -> None:
    ci_rule = ConfidenceIntervalStopping(
        requirement_id="mission_success",
        target_half_width=0.05,
        minimum_samples=20,
        maximum_samples=100,
        minimum_effective_sample_size=18.0,
        batch_size=5,
    )
    stability_rule = MetricStabilityStopping(
        metric_id="fuel_margin",
        absolute_tolerance=0.01,
        minimum_samples=20,
        maximum_samples=100,
    )

    assert ci_rule.kind == "ci_half_width"
    assert stability_rule.kind == "metric_stability"

    with pytest.raises(ValidationError, match="minimum_samples <= maximum_samples"):
        MetricStabilityStopping(
            metric_id="fuel_margin",
            absolute_tolerance=0.01,
            minimum_samples=21,
            maximum_samples=20,
        )


def test_stopping_rule_discriminator_parses_ci_contract() -> None:
    rule = TypeAdapter(StoppingRule).validate_python(
        {
            "kind": "ci_half_width",
            "requirement_id": "mission_success",
            "target_half_width": 0.05,
            "minimum_samples": 20,
            "maximum_samples": 100,
        }
    )

    assert isinstance(rule, ConfidenceIntervalStopping)
    assert rule.minimum_effective_sample_size is None
