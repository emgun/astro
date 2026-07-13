from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from astro_uq.statistics import wilson_interval


@dataclass(frozen=True)
class StoppingDecision:
    stop: bool
    reason: str


@dataclass(frozen=True)
class ConfidenceIntervalRule:
    target_half_width: float
    minimum_samples: int
    maximum_samples: int
    confidence: float = 0.95
    minimum_effective_sample_size: float | None = None
    batch_size: int = 1


@dataclass(frozen=True)
class MetricStabilityRule:
    absolute_tolerance: float
    minimum_samples: int
    maximum_samples: int
    window: int = 3
    minimum_effective_sample_size: float | None = None
    batch_size: int = 1


def fixed_count_decision(completed: int, requested: int) -> StoppingDecision:
    _validate_counts(completed, requested)
    return StoppingDecision(
        stop=completed >= requested,
        reason="fixed_count_reached" if completed >= requested else "continue",
    )


def confidence_interval_decision(
    *,
    successes: int,
    completed: int,
    effective_sample_size: float,
    rule: ConfidenceIntervalRule,
) -> StoppingDecision:
    _validate_counts(completed, rule.maximum_samples)
    _validate_binomial_inputs(successes, completed)
    _validate_effective_sample_size(effective_sample_size, completed)
    _validate_confidence_interval_rule(rule)
    if completed >= rule.maximum_samples:
        return StoppingDecision(True, "maximum_samples_reached")
    if completed < rule.minimum_samples:
        return StoppingDecision(False, "minimum_samples_not_reached")
    if completed % rule.batch_size != 0:
        return StoppingDecision(False, "batch_boundary_not_reached")
    if effective_sample_size < _effective_sample_size_floor(rule):
        return StoppingDecision(False, "effective_sample_size_insufficient")
    lower, upper = wilson_interval(successes, completed, confidence=rule.confidence)
    if (upper - lower) / 2.0 <= rule.target_half_width:
        return StoppingDecision(True, "confidence_interval_converged")
    return StoppingDecision(False, "continue")


def metric_stability_decision(
    history: tuple[float, ...],
    *,
    completed: int,
    effective_sample_size: float | None = None,
    rule: MetricStabilityRule,
) -> StoppingDecision:
    _validate_counts(completed, rule.maximum_samples)
    _validate_metric_stability_rule(rule)
    effective_count = float(completed) if effective_sample_size is None else effective_sample_size
    _validate_effective_sample_size(effective_count, completed)
    if any(not isfinite(value) for value in history):
        raise ValueError("metric history values must be finite")
    if completed >= rule.maximum_samples:
        return StoppingDecision(True, "maximum_samples_reached")
    if completed < rule.minimum_samples:
        return StoppingDecision(False, "minimum_samples_not_reached")
    if completed % rule.batch_size != 0:
        return StoppingDecision(False, "batch_boundary_not_reached")
    if effective_count < _effective_sample_size_floor(rule):
        return StoppingDecision(False, "effective_sample_size_insufficient")
    if len(history) < rule.window:
        return StoppingDecision(False, "minimum_history_not_reached")
    recent = history[-rule.window :]
    if max(recent) - min(recent) <= rule.absolute_tolerance:
        return StoppingDecision(True, "metric_stability_converged")
    return StoppingDecision(False, "continue")


def _validate_counts(completed: int, requested: int) -> None:
    if (
        isinstance(completed, bool)
        or isinstance(requested, bool)
        or not isinstance(completed, int)
        or not isinstance(requested, int)
    ):
        raise ValueError("counts must be integers")
    if completed < 0 or requested <= 0:
        raise ValueError("completed must be nonnegative and requested must be positive")


def _validate_rule_bounds(minimum: int, maximum: int) -> None:
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise ValueError("stopping-rule bounds must be integers")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("stopping rules require 0 < minimum <= maximum")


def _validate_confidence_interval_rule(rule: ConfidenceIntervalRule) -> None:
    _validate_rule_bounds(rule.minimum_samples, rule.maximum_samples)
    if not isfinite(rule.target_half_width) or not 0.0 < rule.target_half_width <= 0.5:
        raise ValueError("target half-width must be finite and in (0, 0.5]")
    if not isfinite(rule.confidence) or not 0.0 < rule.confidence < 1.0:
        raise ValueError("confidence must be finite and between zero and one")
    _validate_shared_rule_fields(rule)


def _validate_metric_stability_rule(rule: MetricStabilityRule) -> None:
    _validate_rule_bounds(rule.minimum_samples, rule.maximum_samples)
    if not isfinite(rule.absolute_tolerance) or rule.absolute_tolerance < 0.0:
        raise ValueError("absolute tolerance must be finite and nonnegative")
    if isinstance(rule.window, bool) or rule.window < 2:
        raise ValueError("metric-stability window must be an integer of at least two")
    _validate_shared_rule_fields(rule)


def _validate_shared_rule_fields(rule: ConfidenceIntervalRule | MetricStabilityRule) -> None:
    if isinstance(rule.batch_size, bool) or rule.batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    floor = _effective_sample_size_floor(rule)
    if not isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum effective sample size must be finite and positive")
    if floor > rule.maximum_samples:
        raise ValueError("minimum effective sample size cannot exceed maximum samples")


def _effective_sample_size_floor(
    rule: ConfidenceIntervalRule | MetricStabilityRule,
) -> float:
    if rule.minimum_effective_sample_size is None:
        return float(rule.minimum_samples)
    return rule.minimum_effective_sample_size


def _validate_binomial_inputs(successes: int, completed: int) -> None:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if successes < 0 or successes > completed:
        raise ValueError("successes must satisfy 0 <= successes <= completed")


def _validate_effective_sample_size(effective_sample_size: float, completed: int) -> None:
    if isinstance(effective_sample_size, bool) or not isfinite(effective_sample_size):
        raise ValueError("effective sample size must be finite")
    if effective_sample_size < 0.0 or effective_sample_size > completed:
        raise ValueError("effective sample size must be between zero and completed")
