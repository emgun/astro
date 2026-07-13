from __future__ import annotations

from collections import Counter
from math import sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from astro_uq.models import (
    CampaignStatistics,
    CaseObservation,
    OutcomeStatus,
    StatisticSummary,
)


def summarize_campaign(
    *,
    requested_samples: int,
    observations: tuple[CaseObservation, ...],
    weights: dict[str, float],
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    requirement_ids: tuple[str, ...] | None = None,
    convergence_history: tuple[dict[str, object], ...] = (),
) -> CampaignStatistics:
    _validate_weights(observations, weights)
    outcome_counts = Counter(observation.outcome_status.value for observation in observations)
    metric_ids = sorted(
        {
            metric_id
            for observation in observations
            if observation.outcome_status is OutcomeStatus.SUCCESS
            for metric_id, value in observation.metric_values.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    )
    metric_summaries = tuple(
        _summarize_metric(metric_id, observations, weights, quantiles)
        for metric_id in metric_ids
    )
    configured_requirement_ids = requirement_ids or tuple(
        sorted(
            {
                result.requirement_id
                for observation in observations
                for result in observation.requirements
            }
        )
    )
    probabilities = {
        requirement_id: _requirement_probability(requirement_id, observations, weights)
        for requirement_id in configured_requirement_ids
    }
    return CampaignStatistics(
        requested_samples=requested_samples,
        completed_samples=len(observations),
        outcome_counts=dict(outcome_counts),
        metrics=metric_summaries,
        requirement_probabilities=probabilities,
        convergence_history=convergence_history,
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("successes and total must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def effective_sample_size(sample_weights: NDArray[np.float64]) -> float:
    if sample_weights.size == 0:
        return 0.0
    total = float(sample_weights.sum())
    squared = float(np.square(sample_weights).sum())
    return 0.0 if squared == 0.0 else total * total / squared


def _summarize_metric(
    metric_id: str,
    observations: tuple[CaseObservation, ...],
    weights: dict[str, float],
    quantiles: tuple[float, ...],
) -> StatisticSummary:
    pairs = [
        (float(value), weights[observation.sample_id])
        for observation in observations
        if observation.outcome_status is OutcomeStatus.SUCCESS
        if (value := observation.metric_values.get(metric_id)) is not None
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ]
    values = np.asarray([value for value, _ in pairs], dtype=np.float64)
    sample_weights = np.asarray([weight for _, weight in pairs], dtype=np.float64)
    normalized = sample_weights / sample_weights.sum()
    mean = float(np.sum(normalized * values))
    variance = float(np.sum(normalized * np.square(values - mean)))
    ess = effective_sample_size(sample_weights)
    return StatisticSummary(
        metric_id=metric_id,
        count=len(values),
        effective_sample_size=ess,
        mean=mean,
        variance=variance,
        standard_error=sqrt(variance / ess) if ess > 0.0 else None,
        quantiles={
            _quantile_label(quantile): _weighted_quantile(values, sample_weights, quantile)
            for quantile in quantiles
        },
    )


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantiles must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _requirement_probability(
    requirement_id: str,
    observations: tuple[CaseObservation, ...],
    weights: dict[str, float],
) -> float:
    passed_sample_ids = {
        observation.sample_id
        for observation in observations
        for result in observation.requirements
        if result.requirement_id == requirement_id and result.passed is True
    }
    total = sum(weights[observation.sample_id] for observation in observations)
    return sum(weights[sample_id] for sample_id in passed_sample_ids) / total


def _validate_weights(
    observations: tuple[CaseObservation, ...], weights: dict[str, float]
) -> None:
    sample_ids = [observation.sample_id for observation in observations]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("observations must have unique sample ids")
    if set(sample_ids) != set(weights):
        raise ValueError("weights must exactly match observation sample ids")
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights.values()):
        raise ValueError("weights must be finite and positive")


def _quantile_label(quantile: float) -> str:
    return f"q{quantile:.6f}".rstrip("0").rstrip(".")
