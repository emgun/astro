from __future__ import annotations

from statistics import median

from astro_uq.models import (
    CampaignTimingSummary,
    CaseObservation,
    OutcomeStatus,
    TimingPhaseSummary,
)


def summarize_case_timings(
    observations: tuple[CaseObservation, ...],
) -> CampaignTimingSummary:
    """Build machine-scoped timing evidence without changing campaign statistics."""
    if any(
        observation.outcome_status is not OutcomeStatus.SUCCESS for observation in observations
    ):
        raise ValueError("timing profiles require every campaign case to succeed")
    timings = [
        observation.evaluation_timing
        for observation in observations
        if observation.evaluation_timing is not None
    ]
    if not timings:
        raise ValueError("campaign cases do not contain evaluator timing evidence")

    setup = [float(timing.setup_s) for timing in timings]
    evaluation = [float(timing.evaluation_s) for timing in timings]
    extraction = [
        float(timing.metric_extraction_s)
        for timing in timings
        if timing.metric_extraction_s is not None
    ]
    serialization = [float(timing.serialization_s) for timing in timings]
    totals = [float(timing.total_s) for timing in timings]
    fully_instrumented = len(timings) == len(observations) and len(extraction) == len(observations)
    attributed = [
        setup_s
        + evaluation_s
        + (float(timing.metric_extraction_s) if timing.metric_extraction_s is not None else 0.0)
        + serialization_s
        for timing, setup_s, evaluation_s, serialization_s in zip(
            timings, setup, evaluation, serialization, strict=True
        )
    ]
    unattributed = [
        max(0.0, total_s - attributed_s)
        for total_s, attributed_s in zip(totals, attributed, strict=True)
    ]
    total_time = sum(totals)
    return CampaignTimingSummary(
        case_count=len(observations),
        fully_instrumented_case_count=len(extraction),
        setup=_summarize_phase(setup),
        evaluation=_summarize_phase(evaluation),
        metric_extraction=_summarize_phase(extraction) if extraction else None,
        serialization=_summarize_phase(serialization),
        unattributed=_summarize_phase(unattributed),
        total=_summarize_phase(totals),
        evaluation_share_of_instrumented_time=(
            sum(evaluation) / total_time if fully_instrumented and total_time else None
        ),
        accounted_share_of_instrumented_time=(
            sum(attributed) / total_time if fully_instrumented and total_time else None
        ),
    )


def _summarize_phase(values: list[float]) -> TimingPhaseSummary:
    middle = float(median(values))
    return TimingPhaseSummary(
        count=len(values),
        total_s=sum(values),
        mean_s=sum(values) / len(values),
        median_s=middle,
        minimum_s=min(values),
        maximum_s=max(values),
        median_absolute_deviation_s=float(median(abs(value - middle) for value in values)),
    )
