from __future__ import annotations

from astro_reentry.models import (
    ReentryLimits,
    ReentryMargin,
    ReentryMarginReport,
    ReentryMarginStatus,
    ReentryPeakSummary,
    ReentryTarget,
    ReentryTargetMiss,
)


def build_reentry_margin_report(
    *,
    limits: ReentryLimits,
    peaks: ReentryPeakSummary,
    target: ReentryTarget | None,
    target_miss: ReentryTargetMiss | None,
) -> ReentryMarginReport:
    margins = [
        _maximum_margin(
            "dynamic_pressure_margin_pa",
            peaks.dynamic_pressure.value,
            float(limits.maximum_dynamic_pressure_pa),
            "Pa",
        ),
        _maximum_margin(
            "deceleration_margin_g",
            peaks.deceleration.value,
            float(limits.maximum_deceleration_g),
            "g",
        ),
        _maximum_margin(
            "heat_rate_margin_w_m2",
            peaks.heat_rate.value,
            float(limits.maximum_heat_rate_w_m2),
            "W/m^2",
        ),
        _maximum_margin(
            "heat_load_margin_j_m2",
            float(peaks.total_heat_load_j_m2),
            float(limits.maximum_heat_load_j_m2),
            "J/m^2",
        ),
    ]
    if target is not None and target_miss is not None:
        margins.append(
            _maximum_margin(
                "target_miss_margin_km",
                float(target_miss.distance_km),
                float(target.allowable_miss_km),
                "km",
            )
        )
    limiting = min(margins, key=_limiting_key)
    return ReentryMarginReport(margins=tuple(margins), limiting_margin=limiting)


def _maximum_margin(name: str, value: float, threshold: float, unit: str) -> ReentryMargin:
    margin = threshold - value
    warn_band = 0.1 * threshold
    status = (
        ReentryMarginStatus.FAIL
        if margin < 0.0
        else ReentryMarginStatus.WARN
        if margin <= warn_band
        else ReentryMarginStatus.PASS
    )
    return ReentryMargin(
        name=name,
        value=value,
        threshold=threshold,
        margin=margin,
        unit=unit,
        status=status,
    )


def _limiting_key(margin: ReentryMargin) -> tuple[int, float]:
    severity = {
        ReentryMarginStatus.FAIL: 0,
        ReentryMarginStatus.WARN: 1,
        ReentryMarginStatus.PASS: 2,
    }[margin.status]
    return severity, float(margin.margin / abs(margin.threshold))
