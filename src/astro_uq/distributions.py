from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats  # type: ignore[import-untyped]

from astro_uq.models import (
    DistributionKind,
    DistributionSpec,
    ParameterRealization,
    UncertainParameter,
)

PhysicalValue = float | str


def inverse_cdf(distribution: DistributionSpec, quantiles: ArrayLike) -> NDArray[np.float64]:
    """Map unit-interval coordinates to numeric physical values."""
    unit = _unit_coordinates(quantiles)
    if distribution.kind is DistributionKind.CONSTANT:
        return np.full(unit.shape, cast(float, distribution.value), dtype=np.float64)
    if distribution.kind is DistributionKind.UNIFORM:
        low = cast(float, distribution.low)
        high = cast(float, distribution.high)
        return np.asarray(
            stats.uniform.ppf(
                unit,
                loc=low,
                scale=high - low,
            ),
            dtype=np.float64,
        )

    open_unit = np.clip(unit, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    if distribution.kind is DistributionKind.NORMAL:
        mean = cast(float, distribution.mean)
        sigma = cast(float, distribution.sigma)
        return np.asarray(
            stats.norm.ppf(open_unit, loc=mean, scale=sigma),
            dtype=np.float64,
        )
    if distribution.kind is DistributionKind.LOGNORMAL:
        mean = cast(float, distribution.mean)
        sigma = cast(float, distribution.sigma)
        return np.asarray(
            stats.lognorm.ppf(
                open_unit,
                s=sigma,
                scale=np.exp(mean),
            ),
            dtype=np.float64,
        )
    if distribution.kind is DistributionKind.TRIANGULAR:
        low = cast(float, distribution.low)
        high = cast(float, distribution.high)
        mode = cast(float, distribution.mode)
        width = high - low
        shape = (mode - low) / width
        return np.asarray(
            stats.triang.ppf(unit, c=shape, loc=low, scale=width),
            dtype=np.float64,
        )
    if distribution.kind is DistributionKind.EMPIRICAL:
        values = np.asarray(distribution.values, dtype=np.float64)
        probabilities = _probabilities(distribution.probabilities, values.size)
        order = np.argsort(values, kind="stable")
        ordered_values = values[order]
        cumulative = np.cumsum(probabilities[order])
        indices = np.searchsorted(cumulative, unit, side="left")
        result = ordered_values[np.minimum(indices, ordered_values.size - 1)]
        return np.asarray(result, dtype=np.float64)
    raise TypeError(f"{distribution.kind.value} distributions do not produce numeric values")


def transform_value(distribution: DistributionSpec, quantile: float) -> PhysicalValue:
    """Transform one normalized coordinate, including categorical definitions."""
    unit = _unit_coordinates(quantile)
    if distribution.kind is DistributionKind.CATEGORICAL:
        probabilities = _probabilities(distribution.probabilities, len(distribution.labels))
        index = min(
            int(np.searchsorted(np.cumsum(probabilities), float(unit), side="left")),
            len(distribution.labels) - 1,
        )
        return distribution.labels[index]
    return float(inverse_cdf(distribution, unit))


def build_realization(
    parameters: Sequence[UncertainParameter],
    normalized_values: Mapping[str, float],
    *,
    sample_id: str,
    sample_index: int,
    weight: float = 1.0,
) -> ParameterRealization:
    """Build a replayable physical realization from named normalized coordinates."""
    expected = tuple(parameter.parameter_id for parameter in parameters)
    if tuple(normalized_values) != expected:
        raise ValueError("normalized value ordering must match uncertainty parameter ordering")
    normalized = {
        name: float(_unit_coordinates(value)) for name, value in normalized_values.items()
    }
    physical = {
        parameter.parameter_id: transform_value(
            parameter.distribution, normalized[parameter.parameter_id]
        )
        for parameter in parameters
    }
    return ParameterRealization(
        sample_id=sample_id,
        sample_index=sample_index,
        normalized_values=normalized,
        physical_values=physical,
        weight=weight,
    )


def _unit_coordinates(values: ArrayLike) -> NDArray[np.float64]:
    unit = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(unit)) or np.any(unit < 0.0) or np.any(unit > 1.0):
        raise ValueError("normalized coordinates must be finite and within [0, 1]")
    return unit


def _probabilities(probabilities: Sequence[float], count: int) -> NDArray[np.float64]:
    if probabilities:
        return np.asarray(probabilities, dtype=np.float64)
    return np.full(count, 1.0 / count, dtype=np.float64)
