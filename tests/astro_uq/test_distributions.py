from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from astro_uq.distributions import build_realization, inverse_cdf, transform_value
from astro_uq.models import DistributionKind, DistributionSpec, UncertainParameter


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (DistributionSpec(kind="constant", value=4.0), np.array([4.0, 4.0, 4.0])),
        (DistributionSpec(kind="uniform", low=2.0, high=6.0), np.array([2.0, 4.0, 6.0])),
        (
            DistributionSpec(kind="normal", mean=3.0, sigma=2.0),
            stats.norm.ppf([0.25, 0.5, 0.75], loc=3.0, scale=2.0),
        ),
        (
            DistributionSpec(kind="lognormal", mean=0.5, sigma=0.2),
            stats.lognorm.ppf([0.25, 0.5, 0.75], s=0.2, scale=np.exp(0.5)),
        ),
        (
            DistributionSpec(kind="triangular", low=0.0, mode=2.0, high=8.0),
            stats.triang.ppf([0.25, 0.5, 0.75], c=0.25, loc=0.0, scale=8.0),
        ),
    ],
)
def test_inverse_cdf_matches_scipy(spec: DistributionSpec, expected: np.ndarray) -> None:
    quantiles = (
        np.array([0.0, 0.5, 1.0])
        if spec.kind in {DistributionKind.CONSTANT, DistributionKind.UNIFORM}
        else np.array([0.25, 0.5, 0.75])
    )
    np.testing.assert_allclose(inverse_cdf(spec, quantiles), expected)


def test_empirical_and_categorical_inverse_transforms() -> None:
    empirical = DistributionSpec(
        kind="empirical", values=(30.0, 10.0, 20.0), probabilities=(0.2, 0.3, 0.5)
    )
    np.testing.assert_array_equal(
        inverse_cdf(empirical, [0.0, 0.29, 0.3, 0.79, 1.0]),
        [10, 10, 10, 20, 30],
    )
    categorical = DistributionSpec(
        kind="categorical", labels=("a", "b"), probabilities=(0.25, 0.75)
    )
    assert [
        transform_value(categorical, value) for value in (0.0, 0.249, 0.25, 1.0)
    ] == ["a", "a", "a", "b"]


def test_seeded_independent_normal_moments() -> None:
    rng = np.random.default_rng(817)
    values = inverse_cdf(
        DistributionSpec(kind="normal", mean=12.0, sigma=3.0), rng.random(100_000)
    )
    assert np.mean(values) == pytest.approx(12.0, abs=0.025)
    assert np.std(values) == pytest.approx(3.0, abs=0.025)


def test_realization_preserves_normalized_coordinates() -> None:
    parameters = (
        UncertainParameter(
            parameter_id="mass", target="mass", unit="kg", uncertainty_kind="aleatory",
            distribution=DistributionSpec(kind="uniform", low=10.0, high=20.0),
        ),
    )
    realization = build_realization(
        parameters, {"mass": 0.25}, sample_id="sample-0", sample_index=0
    )
    assert realization.normalized_values == {"mass": 0.25}
    assert realization.physical_values == {"mass": 12.5}


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan])
def test_inverse_cdf_rejects_invalid_normalized_coordinates(value: float) -> None:
    with pytest.raises(ValueError, match="within"):
        inverse_cdf(DistributionSpec(kind="uniform", low=0.0, high=1.0), value)
