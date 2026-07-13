from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from astro_uq.correlations import (
    gaussian_copula_transform,
    independent_transform,
    validate_correlation,
)
from astro_uq.models import CorrelationModel


def _correlation(matrix: tuple[tuple[float, ...], ...]) -> CorrelationModel:
    return CorrelationModel(parameter_ids=("x", "y"), matrix=matrix)


def test_independent_transform_is_an_unchanged_copy() -> None:
    points = np.array([[0.1, 0.8], [0.4, 0.2]])
    transformed = independent_transform(points)
    np.testing.assert_array_equal(transformed, points)
    assert transformed is not points


def test_gaussian_copula_has_uniform_marginals_and_requested_correlation() -> None:
    points = np.random.default_rng(91).random((150_000, 2))
    transformed = gaussian_copula_transform(
        points, _correlation(((1.0, 0.7), (0.7, 1.0))), ("x", "y")
    )
    np.testing.assert_allclose(np.mean(transformed, axis=0), 0.5, atol=0.003)
    latent = np.column_stack(
        [stats.norm.ppf(transformed[:, 0]), stats.norm.ppf(transformed[:, 1])]
    )
    assert np.corrcoef(latent, rowvar=False)[0, 1] == pytest.approx(0.7, abs=0.006)


def test_singular_psd_correlation_is_supported() -> None:
    transformed = gaussian_copula_transform(
        [[0.2, 0.8], [0.4, 0.1]], _correlation(((1.0, 1.0), (1.0, 1.0))), ("x", "y")
    )
    np.testing.assert_allclose(transformed[:, 0], transformed[:, 1])


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (((1.0, 0.2), (0.3, 1.0)), "symmetric"),
        (((0.9, 0.2), (0.2, 1.0)), "unit diagonal"),
        (((1.0, 1.1), (1.1, 1.0)), "positive semidefinite"),
    ],
)
def test_invalid_correlation_is_rejected_before_transform(
    matrix: tuple[tuple[float, ...], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_correlation(_correlation(matrix), ("x", "y"))


def test_parameter_order_and_point_dimension_are_validated() -> None:
    correlation = _correlation(((1.0, 0.2), (0.2, 1.0)))
    with pytest.raises(ValueError, match="ordering"):
        gaussian_copula_transform([[0.2, 0.3]], correlation, ("y", "x"))
    with pytest.raises(ValueError, match="dimensions"):
        gaussian_copula_transform([[0.2, 0.3, 0.4]], correlation, ("x", "y"))
