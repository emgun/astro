from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats  # type: ignore[import-untyped]

from astro_uq.models import CorrelationModel

PSD_TOLERANCE = 1.0e-10


def validate_correlation(
    correlation: CorrelationModel,
    parameter_ids: Sequence[str],
    *,
    psd_tolerance: float = PSD_TOLERANCE,
) -> NDArray[np.float64]:
    """Validate ordering and numerical correlation-matrix invariants."""
    if psd_tolerance < 0.0:
        raise ValueError("PSD tolerance must be nonnegative")
    if tuple(parameter_ids) != correlation.parameter_ids:
        raise ValueError("correlation parameter ordering must match transform parameter ordering")
    matrix = np.asarray(correlation.matrix, dtype=np.float64)
    dimension = len(parameter_ids)
    if matrix.shape != (dimension, dimension):
        raise ValueError("correlation matrix dimensions must match parameter ids")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation matrix must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=psd_tolerance):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=psd_tolerance):
        raise ValueError("correlation matrix must have a unit diagonal")
    symmetric = (matrix + matrix.T) / 2.0
    if float(np.linalg.eigvalsh(symmetric)[0]) < -psd_tolerance:
        raise ValueError("correlation matrix must be positive semidefinite")
    return symmetric


def independent_transform(normalized_points: ArrayLike) -> NDArray[np.float64]:
    """Validate and copy independent unit-hypercube points."""
    return _normalized_points(normalized_points).copy()


def gaussian_copula_transform(
    normalized_points: ArrayLike,
    correlation: CorrelationModel,
    parameter_ids: Sequence[str],
    *,
    psd_tolerance: float = PSD_TOLERANCE,
) -> NDArray[np.float64]:
    """Apply a Gaussian copula while retaining uniform marginals."""
    points = _normalized_points(normalized_points)
    matrix = validate_correlation(
        correlation, parameter_ids, psd_tolerance=psd_tolerance
    )
    if points.shape[-1] != matrix.shape[0]:
        raise ValueError("normalized point dimensions must match correlation dimensions")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    open_points = np.clip(
        points, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
    )
    latent = stats.norm.ppf(open_points)
    return np.asarray(stats.norm.cdf(latent @ factor.T), dtype=np.float64)


def _normalized_points(values: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 0:
        raise ValueError("normalized points must have at least one dimension")
    if not np.all(np.isfinite(points)) or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("normalized points must be finite and within [0, 1]")
    return points
