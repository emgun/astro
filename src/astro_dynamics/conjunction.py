from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, field_validator

from astro_core.models import (
    AstroModel,
    CovarianceSample,
    Trajectory,
    Vector3,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
    _numeric_sequence_input_must_be_numbers,
)

_PROBABILITY_METHODS: set[str] = {"density", "integrated"}
_PROBABILITY_RADIAL_QUADRATURE_ORDER = 32
_PROBABILITY_ANGULAR_QUADRATURE_ORDER = 64


class ConjunctionScreeningResult(AstroModel):
    primary_scenario_id: str = Field(min_length=1)
    secondary_scenario_id: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    compared_sample_count: int = Field(gt=0)
    tca_epoch: datetime
    tca_sample_index: int = Field(ge=0)
    miss_distance_km: FiniteFloat = Field(ge=0.0)
    relative_speed_km_s: FiniteFloat = Field(ge=0.0)
    threshold_km: FiniteFloat = Field(gt=0.0)
    status: Literal["below_threshold", "above_threshold"]
    hard_body_radius_km: FiniteFloat | None = Field(default=None, gt=0.0)
    probability_of_collision: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    primary_position_km: Vector3
    secondary_position_km: Vector3
    relative_position_km: Vector3
    relative_velocity_km_s: Vector3
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_count", "compared_sample_count", "tca_sample_index", mode="before")
    @classmethod
    def integer_inputs_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Conjunction screening integer")

    @field_validator(
        "miss_distance_km",
        "relative_speed_km_s",
        "threshold_km",
        "hard_body_radius_km",
        "probability_of_collision",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_scalar_input_must_be_number(value, "Conjunction screening scalar")

    @field_validator(
        "primary_position_km",
        "secondary_position_km",
        "relative_position_km",
        "relative_velocity_km_s",
        mode="before",
    )
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Conjunction screening vector")

    @field_validator(
        "primary_position_km",
        "secondary_position_km",
        "relative_position_km",
        "relative_velocity_km_s",
    )
    @classmethod
    def vectors_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Conjunction screening vector values must be finite")
        return value


def _tuple3(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _trajectory_samples_by_epoch(trajectory: Trajectory) -> dict[datetime, int]:
    return {sample.epoch: index for index, sample in enumerate(trajectory.samples)}


def _trajectory_covariance_by_epoch(trajectory: Trajectory) -> dict[datetime, CovarianceSample]:
    return {sample.epoch: sample for sample in trajectory.covariance_history}


def _matrix3_to_nested_list(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> list[list[float]]:
    return [[float(component) for component in row] for row in matrix]


def _orthonormal_encounter_plane(
    relative_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    relative_velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    velocity_norm = float(np.linalg.norm(relative_velocity))
    if velocity_norm > 0.0:
        normal = relative_velocity / velocity_norm
    else:
        position_norm = float(np.linalg.norm(relative_position))
        if position_norm > 0.0:
            normal = relative_position / position_norm
        else:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = reference - float(np.dot(reference, normal)) * normal
    axis_u = axis_u / float(np.linalg.norm(axis_u))
    axis_v = np.cross(normal, axis_u)
    return np.vstack([axis_u, axis_v])


def _probability_metadata_unavailable(reason: str) -> dict[str, Any]:
    return {
        "probability_model": "unavailable",
        "probability_unavailable_reason": reason,
    }


def _density_probability(
    *,
    projected_relative_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    inverse_covariance: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    determinant: float,
    hard_body_radius_km: float,
) -> float:
    mahalanobis_squared = float(
        projected_relative_position @ inverse_covariance @ projected_relative_position
    )
    density_at_miss = float(
        np.exp(-0.5 * mahalanobis_squared) / (2.0 * np.pi * determinant**0.5)
    )
    return min(1.0, max(0.0, density_at_miss * np.pi * hard_body_radius_km**2))


def _integrated_probability(
    *,
    projected_relative_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    inverse_covariance: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    determinant: float,
    hard_body_radius_km: float,
) -> float:
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(  # type: ignore[no-untyped-call]
        _PROBABILITY_RADIAL_QUADRATURE_ORDER
    )
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(  # type: ignore[no-untyped-call]
        _PROBABILITY_ANGULAR_QUADRATURE_ORDER
    )
    radii = 0.5 * hard_body_radius_km * (radial_nodes + 1.0)
    radius_weights = 0.5 * hard_body_radius_km * radial_weights
    angles = np.pi * (angular_nodes + 1.0)
    angle_weights = np.pi * angular_weights
    normalizer = 1.0 / (2.0 * np.pi * determinant**0.5)
    integral = 0.0
    for radius, radius_weight in zip(radii, radius_weights, strict=True):
        for angle, angle_weight in zip(angles, angle_weights, strict=True):
            point = np.array(
                [radius * np.cos(angle), radius * np.sin(angle)],
                dtype=np.float64,
            )
            delta = point - projected_relative_position
            exponent = -0.5 * float(delta @ inverse_covariance @ delta)
            integral += float(normalizer * np.exp(exponent) * radius * radius_weight * angle_weight)
    return min(1.0, max(0.0, integral))


def _covariance_probability_metadata(
    primary: Trajectory,
    secondary: Trajectory,
    *,
    tca_epoch: datetime,
    relative_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    relative_velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
    hard_body_radius_km: float | None,
    probability_method: str,
) -> tuple[float | None, dict[str, Any]]:
    if hard_body_radius_km is None:
        return None, {}
    if not isfinite(hard_body_radius_km) or hard_body_radius_km <= 0.0:
        raise ValueError("hard_body_radius_km must be finite and positive")

    primary_covariance = _trajectory_covariance_by_epoch(primary).get(tca_epoch)
    secondary_covariance = _trajectory_covariance_by_epoch(secondary).get(tca_epoch)
    if primary_covariance is None or secondary_covariance is None:
        return None, _probability_metadata_unavailable("missing_tca_covariance")

    primary_position_covariance = np.array(primary_covariance.covariance, dtype=np.float64)[:3, :3]
    secondary_position_covariance = np.array(secondary_covariance.covariance, dtype=np.float64)[
        :3,
        :3,
    ]
    combined_position_covariance = primary_position_covariance + secondary_position_covariance
    encounter_plane = _orthonormal_encounter_plane(relative_position, relative_velocity)
    encounter_covariance = encounter_plane @ combined_position_covariance @ encounter_plane.T
    encounter_covariance = 0.5 * (encounter_covariance + encounter_covariance.T)
    determinant = float(np.linalg.det(encounter_covariance))
    if determinant <= 0.0 or not isfinite(determinant):
        return None, _probability_metadata_unavailable("singular_encounter_covariance")

    projected_relative_position = encounter_plane @ relative_position
    inverse_covariance = np.linalg.inv(encounter_covariance)
    if probability_method == "density":
        probability = _density_probability(
            projected_relative_position=projected_relative_position,
            inverse_covariance=inverse_covariance,
            determinant=determinant,
            hard_body_radius_km=hard_body_radius_km,
        )
        probability_metadata: dict[str, Any] = {
            "probability_model": "encounter_plane_gaussian_density",
            "probability_approximation": "local_gaussian_density_times_hard_body_area",
        }
    else:
        probability = _integrated_probability(
            projected_relative_position=projected_relative_position,
            inverse_covariance=inverse_covariance,
            determinant=determinant,
            hard_body_radius_km=hard_body_radius_km,
        )
        probability_metadata = {
            "probability_model": "encounter_plane_gaussian_integral",
            "probability_quadrature": "gauss_legendre_polar",
            "probability_radial_quadrature_order": _PROBABILITY_RADIAL_QUADRATURE_ORDER,
            "probability_angular_quadrature_order": _PROBABILITY_ANGULAR_QUADRATURE_ORDER,
        }
    return probability, probability_metadata | {
        "covariance_source": "trajectory_covariance_history",
        "combined_position_covariance_km2": _matrix3_to_nested_list(
            combined_position_covariance
        ),
        "encounter_plane_covariance_km2": [
            [float(component) for component in row] for row in encounter_covariance
        ],
        "encounter_plane_relative_position_km": [
            float(component) for component in projected_relative_position
        ],
    }


def screen_conjunction(
    primary: Trajectory,
    secondary: Trajectory,
    *,
    threshold_km: float,
    hard_body_radius_km: float | None = None,
    probability_method: str = "integrated",
) -> ConjunctionScreeningResult:
    if not isfinite(threshold_km) or threshold_km <= 0.0:
        raise ValueError("threshold_km must be finite and positive")
    if probability_method not in _PROBABILITY_METHODS:
        raise ValueError("probability_method must be density or integrated")

    primary_by_epoch = _trajectory_samples_by_epoch(primary)
    secondary_by_epoch = _trajectory_samples_by_epoch(secondary)
    common_epochs = sorted(set(primary_by_epoch) & set(secondary_by_epoch))
    if not common_epochs:
        raise ValueError("Conjunction screening requires trajectories with common sample epochs")

    miss_distances: list[float] = []
    relative_positions: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    relative_velocities: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for epoch in common_epochs:
        primary_sample = primary.samples[primary_by_epoch[epoch]]
        secondary_sample = secondary.samples[secondary_by_epoch[epoch]]
        primary_position = primary_sample.state.position_array()
        secondary_position = secondary_sample.state.position_array()
        primary_velocity = primary_sample.state.velocity_array()
        secondary_velocity = secondary_sample.state.velocity_array()
        relative_position = secondary_position - primary_position
        relative_velocity = secondary_velocity - primary_velocity
        relative_positions.append(relative_position)
        relative_velocities.append(relative_velocity)
        miss_distances.append(float(np.linalg.norm(relative_position)))

    tca_sample_index = int(np.argmin(np.array(miss_distances, dtype=np.float64)))
    tca_epoch = common_epochs[tca_sample_index]
    primary_tca = primary.samples[primary_by_epoch[tca_epoch]]
    secondary_tca = secondary.samples[secondary_by_epoch[tca_epoch]]
    relative_position = relative_positions[tca_sample_index]
    relative_velocity = relative_velocities[tca_sample_index]
    miss_distance_km = miss_distances[tca_sample_index]
    relative_speed_km_s = float(np.linalg.norm(relative_velocity))
    status: Literal["below_threshold", "above_threshold"] = (
        "below_threshold" if miss_distance_km <= threshold_km else "above_threshold"
    )
    probability, probability_metadata = _covariance_probability_metadata(
        primary,
        secondary,
        tca_epoch=tca_epoch,
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        hard_body_radius_km=hard_body_radius_km,
        probability_method=probability_method,
    )

    return ConjunctionScreeningResult(
        primary_scenario_id=primary.scenario_id,
        secondary_scenario_id=secondary.scenario_id,
        sample_count=len(primary.samples) + len(secondary.samples),
        compared_sample_count=len(common_epochs),
        tca_epoch=tca_epoch,
        tca_sample_index=tca_sample_index,
        miss_distance_km=miss_distance_km,
        relative_speed_km_s=relative_speed_km_s,
        threshold_km=threshold_km,
        status=status,
        hard_body_radius_km=hard_body_radius_km,
        probability_of_collision=probability,
        primary_position_km=primary_tca.state.position_km,
        secondary_position_km=secondary_tca.state.position_km,
        relative_position_km=_tuple3(relative_position),
        relative_velocity_km_s=_tuple3(relative_velocity),
        metadata={
            "screening_model": "time_aligned_sample_minimum_distance",
            "primary_backend": primary.backend,
            "secondary_backend": secondary.backend,
            "primary_sample_count": len(primary.samples),
            "secondary_sample_count": len(secondary.samples),
        }
        | probability_metadata,
    )
