from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, field_validator

from astro_core.models import (
    AstroModel,
    Trajectory,
    Vector3,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
    _numeric_sequence_input_must_be_numbers,
)


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
    primary_position_km: Vector3
    secondary_position_km: Vector3
    relative_position_km: Vector3
    relative_velocity_km_s: Vector3
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_count", "compared_sample_count", "tca_sample_index", mode="before")
    @classmethod
    def integer_inputs_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Conjunction screening integer")

    @field_validator("miss_distance_km", "relative_speed_km_s", "threshold_km", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
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


def screen_conjunction(
    primary: Trajectory,
    secondary: Trajectory,
    *,
    threshold_km: float,
) -> ConjunctionScreeningResult:
    if not isfinite(threshold_km) or threshold_km <= 0.0:
        raise ValueError("threshold_km must be finite and positive")

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
        },
    )
