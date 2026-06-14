from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

Vector3 = tuple[float, float, float]


class AstroModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _datetime_must_be_aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include timezone information")
    return value


class Body(StrEnum):
    EARTH = "earth"


class Frame(StrEnum):
    EME2000 = "EME2000"


class TimeScale(StrEnum):
    UTC = "UTC"


class OrbitRepresentation(StrEnum):
    CARTESIAN = "cartesian"


class ForceModelName(StrEnum):
    TWO_BODY = "two_body"
    J2 = "j2"
    OREKIT_HIGH_FIDELITY = "orekit_high_fidelity"


class MeasurementType(StrEnum):
    RANGE = "range"
    RANGE_RATE = "range_rate"


class CartesianState(AstroModel):
    position_km: Vector3
    velocity_km_s: Vector3

    @field_validator("position_km", "velocity_km_s")
    @classmethod
    def values_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Cartesian state values must be finite")
        return value

    def position_array(self) -> NDArray[np.float64]:
        return np.array(self.position_km, dtype=np.float64)

    def velocity_array(self) -> NDArray[np.float64]:
        return np.array(self.velocity_km_s, dtype=np.float64)


class OrbitState(AstroModel):
    epoch: datetime
    time_scale: TimeScale
    frame: Frame
    central_body: Body
    representation: OrbitRepresentation
    cartesian: CartesianState

    @model_validator(mode="after")
    def validate_epoch(self) -> OrbitState:
        _datetime_must_be_aware(self.epoch, "OrbitState epoch")
        return self


class Spacecraft(AstroModel):
    name: str = Field(min_length=1)
    mass_kg: FiniteFloat = Field(gt=0.0)
    area_m2: FiniteFloat = Field(gt=0.0)
    drag_coefficient: FiniteFloat = Field(ge=0.0, le=10.0)
    reflectivity_coefficient: FiniteFloat = Field(ge=0.0, le=5.0)


class ForceModelConfig(AstroModel):
    gravity: ForceModelName


class PropagationConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s / self.step_s)) + 1

    @model_validator(mode="after")
    def validate_steps(self) -> PropagationConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("Propagation duration_s must be an integer multiple of step_s")
        return self


class GroundStation(AstroModel):
    name: str = Field(min_length=1)
    position_eci_km: Vector3
    frame: Frame
    elevation_mask_deg: float = Field(ge=-90.0, le=90.0)

    @field_validator("position_eci_km")
    @classmethod
    def position_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Ground station position values must be finite")
        return value

    def position_array(self) -> NDArray[np.float64]:
        return np.array(self.position_eci_km, dtype=np.float64)


class MeasurementNoise(AstroModel):
    range_sigma_km: FiniteFloat = Field(gt=0.0, default=0.01)
    range_rate_sigma_km_s: FiniteFloat = Field(gt=0.0, default=1.0e-5)
    seed: int = 42


class MeasurementConfig(AstroModel):
    types: tuple[MeasurementType, ...] = (MeasurementType.RANGE, MeasurementType.RANGE_RATE)
    cadence_s: FiniteFloat = Field(gt=0.0, default=60.0)
    noise: MeasurementNoise = Field(default_factory=MeasurementNoise)


class MeasurementRecord(AstroModel):
    measurement_type: MeasurementType
    epoch: datetime
    observer: str
    observed_object: str
    value: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)
    units: Literal["km", "km/s"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "MeasurementRecord epoch")

    @field_validator("value", "sigma")
    @classmethod
    def numeric_values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Measurement numeric values must be finite")
        return value


class TrajectorySample(AstroModel):
    epoch: datetime
    state: CartesianState

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "TrajectorySample epoch")


class Trajectory(AstroModel):
    scenario_id: str
    samples: list[TrajectorySample] = Field(min_length=1)
    force_model: ForceModelConfig
    backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def epochs_must_be_monotonic(self) -> Trajectory:
        epochs = [sample.epoch for sample in self.samples]
        for epoch in epochs:
            _datetime_must_be_aware(epoch, "Trajectory sample epoch")

        try:
            is_strictly_increasing = all(
                previous_epoch < next_epoch
                for previous_epoch, next_epoch in zip(epochs, epochs[1:], strict=False)
            )
        except TypeError as exc:
            raise ValueError("Trajectory sample epochs must be comparable aware datetimes") from exc

        if not is_strictly_increasing:
            raise ValueError("Trajectory sample epochs must be strictly increasing")
        return self


class EstimateResult(AstroModel):
    estimated_state: OrbitState
    residuals: list[FiniteFloat]
    covariance: list[list[FiniteFloat]]
    rms: FiniteFloat
    iterations: int = Field(ge=0)
    converged: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("residuals")
    @classmethod
    def residuals_must_be_finite(cls, value: list[float]) -> list[float]:
        if not all(isfinite(residual) for residual in value):
            raise ValueError("EstimateResult residuals must be finite")
        return value

    @field_validator("rms")
    @classmethod
    def rms_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("EstimateResult rms must be finite")
        return value

    @field_validator("covariance")
    @classmethod
    def covariance_must_be_6x6_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("EstimateResult covariance must be 6x6")
        if not all(isfinite(component) for row in value for component in row):
            raise ValueError("EstimateResult covariance values must be finite")
        return value


class Scenario(AstroModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    scenario_id: str = Field(min_length=1)
    description: str = ""
    spacecraft: Spacecraft
    initial_state: OrbitState
    force_model: ForceModelConfig
    propagation: PropagationConfig
    ground_stations: list[GroundStation] = Field(default_factory=list)
    measurements: MeasurementConfig = Field(default_factory=MeasurementConfig)
