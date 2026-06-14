from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vector3 = tuple[float, float, float]


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


class CartesianState(BaseModel):
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


class OrbitState(BaseModel):
    epoch: datetime
    time_scale: TimeScale
    frame: Frame
    central_body: Body
    representation: OrbitRepresentation
    cartesian: CartesianState

    @model_validator(mode="after")
    def validate_epoch(self) -> OrbitState:
        if self.epoch.tzinfo is None or self.epoch.utcoffset() is None:
            raise ValueError("OrbitState epoch must include timezone information")
        return self


class Spacecraft(BaseModel):
    name: str = Field(min_length=1)
    mass_kg: float = Field(gt=0.0)
    area_m2: float = Field(gt=0.0)
    drag_coefficient: float = Field(ge=0.0, le=10.0)
    reflectivity_coefficient: float = Field(ge=0.0, le=5.0)


class ForceModelConfig(BaseModel):
    gravity: ForceModelName


class PropagationConfig(BaseModel):
    duration_s: float = Field(gt=0.0)
    step_s: float = Field(gt=0.0)

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s / self.step_s)) + 1

    @model_validator(mode="after")
    def validate_steps(self) -> PropagationConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("Propagation duration_s must be an integer multiple of step_s")
        return self


class GroundStation(BaseModel):
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


class MeasurementNoise(BaseModel):
    range_sigma_km: float = Field(gt=0.0, default=0.01)
    range_rate_sigma_km_s: float = Field(gt=0.0, default=1.0e-5)
    seed: int = 42


class MeasurementConfig(BaseModel):
    types: tuple[MeasurementType, ...] = (MeasurementType.RANGE, MeasurementType.RANGE_RATE)
    cadence_s: float = Field(gt=0.0, default=60.0)
    noise: MeasurementNoise = Field(default_factory=MeasurementNoise)


class MeasurementRecord(BaseModel):
    measurement_type: MeasurementType
    epoch: datetime
    observer: str
    observed_object: str
    value: float
    sigma: float = Field(gt=0.0)
    units: Literal["km", "km/s"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectorySample(BaseModel):
    epoch: datetime
    state: CartesianState


class Trajectory(BaseModel):
    scenario_id: str
    samples: list[TrajectorySample]
    force_model: ForceModelConfig
    backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def epochs_must_be_monotonic(self) -> Trajectory:
        epochs = [sample.epoch for sample in self.samples]
        if epochs != sorted(epochs):
            raise ValueError("Trajectory sample epochs must be monotonic")
        return self


class EstimateResult(BaseModel):
    estimated_state: OrbitState
    residuals: list[float]
    covariance: list[list[float]]
    rms: float
    iterations: int
    converged: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    scenario_id: str = Field(min_length=1)
    description: str = ""
    spacecraft: Spacecraft
    initial_state: OrbitState
    force_model: ForceModelConfig
    propagation: PropagationConfig
    ground_stations: list[GroundStation] = Field(default_factory=list)
    measurements: MeasurementConfig = Field(default_factory=MeasurementConfig)
