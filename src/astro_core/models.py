from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from numbers import Number
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


def _datetime_input_must_be_datetime_or_string(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_ | Number | np.number | Decimal | Fraction):
        raise ValueError(f"{label} must be a datetime or ISO datetime string")
    if isinstance(value, str):
        try:
            float(value.strip())
        except ValueError:
            return value
        else:
            raise ValueError(f"{label} must be a datetime or ISO datetime string")
    return value


def _numeric_scalar_input_must_be_number(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_ | str):
        raise ValueError(f"{label} must be a numeric scalar")
    return value


def _integer_input_must_be_int(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _numeric_sequence_input_must_be_numbers(value: Any, label: str) -> Any:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} does not accept NumPy arrays")
    if isinstance(value, str | bytes):
        raise ValueError(f"{label} must contain numeric scalar values")
    if not isinstance(value, list | tuple):
        return value

    for component in value:
        _numeric_scalar_input_must_be_number(component, label)
    return value


def _numeric_matrix_input_must_be_numbers(value: Any, label: str) -> Any:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} does not accept NumPy arrays")
    if isinstance(value, str | bytes):
        raise ValueError(f"{label} must contain numeric scalar values")
    if not isinstance(value, list | tuple):
        return value

    for row in value:
        _numeric_sequence_input_must_be_numbers(row, label)
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

    @field_validator("position_km", "velocity_km_s", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Cartesian state vector")

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

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "OrbitState epoch")

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

    @field_validator(
        "mass_kg",
        "area_m2",
        "drag_coefficient",
        "reflectivity_coefficient",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Spacecraft scalar")


class ForceModelConfig(AstroModel):
    gravity: ForceModelName


class PropagationConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)

    @field_validator("duration_s", "step_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Propagation scalar")

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
    elevation_mask_deg: FiniteFloat = Field(ge=-90.0, le=90.0)

    @field_validator("position_eci_km", mode="before")
    @classmethod
    def position_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Ground station position")

    @field_validator("elevation_mask_deg", mode="before")
    @classmethod
    def elevation_mask_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Ground station elevation mask")

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

    @field_validator("range_sigma_km", "range_rate_sigma_km_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement noise scalar")

    @field_validator("seed", mode="before")
    @classmethod
    def seed_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Measurement noise seed")


class MeasurementConfig(AstroModel):
    types: tuple[MeasurementType, ...] = Field(
        default=(MeasurementType.RANGE, MeasurementType.RANGE_RATE),
        min_length=1,
    )
    cadence_s: FiniteFloat = Field(gt=0.0, default=60.0)
    noise: MeasurementNoise = Field(default_factory=MeasurementNoise)

    @field_validator("cadence_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement cadence")


class MeasurementRecord(AstroModel):
    measurement_type: MeasurementType
    epoch: datetime
    observer: str
    observed_object: str
    value: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)
    units: Literal["km", "km/s"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "MeasurementRecord epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "MeasurementRecord epoch")

    @field_validator("value", "sigma", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement scalar")

    @field_validator("value", "sigma")
    @classmethod
    def numeric_values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Measurement numeric values must be finite")
        return value

    @model_validator(mode="after")
    def measurement_units_must_match_type(self) -> MeasurementRecord:
        expected_units = {
            MeasurementType.RANGE: "km",
            MeasurementType.RANGE_RATE: "km/s",
        }
        if self.units != expected_units[self.measurement_type]:
            expected_unit = expected_units[self.measurement_type]
            raise ValueError(f"{self.measurement_type} measurements must use units {expected_unit}")
        return self


class TrajectorySample(AstroModel):
    epoch: datetime
    state: CartesianState

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "TrajectorySample epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "TrajectorySample epoch")


class Trajectory(AstroModel):
    scenario_id: str = Field(min_length=1)
    samples: list[TrajectorySample] = Field(min_length=1)
    force_model: ForceModelConfig
    backend: str = Field(min_length=1)
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

    @field_validator("residuals", mode="before")
    @classmethod
    def residual_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "EstimateResult residuals")

    @field_validator("residuals")
    @classmethod
    def residuals_must_be_finite(cls, value: list[float]) -> list[float]:
        if not all(isfinite(residual) for residual in value):
            raise ValueError("EstimateResult residuals must be finite")
        return value

    @field_validator("rms", mode="before")
    @classmethod
    def rms_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "EstimateResult rms")

    @field_validator("rms")
    @classmethod
    def rms_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("EstimateResult rms must be finite")
        return value

    @field_validator("covariance", mode="before")
    @classmethod
    def covariance_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_matrix_input_must_be_numbers(value, "EstimateResult covariance")

    @field_validator("covariance")
    @classmethod
    def covariance_must_be_6x6_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("EstimateResult covariance must be 6x6")
        if not all(isfinite(component) for row in value for component in row):
            raise ValueError("EstimateResult covariance values must be finite")
        return value

    @field_validator("iterations", mode="before")
    @classmethod
    def iterations_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "EstimateResult iterations")


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
    metadata: dict[str, Any] = Field(default_factory=dict)
