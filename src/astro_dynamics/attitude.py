from __future__ import annotations

from math import cos, isfinite, sin, sqrt
from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.models import (
    AstroModel,
    Quaternion4,
    Vector3,
    _numeric_scalar_input_must_be_number,
    _numeric_sequence_input_must_be_numbers,
)


class TorqueCommand(AstroModel):
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    torque_n_m: Vector3

    @field_validator("start_s", "end_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Torque command scalar")

    @field_validator("torque_n_m", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Torque command vector")

    @model_validator(mode="after")
    def end_must_follow_start(self) -> TorqueCommand:
        if self.end_s <= self.start_s:
            raise ValueError("Torque command end_s must be greater than start_s")
        return self


class RigidBodyAttitudeConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    inertia_kg_m2: Vector3
    initial_body_to_inertial_quaternion: Quaternion4 = (1.0, 0.0, 0.0, 0.0)
    initial_angular_rate_rad_s: Vector3 = (0.0, 0.0, 0.0)
    torque_commands: tuple[TorqueCommand, ...] = ()

    @field_validator("duration_s", "step_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Attitude propagation scalar")

    @field_validator(
        "inertia_kg_m2",
        "initial_angular_rate_rad_s",
        mode="before",
    )
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude propagation vector")

    @field_validator("initial_body_to_inertial_quaternion", mode="before")
    @classmethod
    def quaternion_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude propagation quaternion")

    @field_validator("inertia_kg_m2")
    @classmethod
    def inertia_must_be_positive(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) and component > 0.0 for component in value):
            raise ValueError("Attitude inertia values must be finite and positive")
        return value

    @field_validator("initial_body_to_inertial_quaternion")
    @classmethod
    def quaternion_must_be_unit(cls, value: Quaternion4) -> Quaternion4:
        norm = sqrt(sum(component * component for component in value))
        if not isfinite(norm) or abs(norm - 1.0) > 1.0e-9:
            raise ValueError("Initial attitude quaternion must be a unit quaternion")
        return value

    @model_validator(mode="after")
    def duration_must_be_integer_step_count(self) -> RigidBodyAttitudeConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError("Attitude duration_s must be an integer multiple of step_s")
        return self


class AttitudeDynamicsSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    body_to_inertial_quaternion: Quaternion4
    angular_rate_rad_s: Vector3
    applied_torque_n_m: Vector3


class AttitudeDynamicsResult(AstroModel):
    sample_count: int = Field(gt=0)
    samples: tuple[AttitudeDynamicsSample, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def propagate_rigid_body_attitude(config: RigidBodyAttitudeConfig) -> AttitudeDynamicsResult:
    inertia = np.array(config.inertia_kg_m2, dtype=np.float64)
    angular_rate = np.array(config.initial_angular_rate_rad_s, dtype=np.float64)
    quaternion = np.array(config.initial_body_to_inertial_quaternion, dtype=np.float64)
    samples: list[AttitudeDynamicsSample] = []
    step_count = int(round(config.duration_s / config.step_s))

    for step_index in range(step_count + 1):
        elapsed_s = step_index * config.step_s
        torque = _active_torque(config, elapsed_s)
        samples.append(
            AttitudeDynamicsSample(
                elapsed_s=elapsed_s,
                body_to_inertial_quaternion=_quaternion_tuple(quaternion),
                angular_rate_rad_s=_vector_tuple(angular_rate),
                applied_torque_n_m=_vector_tuple(torque),
            )
        )
        if step_index == step_count:
            break

        angular_acceleration = torque / inertia
        average_angular_rate = angular_rate + 0.5 * angular_acceleration * config.step_s
        quaternion = _normalize_quaternion(
            _quaternion_multiply(
                quaternion,
                _delta_quaternion(average_angular_rate * config.step_s),
            )
        )
        angular_rate = angular_rate + angular_acceleration * config.step_s

    return AttitudeDynamicsResult(
        sample_count=len(samples),
        samples=tuple(samples),
        metadata={
            "attitude_dynamics_model": "diagonal_rigid_body_torque",
            "integration_model": "piecewise_constant_torque_average_rate_quaternion",
            "torque_command_count": len(config.torque_commands),
        },
    )


def _active_torque(config: RigidBodyAttitudeConfig, elapsed_s: float) -> np.ndarray[Any, Any]:
    torque = np.zeros(3, dtype=np.float64)
    for command in config.torque_commands:
        if command.start_s <= elapsed_s < command.end_s:
            torque += np.array(command.torque_n_m, dtype=np.float64)
    return torque


def _delta_quaternion(rotation_vector_rad: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    angle = float(np.linalg.norm(rotation_vector_rad))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotation_vector_rad / angle
    half_angle = 0.5 * angle
    return np.array(
        [cos(half_angle), *(sin(half_angle) * axis)],
        dtype=np.float64,
    )


def _quaternion_multiply(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _normalize_quaternion(quaternion: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return quaternion / float(np.linalg.norm(quaternion))


def _quaternion_tuple(quaternion: np.ndarray[Any, Any]) -> Quaternion4:
    return tuple(float(component) for component in quaternion)  # type: ignore[return-value]


def _vector_tuple(vector: np.ndarray[Any, Any]) -> Vector3:
    return tuple(float(component) for component in vector)  # type: ignore[return-value]
