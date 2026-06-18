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


class AttitudeSensorConfig(AstroModel):
    attitude_bias_rad: Vector3 = (0.0, 0.0, 0.0)
    angular_rate_bias_rad_s: Vector3 = (0.0, 0.0, 0.0)

    @field_validator("attitude_bias_rad", "angular_rate_bias_rad_s", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude sensor vector")

    @field_validator("attitude_bias_rad", "angular_rate_bias_rad_s")
    @classmethod
    def sensor_vectors_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Attitude sensor bias values must be finite")
        return value


class AttitudeActuatorConfig(AstroModel):
    torque_scale: Vector3 = (1.0, 1.0, 1.0)
    torque_bias_n_m: Vector3 = (0.0, 0.0, 0.0)
    deadband_n_m: Vector3 = (0.0, 0.0, 0.0)

    @field_validator("torque_scale", "torque_bias_n_m", "deadband_n_m", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude actuator vector")

    @field_validator("torque_scale", "deadband_n_m")
    @classmethod
    def nonnegative_vectors_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) and component >= 0.0 for component in value):
            raise ValueError("Attitude actuator scale and deadband values must be nonnegative")
        return value

    @field_validator("torque_bias_n_m")
    @classmethod
    def torque_bias_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Attitude actuator torque bias values must be finite")
        return value


class AttitudeControlConfig(AstroModel):
    target_body_to_inertial_quaternion: Quaternion4
    target_angular_rate_rad_s: Vector3 = (0.0, 0.0, 0.0)
    proportional_gain_n_m_per_rad: Vector3
    derivative_gain_n_m_per_rad_s: Vector3
    max_torque_n_m: Vector3
    sensor: AttitudeSensorConfig | None = None
    actuator: AttitudeActuatorConfig | None = None

    @field_validator(
        "target_body_to_inertial_quaternion",
        mode="before",
    )
    @classmethod
    def quaternion_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude control quaternion")

    @field_validator(
        "target_angular_rate_rad_s",
        "proportional_gain_n_m_per_rad",
        "derivative_gain_n_m_per_rad_s",
        "max_torque_n_m",
        mode="before",
    )
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Attitude control vector")

    @field_validator("target_body_to_inertial_quaternion")
    @classmethod
    def target_quaternion_must_be_unit(cls, value: Quaternion4) -> Quaternion4:
        norm = sqrt(sum(component * component for component in value))
        if not isfinite(norm) or abs(norm - 1.0) > 1.0e-9:
            raise ValueError("Target attitude quaternion must be a unit quaternion")
        return value

    @field_validator(
        "proportional_gain_n_m_per_rad",
        "derivative_gain_n_m_per_rad_s",
        "max_torque_n_m",
    )
    @classmethod
    def control_vectors_must_be_nonnegative(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) and component >= 0.0 for component in value):
            raise ValueError("Attitude control gains and torque limits must be nonnegative")
        return value


class RigidBodyAttitudeConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    inertia_kg_m2: Vector3
    initial_body_to_inertial_quaternion: Quaternion4 = (1.0, 0.0, 0.0, 0.0)
    initial_angular_rate_rad_s: Vector3 = (0.0, 0.0, 0.0)
    torque_commands: tuple[TorqueCommand, ...] = ()
    control: AttitudeControlConfig | None = None

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
    control_torque_n_m: Vector3 | None = None
    commanded_control_torque_n_m: Vector3 | None = None
    measured_body_to_inertial_quaternion: Quaternion4 | None = None
    measured_angular_rate_rad_s: Vector3 | None = None


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
        feedforward_torque = _active_torque(config, elapsed_s)
        measured_quaternion, measured_rate = _attitude_sensor_measurement(
            config.control,
            quaternion,
            angular_rate,
        )
        commanded_control_torque = _closed_loop_control_torque(
            config.control,
            measured_quaternion,
            measured_rate,
        )
        control_torque = _actuated_control_torque(config.control, commanded_control_torque)
        torque = feedforward_torque + control_torque
        samples.append(
            AttitudeDynamicsSample(
                elapsed_s=elapsed_s,
                body_to_inertial_quaternion=_quaternion_tuple(quaternion),
                angular_rate_rad_s=_vector_tuple(angular_rate),
                applied_torque_n_m=_vector_tuple(torque),
                control_torque_n_m=(
                    _vector_tuple(control_torque) if config.control is not None else None
                ),
                commanded_control_torque_n_m=(
                    _vector_tuple(commanded_control_torque)
                    if config.control is not None
                    else None
                ),
                measured_body_to_inertial_quaternion=(
                    _quaternion_tuple(measured_quaternion)
                    if config.control is not None and config.control.sensor is not None
                    else None
                ),
                measured_angular_rate_rad_s=(
                    _vector_tuple(measured_rate)
                    if config.control is not None and config.control.sensor is not None
                    else None
                ),
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

    metadata = {
        "attitude_dynamics_model": "diagonal_rigid_body_torque",
        "integration_model": "piecewise_constant_torque_average_rate_quaternion",
        "torque_command_count": len(config.torque_commands),
    }
    if config.control is not None:
        metadata.update(
            {
                "attitude_dynamics_model": "diagonal_rigid_body_closed_loop_pd",
                "control_model": "quaternion_error_pd",
                "control_saturation_enabled": True,
                "target_body_to_inertial_quaternion": list(
                    config.control.target_body_to_inertial_quaternion
                ),
                "target_angular_rate_rad_s": list(
                    config.control.target_angular_rate_rad_s
                ),
            }
        )
        if config.control.sensor is not None:
            metadata.update(
                {
                    "attitude_sensor_model": "deterministic_bias",
                    "sensor_attitude_bias_rad": list(config.control.sensor.attitude_bias_rad),
                    "sensor_angular_rate_bias_rad_s": list(
                        config.control.sensor.angular_rate_bias_rad_s
                    ),
                }
            )
        if config.control.actuator is not None:
            metadata.update(
                {
                    "attitude_actuator_model": "deterministic_scale_bias_deadband",
                    "actuator_torque_scale": list(config.control.actuator.torque_scale),
                    "actuator_torque_bias_n_m": list(
                        config.control.actuator.torque_bias_n_m
                    ),
                    "actuator_deadband_n_m": list(config.control.actuator.deadband_n_m),
                }
            )
        if config.control.sensor is not None or config.control.actuator is not None:
            metadata["attitude_dynamics_model"] = (
                "diagonal_rigid_body_closed_loop_pd_sensor_actuator_screening"
            )

    return AttitudeDynamicsResult(
        sample_count=len(samples),
        samples=tuple(samples),
        metadata=metadata,
    )


def _active_torque(config: RigidBodyAttitudeConfig, elapsed_s: float) -> np.ndarray[Any, Any]:
    torque = np.zeros(3, dtype=np.float64)
    for command in config.torque_commands:
        if command.start_s <= elapsed_s < command.end_s:
            torque += np.array(command.torque_n_m, dtype=np.float64)
    return torque


def _closed_loop_control_torque(
    control: AttitudeControlConfig | None,
    body_to_inertial_quaternion: np.ndarray[Any, Any],
    angular_rate_rad_s: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if control is None:
        return np.zeros(3, dtype=np.float64)

    target_quaternion = np.array(
        control.target_body_to_inertial_quaternion,
        dtype=np.float64,
    )
    target_rate = np.array(control.target_angular_rate_rad_s, dtype=np.float64)
    proportional_gain = np.array(control.proportional_gain_n_m_per_rad, dtype=np.float64)
    derivative_gain = np.array(control.derivative_gain_n_m_per_rad_s, dtype=np.float64)
    max_torque = np.array(control.max_torque_n_m, dtype=np.float64)
    quaternion_error = _quaternion_multiply(
        target_quaternion,
        _quaternion_conjugate(body_to_inertial_quaternion),
    )
    quaternion_error = _normalize_quaternion(quaternion_error)
    if quaternion_error[0] < 0.0:
        quaternion_error = -quaternion_error
    error_vector_rad = 2.0 * quaternion_error[1:]
    rate_error = target_rate - angular_rate_rad_s
    unsaturated_torque = proportional_gain * error_vector_rad + derivative_gain * rate_error
    return np.array(
        [
            min(max(float(torque), -float(limit)), float(limit))
            for torque, limit in zip(unsaturated_torque, max_torque, strict=True)
        ],
        dtype=np.float64,
    )


def _attitude_sensor_measurement(
    control: AttitudeControlConfig | None,
    body_to_inertial_quaternion: np.ndarray[Any, Any],
    angular_rate_rad_s: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if control is None or control.sensor is None:
        return body_to_inertial_quaternion, angular_rate_rad_s
    attitude_bias = np.array(control.sensor.attitude_bias_rad, dtype=np.float64)
    rate_bias = np.array(control.sensor.angular_rate_bias_rad_s, dtype=np.float64)
    measured_quaternion = _normalize_quaternion(
        _quaternion_multiply(
            _delta_quaternion(attitude_bias),
            body_to_inertial_quaternion,
        )
    )
    measured_rate = angular_rate_rad_s + rate_bias
    return measured_quaternion, measured_rate


def _actuated_control_torque(
    control: AttitudeControlConfig | None,
    commanded_torque_n_m: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if control is None or control.actuator is None:
        return commanded_torque_n_m
    actuator = control.actuator
    torque = commanded_torque_n_m * np.array(actuator.torque_scale, dtype=np.float64)
    torque = torque + np.array(actuator.torque_bias_n_m, dtype=np.float64)
    deadband = np.array(actuator.deadband_n_m, dtype=np.float64)
    torque = np.where(np.abs(torque) < deadband, 0.0, torque)
    max_torque = np.array(control.max_torque_n_m, dtype=np.float64)
    return np.array(
        [
            min(max(float(axis_torque), -float(limit)), float(limit))
            for axis_torque, limit in zip(torque, max_torque, strict=True)
        ],
        dtype=np.float64,
    )


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


def _quaternion_conjugate(quaternion: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    )


def _normalize_quaternion(quaternion: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return quaternion / float(np.linalg.norm(quaternion))


def _quaternion_tuple(quaternion: np.ndarray[Any, Any]) -> Quaternion4:
    return tuple(float(component) for component in quaternion)  # type: ignore[return-value]


def _vector_tuple(vector: np.ndarray[Any, Any]) -> Vector3:
    return tuple(float(component) for component in vector)  # type: ignore[return-value]
