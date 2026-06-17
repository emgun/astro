from math import acos

import pytest

from astro_dynamics.attitude import (
    AttitudeControlConfig,
    RigidBodyAttitudeConfig,
    TorqueCommand,
    propagate_rigid_body_attitude,
)


def test_propagate_rigid_body_attitude_integrates_constant_body_torque() -> None:
    config = RigidBodyAttitudeConfig(
        duration_s=2.0,
        step_s=1.0,
        inertia_kg_m2=(2.0, 3.0, 4.0),
        torque_commands=(
            TorqueCommand(start_s=0.0, end_s=2.1, torque_n_m=(0.0, 0.0, 2.0)),
        ),
    )

    result = propagate_rigid_body_attitude(config)

    assert result.sample_count == 3
    assert result.samples[-1].elapsed_s == 2.0
    assert result.samples[-1].angular_rate_rad_s == pytest.approx((0.0, 0.0, 1.0))
    assert result.samples[-1].body_to_inertial_quaternion == pytest.approx(
        (0.87758256189, 0.0, 0.0, 0.4794255386)
    )
    assert result.metadata["attitude_dynamics_model"] == "diagonal_rigid_body_torque"


def test_propagate_rigid_body_attitude_preserves_state_without_torque() -> None:
    config = RigidBodyAttitudeConfig(
        duration_s=2.0,
        step_s=1.0,
        inertia_kg_m2=(2.0, 3.0, 4.0),
        initial_angular_rate_rad_s=(0.0, 0.0, 0.0),
    )

    result = propagate_rigid_body_attitude(config)

    assert {sample.body_to_inertial_quaternion for sample in result.samples} == {
        (1.0, 0.0, 0.0, 0.0)
    }
    assert {sample.angular_rate_rad_s for sample in result.samples} == {
        (0.0, 0.0, 0.0)
    }


def test_propagate_closed_loop_attitude_reduces_quaternion_error() -> None:
    target_quaternion = (0.96592582629, 0.0, 0.0, 0.2588190451)
    config = RigidBodyAttitudeConfig(
        duration_s=8.0,
        step_s=0.5,
        inertia_kg_m2=(8.0, 8.0, 4.0),
        control=AttitudeControlConfig(
            target_body_to_inertial_quaternion=target_quaternion,
            proportional_gain_n_m_per_rad=(0.0, 0.0, 0.45),
            derivative_gain_n_m_per_rad_s=(0.0, 0.0, 2.4),
            max_torque_n_m=(0.0, 0.0, 0.35),
        ),
    )

    result = propagate_rigid_body_attitude(config)

    initial_error = _quaternion_distance_rad(
        result.samples[0].body_to_inertial_quaternion,
        target_quaternion,
    )
    final_error = _quaternion_distance_rad(
        result.samples[-1].body_to_inertial_quaternion,
        target_quaternion,
    )
    assert final_error < initial_error * 0.35
    assert result.samples[0].control_torque_n_m == pytest.approx((0.0, 0.0, 0.23293714059))
    assert result.samples[-1].control_torque_n_m is not None
    assert abs(result.samples[-1].control_torque_n_m[2]) <= 0.35
    assert result.metadata["attitude_dynamics_model"] == "diagonal_rigid_body_closed_loop_pd"
    assert result.metadata["control_model"] == "quaternion_error_pd"
    assert result.metadata["control_saturation_enabled"] is True


def _quaternion_distance_rad(
    actual: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    dot = abs(sum(left * right for left, right in zip(actual, target, strict=True)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * acos(dot)
