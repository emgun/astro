import pytest

from astro_dynamics.attitude import (
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
