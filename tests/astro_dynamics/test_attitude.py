from math import acos

import pytest

from astro_dynamics.attitude import (
    AttitudeActuatorConfig,
    AttitudeControlConfig,
    AttitudeSensorConfig,
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
    assert result.samples[0].attitude_error_rad == pytest.approx(initial_error)
    assert result.samples[-1].attitude_error_rad == pytest.approx(final_error)
    assert result.samples[-1].angular_rate_error_norm_rad_s == pytest.approx(
        abs(result.samples[-1].angular_rate_rad_s[2])
    )
    assert result.samples[0].torque_saturated is False
    assert result.metadata["attitude_dynamics_model"] == "diagonal_rigid_body_closed_loop_pd"
    assert result.metadata["control_model"] == "quaternion_error_pd"
    assert result.metadata["control_saturation_enabled"] is True
    assert result.metadata["final_attitude_error_rad"] == pytest.approx(final_error)
    assert result.metadata["max_attitude_error_rad"] == pytest.approx(initial_error)
    assert result.metadata["attitude_control_status"] == "miss"


def test_propagate_closed_loop_attitude_applies_sensor_and_actuator_screening() -> None:
    config = RigidBodyAttitudeConfig(
        duration_s=1.0,
        step_s=1.0,
        inertia_kg_m2=(2.0, 2.0, 2.0),
        control=AttitudeControlConfig(
            target_body_to_inertial_quaternion=(1.0, 0.0, 0.0, 0.0),
            proportional_gain_n_m_per_rad=(0.0, 0.0, 0.0),
            derivative_gain_n_m_per_rad_s=(0.0, 0.0, 1.0),
            max_torque_n_m=(0.0, 0.0, 1.0),
            sensor=AttitudeSensorConfig(
                attitude_bias_rad=(0.0, 0.0, 0.02),
                angular_rate_bias_rad_s=(0.0, 0.0, 0.1),
            ),
            actuator=AttitudeActuatorConfig(
                torque_scale=(1.0, 1.0, 0.5),
                torque_bias_n_m=(0.0, 0.0, 0.02),
                deadband_n_m=(0.0, 0.0, 0.0),
            ),
        ),
    )

    result = propagate_rigid_body_attitude(config)

    first_sample = result.samples[0]
    assert first_sample.measured_angular_rate_rad_s == pytest.approx((0.0, 0.0, 0.1))
    assert first_sample.measured_body_to_inertial_quaternion == pytest.approx(
        (0.99995000042, 0.0, 0.0, 0.00999983333)
    )
    assert first_sample.commanded_control_torque_n_m == pytest.approx((0.0, 0.0, -0.1))
    assert first_sample.control_torque_n_m == pytest.approx((0.0, 0.0, -0.03))
    assert first_sample.control_torque_tracking_error_n_m == pytest.approx(
        (0.0, 0.0, 0.07)
    )
    assert first_sample.applied_torque_n_m == pytest.approx((0.0, 0.0, -0.03))
    assert first_sample.attitude_error_rad == pytest.approx(0.02)
    assert first_sample.angular_rate_error_norm_rad_s == pytest.approx(0.1)
    assert first_sample.torque_saturated is False
    assert first_sample.actuator_deadband_applied is False
    assert first_sample.actuator_saturated is False
    assert result.metadata["attitude_dynamics_model"] == (
        "diagonal_rigid_body_closed_loop_pd_sensor_actuator_screening"
    )
    assert result.metadata["attitude_sensor_model"] == "deterministic_bias"
    assert result.metadata["attitude_actuator_model"] == "deterministic_scale_bias_deadband"
    assert result.metadata["control_torque_tracking_error_model"] == (
        "applied_minus_commanded_control_torque"
    )
    assert result.metadata["max_control_torque_tracking_error_norm_n_m"] == pytest.approx(
        0.07
    )
    assert result.metadata["torque_saturation_sample_count"] == 0
    assert result.metadata["torque_saturation_fraction"] == 0.0
    assert result.metadata["actuator_deadband_sample_count"] == 0
    assert result.metadata["actuator_deadband_fraction"] == 0.0
    assert result.metadata["actuator_saturation_sample_count"] == 0
    assert result.metadata["actuator_saturation_fraction"] == 0.0


def test_attitude_control_status_uses_configured_tolerances() -> None:
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
            pointing_tolerance_rad=0.2,
            angular_rate_tolerance_rad_s=0.05,
        ),
    )

    result = propagate_rigid_body_attitude(config)

    within_tolerance_samples = [
        sample
        for sample in result.samples
        if sample.attitude_error_rad is not None
        and sample.attitude_error_rad <= 0.2
        and sample.angular_rate_error_norm_rad_s is not None
        and sample.angular_rate_error_norm_rad_s <= 0.05
    ]
    settled_sample = next(
        sample
        for sample_index, sample in enumerate(result.samples)
        if sample.attitude_error_rad is not None
        and sample.angular_rate_error_norm_rad_s is not None
        and all(
            later.attitude_error_rad is not None
            and later.attitude_error_rad <= 0.2
            and later.angular_rate_error_norm_rad_s is not None
            and later.angular_rate_error_norm_rad_s <= 0.05
            for later in result.samples[sample_index:]
        )
    )

    assert result.metadata["attitude_control_status"] == "within_tolerance"
    assert result.metadata["pointing_tolerance_rad"] == 0.2
    assert result.metadata["angular_rate_tolerance_rad_s"] == 0.05
    assert result.metadata["within_tolerance_sample_count"] == len(within_tolerance_samples)
    assert result.metadata["within_tolerance_fraction"] == pytest.approx(
        len(within_tolerance_samples) / len(result.samples)
    )
    assert result.metadata["first_within_tolerance_elapsed_s"] == pytest.approx(
        within_tolerance_samples[0].elapsed_s
    )
    assert result.metadata["settled_within_tolerance_elapsed_s"] == pytest.approx(
        settled_sample.elapsed_s
    )


def test_attitude_control_reports_saturation_and_deadband_samples() -> None:
    config = RigidBodyAttitudeConfig(
        duration_s=1.0,
        step_s=1.0,
        inertia_kg_m2=(2.0, 2.0, 2.0),
        control=AttitudeControlConfig(
            target_body_to_inertial_quaternion=(0.96592582629, 0.0, 0.0, 0.2588190451),
            proportional_gain_n_m_per_rad=(0.0, 0.0, 1.0),
            derivative_gain_n_m_per_rad_s=(0.0, 0.0, 0.0),
            max_torque_n_m=(0.0, 0.0, 0.05),
            actuator=AttitudeActuatorConfig(
                torque_scale=(1.0, 1.0, 1.0),
                torque_bias_n_m=(0.0, 0.0, 0.0),
                deadband_n_m=(0.0, 0.0, 0.1),
            ),
        ),
    )

    result = propagate_rigid_body_attitude(config)

    assert result.samples[0].torque_saturated is True
    assert result.samples[0].actuator_deadband_applied is True
    assert result.samples[0].commanded_control_torque_n_m == pytest.approx((0.0, 0.0, 0.05))
    assert result.samples[0].control_torque_n_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.metadata["torque_saturation_sample_count"] >= 1
    assert result.metadata["actuator_deadband_sample_count"] >= 1


def test_attitude_actuator_reports_post_actuator_saturation_and_tracking_error() -> None:
    config = RigidBodyAttitudeConfig(
        duration_s=1.0,
        step_s=1.0,
        inertia_kg_m2=(2.0, 2.0, 2.0),
        control=AttitudeControlConfig(
            target_body_to_inertial_quaternion=(1.0, 0.0, 0.0, 0.0),
            proportional_gain_n_m_per_rad=(0.0, 0.0, 0.0),
            derivative_gain_n_m_per_rad_s=(0.0, 0.0, 1.0),
            max_torque_n_m=(0.0, 0.0, 0.05),
            sensor=AttitudeSensorConfig(
                angular_rate_bias_rad_s=(0.0, 0.0, -0.04),
            ),
            actuator=AttitudeActuatorConfig(
                torque_scale=(1.0, 1.0, 3.0),
                torque_bias_n_m=(0.0, 0.0, 0.0),
                deadband_n_m=(0.0, 0.0, 0.0),
            ),
        ),
    )

    result = propagate_rigid_body_attitude(config)

    first_sample = result.samples[0]
    assert first_sample.torque_saturated is False
    assert first_sample.actuator_saturated is True
    assert first_sample.commanded_control_torque_n_m == pytest.approx((0.0, 0.0, 0.04))
    assert first_sample.control_torque_n_m == pytest.approx((0.0, 0.0, 0.05))
    assert first_sample.control_torque_tracking_error_n_m == pytest.approx(
        (0.0, 0.0, 0.01)
    )
    assert result.metadata["actuator_saturation_sample_count"] >= 1
    assert result.metadata["actuator_saturation_fraction"] > 0.0
    assert result.metadata["max_control_torque_tracking_error_norm_n_m"] >= 0.01
    assert result.metadata["rms_control_torque_tracking_error_norm_n_m"] > 0.0


def test_attitude_actuator_config_rejects_negative_deadband() -> None:
    with pytest.raises(ValueError, match="scale and deadband"):
        AttitudeActuatorConfig(deadband_n_m=(0.0, 0.0, -0.1))


def _quaternion_distance_rad(
    actual: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    dot = abs(sum(left * right for left, right in zip(actual, target, strict=True)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * acos(dot)
