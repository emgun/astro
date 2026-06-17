from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from astro_core.constants import MU_EARTH_KM3_S2
from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName, Maneuver, Scenario
from astro_dynamics.local import (
    acceleration_km_s2,
    derivative,
    j2_acceleration_km_s2,
    propagate_local,
    rk4_step,
    two_body_acceleration_km_s2,
)

FORCE_MODEL_TYPE_ERROR = "force_model must be a ForceModelName"
LOCAL_FORCE_MODEL_ERROR = "Local backend supports only two_body and j2 force models"


def test_two_body_acceleration_points_toward_origin() -> None:
    acceleration = two_body_acceleration_km_s2(np.array([7000.0, 0.0, 0.0]))

    assert acceleration[0] < 0.0
    assert acceleration[1] == 0.0
    assert acceleration[2] == 0.0


def test_two_body_acceleration_matches_expected_magnitude_at_7000_km() -> None:
    acceleration = two_body_acceleration_km_s2(np.array([7000.0, 0.0, 0.0]))

    expected = np.array([-MU_EARTH_KM3_S2 / 7000.0**2, 0.0, 0.0])
    np.testing.assert_allclose(acceleration, expected, rtol=1e-12, atol=0.0)


def test_two_body_acceleration_rejects_zero_radius() -> None:
    with pytest.raises(ValueError, match="zero-radius"):
        two_body_acceleration_km_s2(np.array([0.0, 0.0, 0.0]))


def test_j2_acceleration_is_nonzero_for_inclined_state() -> None:
    acceleration = j2_acceleration_km_s2(np.array([5000.0, 3000.0, 4000.0]))

    assert np.linalg.norm(acceleration) > 0.0


def test_j2_acceleration_rejects_zero_radius() -> None:
    with pytest.raises(ValueError, match="zero-radius"):
        j2_acceleration_km_s2(np.array([0.0, 0.0, 0.0]))


def test_acceleration_rejects_unsupported_local_force_model() -> None:
    with pytest.raises(ValueError, match=LOCAL_FORCE_MODEL_ERROR):
        acceleration_km_s2(
            np.array([7000.0, 0.0, 0.0]),
            ForceModelName.OREKIT_HIGH_FIDELITY,
        )


def test_acceleration_rejects_raw_string_force_model() -> None:
    with pytest.raises(ValueError, match=FORCE_MODEL_TYPE_ERROR):
        acceleration_km_s2(np.array([7000.0, 0.0, 0.0]), "j2")


def test_derivative_rejects_raw_string_force_model() -> None:
    with pytest.raises(ValueError, match=FORCE_MODEL_TYPE_ERROR):
        derivative(np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0]), "two_body")


def test_rk4_step_rejects_unsupported_local_force_model() -> None:
    with pytest.raises(ValueError, match=LOCAL_FORCE_MODEL_ERROR):
        rk4_step(
            np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0]),
            60.0,
            ForceModelName.OREKIT_HIGH_FIDELITY,
        )


def test_propagate_local_returns_expected_sample_count() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    assert trajectory.backend == "local"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.samples[0].state.position_km == scenario.initial_state.cartesian.position_km
    assert trajectory.metadata["integrator"] == "rk4"
    assert trajectory.metadata["step_s"] == scenario.propagation.step_s
    assert trajectory.metadata["sample_step_s"] == scenario.propagation.step_s


def test_propagate_local_applies_finite_burn_schedule() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    burn = Maneuver(
        name="finite-trim",
        epoch=scenario.initial_state.epoch + timedelta(seconds=60),
        frame=scenario.initial_state.frame,
        delta_v_km_s=(0.0, 0.06, 0.0),
        duration_s=120.0,
    )
    payload = scenario.model_dump(mode="json") | {
        "maneuvers": [burn.model_dump(mode="json")]
    }
    maneuvered_scenario = Scenario.model_validate(payload)

    baseline = propagate_local(scenario)
    trajectory = propagate_local(maneuvered_scenario)
    baseline_final_velocity = np.array(baseline.samples[-1].state.velocity_km_s)
    maneuvered_final_velocity = np.array(trajectory.samples[-1].state.velocity_km_s)

    assert maneuvered_scenario.maneuvers == [burn]
    assert trajectory.maneuvers == [burn]
    assert maneuvered_final_velocity[1] - baseline_final_velocity[1] > 0.05
    assert {event.event_type for event in trajectory.events} >= {
        "maneuver_start",
        "maneuver_end",
    }
    assert trajectory.metadata["finite_burn_count"] == 1


def test_propagate_local_applies_thrust_vector_mass_flow_burn() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    burn = Maneuver(
        name="thrust-trim",
        epoch=scenario.initial_state.epoch + timedelta(seconds=60),
        frame=scenario.initial_state.frame,
        delta_v_km_s=(0.0, 0.0, 0.0),
        duration_s=120.0,
        thrust_vector_n=(0.0, 0.25, 0.0),
        specific_impulse_s=220.0,
    )
    maneuvered_scenario = scenario.model_copy(update={"maneuvers": [burn]})

    baseline = propagate_local(scenario)
    trajectory = propagate_local(maneuvered_scenario)
    baseline_final_velocity = np.array(baseline.samples[-1].state.velocity_km_s)
    maneuvered_final_velocity = np.array(trajectory.samples[-1].state.velocity_km_s)
    masses = [sample.mass_kg for sample in trajectory.samples]

    assert maneuvered_final_velocity[1] > baseline_final_velocity[1]
    assert masses[0] == pytest.approx(scenario.spacecraft.mass_kg)
    assert masses[-1] is not None
    assert masses[-1] < scenario.spacecraft.mass_kg
    assert masses == sorted(masses, reverse=True)
    assert trajectory.metadata["maneuver_model"] == "thrust_vector_mass_flow"
    assert trajectory.metadata["thrust_vector_burn_count"] == 1
    assert trajectory.metadata["final_mass_kg"] == pytest.approx(masses[-1])
    assert trajectory.events[0].metadata["thrust_vector_n"] == (0.0, 0.25, 0.0)


def test_propagate_local_applies_velocity_aligned_thrust_vector_burn() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    burn = Maneuver(
        name="velocity-aligned-trim",
        epoch=scenario.initial_state.epoch + timedelta(seconds=60),
        frame=scenario.initial_state.frame,
        delta_v_km_s=(0.0, 0.0, 0.0),
        duration_s=120.0,
        thrust_vector_n=(0.0, 0.25, 0.0),
        specific_impulse_s=220.0,
        thrust_direction_mode="velocity_aligned",
    )
    maneuvered_scenario = scenario.model_copy(update={"maneuvers": [burn]})

    baseline = propagate_local(scenario)
    trajectory = propagate_local(maneuvered_scenario)
    baseline_final_velocity = np.array(baseline.samples[-1].state.velocity_km_s)
    maneuvered_final_velocity = np.array(trajectory.samples[-1].state.velocity_km_s)
    velocity_delta = maneuvered_final_velocity - baseline_final_velocity

    assert velocity_delta[1] > 0.0
    assert velocity_delta[2] > 0.0
    assert trajectory.metadata["attitude_coupled_burn_count"] == 1
    assert trajectory.metadata["thrust_direction_modes"] == ["velocity_aligned"]
    assert trajectory.events[0].metadata["thrust_direction_mode"] == "velocity_aligned"


@pytest.mark.parametrize(
    ("thrust_direction_mode", "expected_x_sign"),
    [
        ("radial_outward", 1.0),
        ("radial_inward", -1.0),
    ],
)
def test_propagate_local_applies_radial_thrust_vector_burn(
    thrust_direction_mode: Literal["radial_outward", "radial_inward"],
    expected_x_sign: float,
) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    burn = Maneuver(
        name="radial-trim",
        epoch=scenario.initial_state.epoch + timedelta(seconds=60),
        frame=scenario.initial_state.frame,
        delta_v_km_s=(0.0, 0.0, 0.0),
        duration_s=120.0,
        thrust_vector_n=(0.0, 0.25, 0.0),
        specific_impulse_s=220.0,
        thrust_direction_mode=thrust_direction_mode,
    )
    maneuvered_scenario = scenario.model_copy(update={"maneuvers": [burn]})

    baseline = propagate_local(scenario)
    trajectory = propagate_local(maneuvered_scenario)
    baseline_final_velocity = np.array(baseline.samples[-1].state.velocity_km_s)
    maneuvered_final_velocity = np.array(trajectory.samples[-1].state.velocity_km_s)
    velocity_delta = maneuvered_final_velocity - baseline_final_velocity

    assert expected_x_sign * velocity_delta[0] > 0.0
    assert trajectory.metadata["attitude_coupled_burn_count"] == 1
    assert trajectory.metadata["thrust_direction_modes"] == [thrust_direction_mode]
    assert trajectory.events[0].metadata["thrust_direction_mode"] == thrust_direction_mode


def test_propagate_local_generates_covariance_history_from_initial_covariance() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    initial_covariance = [
        [1.0 if row == column else 0.0 for column in range(6)] for row in range(6)
    ]
    covariance_scenario = Scenario.model_validate(
        scenario.model_dump(mode="json") | {"initial_covariance": initial_covariance}
    )

    trajectory = propagate_local(covariance_scenario)

    assert len(trajectory.covariance_history) == covariance_scenario.propagation.sample_count
    assert trajectory.covariance_history[0].epoch == covariance_scenario.initial_state.epoch
    assert trajectory.covariance_history[0].covariance == initial_covariance
    assert trajectory.covariance_history[-1].epoch == trajectory.samples[-1].epoch
    final_covariance = np.array(trajectory.covariance_history[-1].covariance)
    final_transition = np.array(trajectory.covariance_history[-1].state_transition_matrix)
    final_accumulated_transition = np.array(
        trajectory.covariance_history[-1].accumulated_state_transition_matrix
    )
    assert final_covariance.shape == (6, 6)
    assert final_transition.shape == (6, 6)
    assert final_accumulated_transition.shape == (6, 6)
    np.testing.assert_allclose(final_covariance, final_covariance.T, rtol=0.0, atol=1.0e-10)
    assert not np.allclose(final_covariance, np.array(initial_covariance))
    assert trajectory.metadata["covariance_model"] == "finite_difference_state_transition"
    assert trajectory.metadata["covariance_state_transition_storage"] == (
        "per_sample_and_accumulated"
    )
    assert trajectory.covariance_history[0].metadata["covariance_sample_role"] == "initial"
    assert trajectory.covariance_history[-1].metadata["covariance_sample_role"] == "propagated"
    assert trajectory.covariance_history[-1].metadata["transition_step_s"] == (
        covariance_scenario.propagation.step_s
    )
    assert trajectory.covariance_history[-1].metadata["state_transition_model"] == (
        "finite_difference"
    )
    np.testing.assert_allclose(
        np.array(trajectory.covariance_history[0].state_transition_matrix),
        np.eye(6),
        rtol=0.0,
        atol=0.0,
    )


def test_propagate_local_adds_acceleration_process_noise_to_covariance() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    initial_covariance = [[0.0 for _column in range(6)] for _row in range(6)]
    covariance_scenario = Scenario.model_validate(
        scenario.model_dump(mode="json")
        | {
            "initial_covariance": initial_covariance,
            "covariance_process_noise_acceleration_km_s2": 1.0e-9,
        }
    )

    trajectory = propagate_local(covariance_scenario)

    final_covariance = np.array(trajectory.covariance_history[-1].covariance)
    final_process_noise = np.array(trajectory.covariance_history[-1].process_noise_covariance)
    assert np.any(np.diag(final_covariance) > 0.0)
    assert np.any(np.diag(final_process_noise) > 0.0)
    np.testing.assert_allclose(final_covariance, final_covariance.T, rtol=0.0, atol=1.0e-20)
    np.testing.assert_allclose(final_process_noise, final_process_noise.T, rtol=0.0, atol=1.0e-20)
    assert trajectory.metadata["covariance_process_noise"] == "white_acceleration"
    assert trajectory.covariance_history[-1].metadata["process_noise_model"] == (
        "white_acceleration"
    )
    assert trajectory.metadata["covariance_process_noise_acceleration_km_s2"] == 1.0e-9


def test_propagate_local_rejects_unsupported_local_force_model() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    unsupported_scenario = scenario.model_copy(
        update={
            "force_model": ForceModelConfig(gravity=ForceModelName.OREKIT_HIGH_FIDELITY),
        }
    )

    with pytest.raises(ValueError, match=LOCAL_FORCE_MODEL_ERROR):
        propagate_local(unsupported_scenario)


def test_propagate_local_rejects_unsupported_high_fidelity_flags() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    unsupported_scenario = scenario.model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.TWO_BODY,
                atmospheric_drag=True,
            ),
        }
    )

    with pytest.raises(ValueError, match="atmospheric_drag"):
        propagate_local(unsupported_scenario)


def test_j2_and_two_body_propagations_diverge() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    j2_scenario = scenario.model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )

    two_body = propagate_local(scenario)
    j2 = propagate_local(j2_scenario)

    two_body_final = np.array(two_body.samples[-1].state.position_km)
    j2_final = np.array(j2.samples[-1].state.position_km)
    assert np.linalg.norm(two_body_final - j2_final) > 0.0
