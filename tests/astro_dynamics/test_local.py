from datetime import timedelta
from pathlib import Path

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
    assert final_covariance.shape == (6, 6)
    np.testing.assert_allclose(final_covariance, final_covariance.T, rtol=0.0, atol=1.0e-10)
    assert not np.allclose(final_covariance, np.array(initial_covariance))
    assert trajectory.metadata["covariance_model"] == "finite_difference_state_transition"


def test_propagate_local_rejects_unsupported_local_force_model() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    unsupported_scenario = scenario.model_copy(
        update={
            "force_model": ForceModelConfig(gravity=ForceModelName.OREKIT_HIGH_FIDELITY),
        }
    )

    with pytest.raises(ValueError, match=LOCAL_FORCE_MODEL_ERROR):
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
