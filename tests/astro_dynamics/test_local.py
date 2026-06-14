from pathlib import Path

import numpy as np
import pytest

from astro_core.constants import MU_EARTH_KM3_S2
from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName
from astro_dynamics.local import (
    acceleration_km_s2,
    j2_acceleration_km_s2,
    propagate_local,
    rk4_step,
    two_body_acceleration_km_s2,
)

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
