from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName
from astro_dynamics.local import (
    j2_acceleration_km_s2,
    propagate_local,
    two_body_acceleration_km_s2,
)


def test_two_body_acceleration_points_toward_origin() -> None:
    acceleration = two_body_acceleration_km_s2(np.array([7000.0, 0.0, 0.0]))

    assert acceleration[0] < 0.0
    assert acceleration[1] == 0.0
    assert acceleration[2] == 0.0


def test_j2_acceleration_is_nonzero_for_inclined_state() -> None:
    acceleration = j2_acceleration_km_s2(np.array([5000.0, 3000.0, 4000.0]))

    assert np.linalg.norm(acceleration) > 0.0


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
