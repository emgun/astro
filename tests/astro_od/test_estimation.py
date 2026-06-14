from pathlib import Path

import numpy as np

from astro_core.errors import NumericalConvergenceError
from astro_core.io import load_scenario
from astro_core.models import CartesianState, GroundStation, Scenario
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements


def _observable_scenario() -> Scenario:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    station = GroundStation(
        name="north-eci",
        position_eci_km=(0.0, 6378.1363, 0.0),
        frame="EME2000",
        elevation_mask_deg=0.0,
    )
    return scenario.model_copy(update={"ground_stations": (*scenario.ground_stations, station)})


def test_batch_od_recovers_synthetic_initial_state() -> None:
    truth_scenario = _observable_scenario()
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)

    perturbed_state = truth_scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=(7001.0, -0.8, 0.6),
                velocity_km_s=(0.0005, 7.499, 1.0008),
            )
        }
    )
    estimate_scenario = truth_scenario.model_copy(update={"initial_state": perturbed_state})

    result = estimate_initial_state(estimate_scenario, measurements)

    truth_position = truth_scenario.initial_state.cartesian.position_array()
    estimated_position = result.estimated_state.cartesian.position_array()
    truth_velocity = truth_scenario.initial_state.cartesian.velocity_array()
    estimated_velocity = result.estimated_state.cartesian.velocity_array()

    assert result.converged is True
    assert np.linalg.norm(estimated_position - truth_position) < 0.2
    assert np.linalg.norm(estimated_velocity - truth_velocity) < 2.0e-4
    assert result.rms < 3.0
    assert len(result.covariance) == 6


def test_estimate_initial_state_requires_measurements() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    try:
        estimate_initial_state(scenario, [])
    except NumericalConvergenceError as exc:
        assert str(exc) == "At least one measurement is required for estimation"
    else:
        raise AssertionError("estimate_initial_state should reject empty measurements")
