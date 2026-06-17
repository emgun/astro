from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astro_core.io import load_scenario
from astro_core.models import (
    CartesianState,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
    Scenario,
)
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements

MU_EARTH_KM3_S2 = 398600.4418
REFERENCE_SCENARIO = Path("examples/scenarios/leo_two_body.yaml")
EXPECTED_J2_FINAL_POSITION_DELTA_KM = np.array(
    [-2.0115772, -0.47571099, -0.17144279],
    dtype=np.float64,
)


def _specific_energy_km2_s2(
    position_km: NDArray[np.float64],
    velocity_km_s: NDArray[np.float64],
) -> float:
    return float(
        0.5 * np.dot(velocity_km_s, velocity_km_s)
        - MU_EARTH_KM3_S2 / np.linalg.norm(position_km)
    )


def _reference_scenario() -> Scenario:
    return load_scenario(REFERENCE_SCENARIO)


def _observable_scenario() -> Scenario:
    scenario = _reference_scenario()
    station = GroundStation(
        name="reference-y-axis-eci",
        position_eci_km=(0.0, 6378.1363, 0.0),
        frame=Frame.EME2000,
        elevation_mask_deg=0.0,
    )
    return scenario.model_copy(update={"ground_stations": [*scenario.ground_stations, station]})


def _perturbed_scenario(scenario: Scenario) -> Scenario:
    initial = scenario.initial_state.cartesian
    position_perturbation = np.array([1.0, -0.8, 0.6], dtype=np.float64)
    velocity_perturbation = np.array([0.0005, -0.001, 0.0008], dtype=np.float64)
    perturbed_cartesian = CartesianState(
        position_km=tuple(initial.position_array() + position_perturbation),
        velocity_km_s=tuple(initial.velocity_array() + velocity_perturbation),
    )
    perturbed_state = scenario.initial_state.model_copy(update={"cartesian": perturbed_cartesian})
    return scenario.model_copy(update={"initial_state": perturbed_state})


def test_two_body_specific_energy_is_stable_over_short_arc() -> None:
    scenario = _reference_scenario()

    trajectory = propagate_local(scenario)

    first = trajectory.samples[0].state
    last = trajectory.samples[-1].state
    first_energy = _specific_energy_km2_s2(first.position_array(), first.velocity_array())
    last_energy = _specific_energy_km2_s2(last.position_array(), last.velocity_array())

    assert abs(last_energy - first_energy) < 1.0e-7


def test_local_reference_metadata_discloses_internal_integration_cadence() -> None:
    scenario = _reference_scenario()

    trajectory = propagate_local(scenario)

    assert trajectory.metadata == {
        "integrator": "rk4",
        "step_s": 60.0,
        "sample_step_s": 60.0,
        "internal_max_step_s": 30.0,
        "internal_substeps_per_sample": 2,
        "internal_step_s": 30.0,
        "event_detection": "radial_velocity_root",
        "apsis_event_count": 1,
    }


def test_j2_reference_case_diverges_from_two_body_without_breaking_energy_scale() -> None:
    scenario = _reference_scenario()
    j2_scenario = scenario.model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )

    two_body = propagate_local(scenario)
    j2 = propagate_local(j2_scenario)

    position_delta_km = (
        j2.samples[-1].state.position_array() - two_body.samples[-1].state.position_array()
    )

    np.testing.assert_allclose(
        position_delta_km,
        EXPECTED_J2_FINAL_POSITION_DELTA_KM,
        rtol=0.0,
        atol=1.0e-3,
    )


def test_synthetic_od_reference_case_converges() -> None:
    truth_scenario = _observable_scenario()
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)
    estimate_scenario = _perturbed_scenario(truth_scenario)

    result = estimate_initial_state(estimate_scenario, measurements)
    truth_position = truth_scenario.initial_state.cartesian.position_array()
    estimated_position = result.estimated_state.cartesian.position_array()
    truth_velocity = truth_scenario.initial_state.cartesian.velocity_array()
    estimated_velocity = result.estimated_state.cartesian.velocity_array()

    assert result.converged is True
    assert np.linalg.norm(estimated_position - truth_position) < 0.2
    assert np.linalg.norm(estimated_velocity - truth_velocity) < 2.0e-4
    assert result.rms < 3.0
    assert len(result.residuals) == len(measurements)
    assert result.metadata["jacobian_rank"] == 6
