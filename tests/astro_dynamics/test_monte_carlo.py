from pathlib import Path

import pytest

from astro_core.io import load_scenario
from astro_dynamics.monte_carlo import run_initial_state_monte_carlo


def test_run_initial_state_monte_carlo_is_seeded_and_repeatable() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    first = run_initial_state_monte_carlo(
        scenario,
        cases=3,
        position_sigma_km=0.01,
        velocity_sigma_km_s=0.000001,
        seed=7,
        backend="local",
    )
    second = run_initial_state_monte_carlo(
        scenario,
        cases=3,
        position_sigma_km=0.01,
        velocity_sigma_km_s=0.000001,
        seed=7,
        backend="local",
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.scenario_id == "leo-two-body"
    assert first.backend == "local"
    assert first.seed == 7
    assert len(first.cases) == 3
    assert {case.trajectory.backend for case in first.cases} == {"local"}
    assert first.cases[0].position_delta_km != (0.0, 0.0, 0.0)


def test_run_initial_state_monte_carlo_rejects_invalid_inputs() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    with pytest.raises(ValueError, match="cases must be positive"):
        run_initial_state_monte_carlo(
            scenario,
            cases=0,
            position_sigma_km=0.01,
            velocity_sigma_km_s=0.000001,
            seed=7,
        )

    with pytest.raises(ValueError, match="sigmas must be nonnegative"):
        run_initial_state_monte_carlo(
            scenario,
            cases=1,
            position_sigma_km=-0.01,
            velocity_sigma_km_s=0.000001,
            seed=7,
        )
