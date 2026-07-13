from pathlib import Path

import pytest

from astro_core.io import load_scenario
from astro_core.models import CartesianState
from astro_dynamics.backends import propagate_with_backend
from astro_dynamics.monte_carlo import run_initial_state_monte_carlo


@pytest.fixture
def legacy_seed_7_deltas() -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
]:
    return (
        (
            (1.2301533574825743e-05, 0.002987455375084699, -0.002741378553622176),
            (-8.905918387572742e-07, -4.546707851717225e-07, -9.916465549964623e-07),
        ),
        (
            (0.0006014360259743849, 0.013402152455545336, -0.004922065185513296),
            (-6.204748998199404e-07, 4.898420501851982e-07, 3.568870081600607e-07),
        ),
        (
            (0.0010541424899789857, -0.009304680447082046, -0.0002925182246327349),
            (6.953031944582878e-07, -1.344214547285082e-06, -4.5761576104021815e-07),
        ),
    )


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


def test_run_initial_state_monte_carlo_preserves_legacy_numpy_golden_result(
    legacy_seed_7_deltas: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    result = run_initial_state_monte_carlo(
        scenario,
        cases=3,
        position_sigma_km=0.01,
        velocity_sigma_km_s=0.000001,
        seed=7,
        backend="local",
    )

    assert (
        tuple((case.position_delta_km, case.velocity_delta_km_s) for case in result.cases)
        == legacy_seed_7_deltas
    )
    for case, (position_delta, velocity_delta) in zip(
        result.cases, legacy_seed_7_deltas, strict=True
    ):
        assert case.initial_state.cartesian.position_km == tuple(
            base + delta
            for base, delta in zip(
                scenario.initial_state.cartesian.position_km, position_delta, strict=True
            )
        )
        assert case.initial_state.cartesian.velocity_km_s == tuple(
            base + delta
            for base, delta in zip(
                scenario.initial_state.cartesian.velocity_km_s, velocity_delta, strict=True
            )
        )
        legacy_cartesian = CartesianState(
            position_km=case.initial_state.cartesian.position_km,
            velocity_km_s=case.initial_state.cartesian.velocity_km_s,
        )
        legacy_initial_state = scenario.initial_state.model_copy(
            update={"cartesian": legacy_cartesian}
        )
        legacy_scenario = scenario.model_copy(update={"initial_state": legacy_initial_state})
        assert case.trajectory.model_dump(mode="json") == propagate_with_backend(
            legacy_scenario, "local"
        ).model_dump(mode="json")
    assert result.metadata == {
        "uq_campaign_kernel": "astro_uq.evaluators.evaluate_authoritatively",
        "uq_campaign_kernel_version": "1.0",
    }


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
