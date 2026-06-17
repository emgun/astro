from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.conjunction import screen_conjunction
from astro_dynamics.local import propagate_local


def test_screen_conjunction_reports_time_aligned_closest_approach() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.5, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))

    result = screen_conjunction(primary, secondary, threshold_km=1.0)

    assert result.primary_scenario_id == "leo-two-body"
    assert result.secondary_scenario_id == "leo-two-body"
    assert result.compared_sample_count == len(primary.samples)
    assert result.tca_sample_index == 0
    assert result.miss_distance_km == 0.5
    assert result.relative_speed_km_s == 0.0
    assert result.status == "below_threshold"
    assert result.metadata["screening_model"] == "time_aligned_sample_minimum_distance"


def test_screen_conjunction_rejects_trajectories_without_common_epochs() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    primary = propagate_local(scenario)
    shifted_initial_state = scenario.initial_state.model_copy(
        update={"epoch": scenario.initial_state.epoch.replace(year=2027)}
    )
    secondary = propagate_local(
        scenario.model_copy(update={"initial_state": shifted_initial_state})
    )

    try:
        screen_conjunction(primary, secondary, threshold_km=1.0)
    except ValueError as exc:
        assert "common sample epochs" in str(exc)
    else:
        raise AssertionError("screen_conjunction should reject trajectories without common epochs")
