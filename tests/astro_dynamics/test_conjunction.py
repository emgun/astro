from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.conjunction import assess_conjunction_screening, screen_conjunction
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


def test_screen_conjunction_estimates_probability_from_covariance_history() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))

    result = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.02,
        probability_method="density",
    )

    assert result.probability_of_collision is not None
    assert 0.0 < result.probability_of_collision < 1.0
    assert result.hard_body_radius_km == 0.02
    assert result.metadata["probability_model"] == "encounter_plane_gaussian_density"
    assert result.metadata["covariance_source"] == "trajectory_covariance_history"
    assert result.metadata["combined_position_covariance_km2"] == [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]


def test_screen_conjunction_integrates_encounter_plane_probability() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))

    density_result = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.5,
        probability_method="density",
    )
    integrated_result = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.5,
        probability_method="integrated",
    )

    assert integrated_result.probability_of_collision is not None
    assert density_result.probability_of_collision is not None
    assert 0.0 < integrated_result.probability_of_collision < 1.0
    assert integrated_result.probability_of_collision < density_result.probability_of_collision
    assert integrated_result.metadata["probability_model"] == (
        "encounter_plane_gaussian_integral"
    )
    assert integrated_result.metadata["probability_quadrature"] == "gauss_legendre_polar"


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


def test_assess_conjunction_screening_marks_covariance_close_approach_for_review() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7000.05, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    screening = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.5,
        probability_method="integrated",
    )

    report = assess_conjunction_screening(screening)

    assert report.assessment_status == "requires_review"
    assert report.screening_status == "below_threshold"
    assert report.has_collision_probability is True
    assert report.probability_model == "encounter_plane_gaussian_integral"
    assert [check.check_id for check in report.checks] == [
        "miss_distance_above_threshold",
        "collision_probability_available",
        "integrated_probability_model",
    ]
    assert report.checks[0].passed is False
    assert report.metadata["workflow"] == "conjunction_screening_assessment"


def test_assess_conjunction_screening_marks_geometry_only_result_as_screening_only() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7002.0, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    screening = screen_conjunction(primary, secondary, threshold_km=1.0)

    report = assess_conjunction_screening(screening)

    assert report.assessment_status == "screening_only"
    assert report.screening_status == "above_threshold"
    assert report.has_collision_probability is False
    assert report.probability_model is None
    assert report.checks[1].check_id == "collision_probability_available"
    assert report.checks[1].passed is False


def test_assess_conjunction_screening_marks_clear_covariance_approach_candidate() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    primary = propagate_local(scenario)
    secondary_state = scenario.initial_state.model_copy(
        update={
            "cartesian": scenario.initial_state.cartesian.model_copy(
                update={"position_km": (7002.0, 0.0, 0.0)}
            )
        }
    )
    secondary = propagate_local(scenario.model_copy(update={"initial_state": secondary_state}))
    screening = screen_conjunction(
        primary,
        secondary,
        threshold_km=1.0,
        hard_body_radius_km=0.02,
        probability_method="integrated",
    )

    report = assess_conjunction_screening(screening)

    assert report.assessment_status == "operational_candidate"
    assert report.screening_status == "above_threshold"
    assert report.has_collision_probability is True
    assert report.probability_model == "encounter_plane_gaussian_integral"
    assert [check.passed for check in report.checks] == [True, True, True]
