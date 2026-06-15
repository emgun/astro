import pytest

from astro_launch.local import propagate_launch_local
from astro_launch.targeting import sweep_pitch_program, tune_pitch_program
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario


def test_sweep_pitch_program_varies_one_knot_without_mutating_scenario() -> None:
    scenario = make_pitch_program_launch_scenario()

    result = sweep_pitch_program(
        scenario,
        point_index=3,
        pitch_values_deg=[10.0, 20.0, 30.0],
    )

    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert result.scenario_id == scenario.scenario_id
    assert result.point_index == 3
    assert result.point_time_s == 110.0
    assert result.baseline_pitch_deg == 20.0
    assert [case.pitch_deg for case in result.cases] == [10.0, 20.0, 30.0]
    assert result.best_case.pitch_deg in {10.0, 20.0, 30.0}
    assert result.best_case.score == min(case.score for case in result.cases)
    assert all(case.final_downrange_km > 0.0 for case in result.cases)
    assert all("altitude_miss_km" in case.target_miss for case in result.cases)
    assert all("velocity_miss_km_s" in case.target_miss for case in result.cases)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"point_index": 0, "pitch_values_deg": [10.0]}, "cannot sweep first"),
        ({"point_index": 99, "pitch_values_deg": [10.0]}, "point_index"),
        ({"point_index": 3, "pitch_values_deg": []}, "at least one"),
        ({"point_index": 3, "pitch_values_deg": [-1.0]}, "between 0 and 90"),
    ],
)
def test_sweep_pitch_program_rejects_invalid_sweep_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    scenario = make_pitch_program_launch_scenario()

    with pytest.raises(ValueError, match=message):
        sweep_pitch_program(scenario, **kwargs)


def test_sweep_pitch_program_requires_pitch_program_guidance() -> None:
    with pytest.raises(ValueError, match="pitch_program guidance"):
        sweep_pitch_program(make_launch_scenario(), point_index=0, pitch_values_deg=[10.0])


def test_tune_pitch_program_refines_two_knots_and_returns_tuned_scenario() -> None:
    scenario = make_pitch_program_launch_scenario()
    baseline_score = sweep_pitch_program(
        scenario,
        point_index=3,
        pitch_values_deg=[scenario.guidance.pitch_program[3].pitch_deg],
    ).best_case.score

    result = tune_pitch_program(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
    )

    tuned_pitches = {
        point.point_index: point.tuned_pitch_deg for point in result.tuned_points
    }
    assert scenario.guidance.pitch_program[2].pitch_deg == 45.0
    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert result.scenario_id == scenario.scenario_id
    assert result.point_indices == [2, 3]
    assert len(result.iterations) == 2
    assert all(len(iteration.cases) == 9 for iteration in result.iterations)
    assert result.best_case.score <= baseline_score
    assert result.best_case.score == min(
        case.score for iteration in result.iterations for case in iteration.cases
    )
    assert result.tuned_scenario.guidance.pitch_program[2].pitch_deg == tuned_pitches[2]
    assert result.tuned_scenario.guidance.pitch_program[3].pitch_deg == tuned_pitches[3]
    assert propagate_launch_local(result.tuned_scenario).target_miss == result.best_case.target_miss


@pytest.mark.parametrize(
    ("scenario", "kwargs", "message"),
    [
        (
            make_launch_scenario(),
            {"point_indices": (2, 3)},
            "pitch_program guidance",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (3,)},
            "exactly two",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (3, 3)},
            "distinct",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (0, 3)},
            "cannot tune first",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (2, 99)},
            "point_indices",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (2, 3), "initial_span_deg": 0.0},
            "initial_span_deg",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (2, 3), "iterations": 0},
            "iterations",
        ),
        (
            make_pitch_program_launch_scenario(),
            {"point_indices": (2, 3), "refinement_factor": 1.0},
            "refinement_factor",
        ),
    ],
)
def test_tune_pitch_program_rejects_invalid_inputs(
    scenario: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tune_pitch_program(scenario, **kwargs)
