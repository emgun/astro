import pytest

from astro_launch.targeting import sweep_pitch_program
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
