import pytest

from astro_launch.reporting import generate_tuned_launch_report
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario


def test_generate_tuned_launch_report_runs_tune_launch_handoff_and_orbit_arc() -> None:
    scenario = make_pitch_program_launch_scenario()

    report = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    assert scenario.guidance.pitch_program[2].pitch_deg == 45.0
    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert report.scenario_id == scenario.scenario_id
    assert report.tuning_result.point_indices == [2, 3]
    assert report.launch_trajectory.scenario_id == scenario.scenario_id
    assert report.launch_trajectory.metadata["guidance_mode"] == "pitch_program"
    assert report.orbit_scenario.initial_state == report.launch_trajectory.insertion_state
    assert report.orbit_scenario.metadata["workflow"] == "launch_orbit_handoff"
    assert len(report.orbit_trajectory.samples) == 11
    assert report.insertion_metrics.altitude_miss_km == report.launch_trajectory.target_miss[
        "altitude_miss_km"
    ]
    assert report.insertion_metrics.velocity_miss_km_s == report.launch_trajectory.target_miss[
        "velocity_miss_km_s"
    ]
    assert report.short_arc_metrics.sample_count == 11
    assert report.short_arc_metrics.duration_s == 600.0
    assert report.short_arc_metrics.final_altitude_km == pytest.approx(
        report.short_arc_metrics.altitudes_km[-1]
    )
    assert report.short_arc_metrics.final_altitude_miss_km == pytest.approx(
        report.short_arc_metrics.final_altitude_km - scenario.target_orbit.altitude_km
    )


def test_generate_tuned_launch_report_requires_pitch_program_guidance() -> None:
    with pytest.raises(ValueError, match="pitch_program guidance"):
        generate_tuned_launch_report(make_launch_scenario(), point_indices=(2, 3))
