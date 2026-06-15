import pytest

from astro_launch.reporting import (
    compare_tuned_launch_reports,
    generate_tuned_launch_report,
    generate_tuned_launch_report_batch,
)
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
    assert report.passed is False
    assert report.insertion_assessment.passed is False
    assert report.short_arc_assessment.passed is False
    assert [check.name for check in report.insertion_assessment.checks] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
    ]
    assert [check.name for check in report.short_arc_assessment.checks] == [
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]
    assert report.insertion_assessment.checks[0].tolerance == pytest.approx(
        scenario.target_orbit.altitude_tolerance_km
    )
    assert report.insertion_assessment.checks[1].tolerance == pytest.approx(
        scenario.target_orbit.velocity_tolerance_km_s
    )


def test_generate_tuned_launch_report_passes_with_loose_target_tolerances() -> None:
    scenario = make_pitch_program_launch_scenario()
    loose_target = scenario.target_orbit.model_copy(
        update={
            "altitude_tolerance_km": 5000.0,
            "velocity_tolerance_km_s": 20.0,
        }
    )
    loose_scenario = scenario.model_copy(update={"target_orbit": loose_target})

    report = generate_tuned_launch_report(
        loose_scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    assert report.passed is True
    assert report.insertion_assessment.passed is True
    assert report.short_arc_assessment.passed is True
    assert all(check.passed for check in report.insertion_assessment.checks)
    assert all(check.passed for check in report.short_arc_assessment.checks)


def test_compare_tuned_launch_reports_summarizes_pass_and_metric_deltas() -> None:
    scenario = make_pitch_program_launch_scenario()
    baseline = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=1,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )
    candidate = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    comparison = compare_tuned_launch_reports(baseline, candidate)

    assert comparison.baseline_scenario_id == baseline.scenario_id
    assert comparison.candidate_scenario_id == candidate.scenario_id
    assert comparison.baseline_passed == baseline.passed
    assert comparison.candidate_passed == candidate.passed
    assert [metric.name for metric in comparison.metric_deltas] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]
    insertion_altitude = comparison.metric_deltas[0]
    assert insertion_altitude.baseline_value == pytest.approx(
        baseline.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.candidate_value == pytest.approx(
        candidate.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.delta == pytest.approx(
        candidate.insertion_metrics.altitude_miss_km
        - baseline.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.improvement == pytest.approx(
        abs(baseline.insertion_metrics.altitude_miss_km)
        - abs(candidate.insertion_metrics.altitude_miss_km)
    )


def test_generate_tuned_launch_report_batch_ranks_iteration_values() -> None:
    scenario = make_pitch_program_launch_scenario()

    batch = generate_tuned_launch_report_batch(
        scenario,
        point_indices=(2, 3),
        iterations_values=(1, 2),
        initial_span_deg=10.0,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    assert batch.scenario_id == scenario.scenario_id
    assert batch.point_indices == [2, 3]
    assert {case.iterations for case in batch.cases} == {1, 2}
    assert [case.rank for case in batch.cases] == [1, 2]
    assert batch.best_case == batch.cases[0]
    assert batch.best_case.normalized_score == min(
        case.normalized_score for case in batch.cases
    )
    for case in batch.cases:
        checks = [
            *case.report.insertion_assessment.checks,
            *case.report.short_arc_assessment.checks,
        ]
        assert case.normalized_score == pytest.approx(
            sum(abs(check.value) / check.tolerance for check in checks)
        )
        assert case.label == f"iterations={case.iterations}"


def test_generate_tuned_launch_report_requires_pitch_program_guidance() -> None:
    with pytest.raises(ValueError, match="pitch_program guidance"):
        generate_tuned_launch_report(make_launch_scenario(), point_indices=(2, 3))
