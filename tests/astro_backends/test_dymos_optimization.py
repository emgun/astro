import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_backends.dymos.optimization import (
    DymosPhaseSummary,
    DymosPitchProgramSummary,
    optimize_launch_dymos,
    run_dymos_multistage_pitch_program_optimization,
    run_dymos_pitch_program_optimization,
)
from astro_backends.dymos.runtime import DymosRuntime
from astro_core.errors import UnsupportedBackendError
from astro_launch.io import load_launch_scenario
from astro_launch.models import LaunchPitchTuningResult, LaunchScenario
from astro_launch.targeting import tune_pitch_program
from tests.astro_launch.helpers import make_pitch_program_launch_scenario


def _fake_runtime() -> DymosRuntime:
    return DymosRuntime(
        dymos_version="1.15.1",
        openmdao_version="3.44.0",
        dymos_module=SimpleNamespace(),
        openmdao_module=SimpleNamespace(),
        trajectory=object,
        phase=object,
        transcription=object,
        problem=object,
    )


def test_optimize_launch_dymos_reports_runtime_unavailable() -> None:
    def fail_runtime() -> DymosRuntime:
        raise UnsupportedBackendError(
            "Dymos backend unavailable: install astro-suite[optimization]"
        )

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[optimization\]"):
        optimize_launch_dymos(
            make_pitch_program_launch_scenario(),
            runtime_loader=fail_runtime,
        )


def test_optimize_launch_dymos_runs_default_phase_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_pitch_program_launch_scenario()
    seen_runtime: list[DymosRuntime] = []

    def fake_phase_solver(candidate: LaunchScenario, runtime: DymosRuntime) -> DymosPhaseSummary:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return DymosPhaseSummary(
            phase_model="stage_aware_vertical_ascent",
            transcription="GaussLobatto",
            num_segments=3,
            order=3,
            duration_s=120.0,
            stage_count=2,
            total_burn_duration_s=120.0,
            final_altitude_km=1.2,
            final_velocity_km_s=0.08,
            optimizer_success=True,
            optimizer_message="Optimization terminated successfully",
        )

    monkeypatch.setattr(
        "astro_backends.dymos.optimization._solve_dymos_vertical_phase",
        fake_phase_solver,
    )

    result = optimize_launch_dymos(scenario, runtime_loader=_fake_runtime)

    assert len(seen_runtime) == 1
    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_phase"
    tuned_pitch_by_index = {
        point.point_index: point.tuned_pitch_deg for point in result.tuned_points
    }
    assert result.metadata["dymos_phase"] == {
        "phase_model": "stage_aware_vertical_ascent",
        "transcription": "GaussLobatto",
        "num_segments": 3,
        "order": 3,
        "duration_s": 120.0,
        "stage_count": 2,
        "total_burn_duration_s": 120.0,
        "final_altitude_km": 1.2,
        "final_velocity_km_s": 0.08,
        "optimizer_success": True,
        "optimizer_message": "Optimization terminated successfully",
        "guidance_model": "pitch_program",
        "pitch_program_control_points": [
            {"index": 0, "time_s": 0.0, "pitch_deg": 90.0, "tuned": False},
            {"index": 1, "time_s": 30.0, "pitch_deg": 75.0, "tuned": False},
            {"index": 2, "time_s": 70.0, "pitch_deg": 45.0, "tuned": True},
            {"index": 3, "time_s": 110.0, "pitch_deg": 20.0, "tuned": True},
            {"index": 4, "time_s": 140.0, "pitch_deg": 5.0, "tuned": False},
        ],
        "optimized_pitch_program_control_points": [
            {
                "index": 0,
                "time_s": 0.0,
                "baseline_pitch_deg": 90.0,
                "pitch_deg": 90.0,
                "tuned": False,
                "source": "baseline",
            },
            {
                "index": 1,
                "time_s": 30.0,
                "baseline_pitch_deg": 75.0,
                "pitch_deg": 75.0,
                "tuned": False,
                "source": "baseline",
            },
            {
                "index": 2,
                "time_s": 70.0,
                "baseline_pitch_deg": 45.0,
                "pitch_deg": tuned_pitch_by_index[2],
                "tuned": True,
                "source": "suite_pitch_tuning",
            },
            {
                "index": 3,
                "time_s": 110.0,
                "baseline_pitch_deg": 20.0,
                "pitch_deg": tuned_pitch_by_index[3],
                "tuned": True,
                "source": "suite_pitch_tuning",
            },
            {
                "index": 4,
                "time_s": 140.0,
                "baseline_pitch_deg": 5.0,
                "pitch_deg": 5.0,
                "tuned": False,
                "source": "baseline",
            },
        ],
        "pitch_program_optimization_coupling": "dymos_phase_plus_suite_pitch_tuning",
        "pitch_program_optimization_scope": "suite_tuning_not_full_dymos_pitch_transcription",
    }
    assert result.metadata["dymos_tuned_pitch_point_indices"] == [2, 3]
    assert result.metadata["dymos_pitch_program_transcription_contract"] == {
        "execution_status": "not_executed",
        "phase_coupling": "dymos_phase_plus_suite_pitch_tuning",
        "control_name": "pitch_deg",
        "control_units": "deg",
        "control_bounds_deg": {"lower": 0.0, "upper": 90.0},
        "control_point_count": 5,
        "tuned_control_point_indices": [2, 3],
        "control_schedule_covers_stage_plan": True,
        "transcription": "GaussLobatto",
        "stage_phase_count": 2,
        "stage_phases": [
            {
                "name": "stage-1",
                "start_s": 0.0,
                "burnout_s": 70.0,
                "control_point_indices": [0, 1, 2],
            },
            {
                "name": "stage-2",
                "start_s": 70.0,
                "burnout_s": 120.0,
                "control_point_indices": [2, 3, 4],
            },
        ],
        "optimized_control_points": [
            {
                "index": 0,
                "time_s": 0.0,
                "baseline_pitch_deg": 90.0,
                "pitch_deg": 90.0,
                "tuned": False,
                "source": "baseline",
            },
            {
                "index": 1,
                "time_s": 30.0,
                "baseline_pitch_deg": 75.0,
                "pitch_deg": 75.0,
                "tuned": False,
                "source": "baseline",
            },
            {
                "index": 2,
                "time_s": 70.0,
                "baseline_pitch_deg": 45.0,
                "pitch_deg": tuned_pitch_by_index[2],
                "tuned": True,
                "source": "suite_pitch_tuning",
            },
            {
                "index": 3,
                "time_s": 110.0,
                "baseline_pitch_deg": 20.0,
                "pitch_deg": tuned_pitch_by_index[3],
                "tuned": True,
                "source": "suite_pitch_tuning",
            },
            {
                "index": 4,
                "time_s": 140.0,
                "baseline_pitch_deg": 5.0,
                "pitch_deg": 5.0,
                "tuned": False,
                "source": "baseline",
            },
        ],
    }
    assert result.metadata["stage_plan"]["total_burn_duration_s"] == 120.0
    assert result.metadata["dymos_phase_covers_stage_schedule"] is True


def test_optimize_launch_dymos_returns_suite_product_with_fake_runner() -> None:
    scenario = make_pitch_program_launch_scenario()
    seen_runtime: list[DymosRuntime] = []

    def fake_runner(candidate: LaunchScenario, runtime: DymosRuntime) -> LaunchPitchTuningResult:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return tune_pitch_program(candidate, point_indices=(2, 3), iterations=1)

    result = optimize_launch_dymos(
        scenario,
        runtime_loader=_fake_runtime,
        optimizer_runner=fake_runner,
    )

    assert len(seen_runtime) == 1
    assert result.backend == "dymos"
    assert result.metadata["adapter"] == "dymos"
    assert result.metadata["source_backend"] == "local"
    assert result.metadata["dymos_version"] == "1.15.1"
    assert result.metadata["openmdao_version"] == "3.44.0"
    assert result.metadata["optimizer_status"] == "completed"
    assert result.metadata["converged"] is True
    assert result.metadata["iteration_count"] == len(result.iterations)
    assert result.metadata["candidate_count"] == result.metadata["source_candidate_count"]
    assert result.metadata["best_score"] == result.best_case.score
    assert result.metadata["target_insertion_residuals"] == result.best_case.target_miss
    assessment = result.metadata["target_insertion_assessment"]
    assert assessment["target_altitude_km"] == scenario.target_orbit.altitude_km
    assert assessment["tolerances"] == {
        "altitude_tolerance_km": scenario.target_orbit.altitude_tolerance_km,
        "velocity_tolerance_km_s": scenario.target_orbit.velocity_tolerance_km_s,
        "radial_velocity_tolerance_km_s": (
            scenario.target_orbit.radial_velocity_tolerance_km_s
        ),
    }
    assert assessment["residuals"] == result.best_case.target_miss
    assert assessment["objective"] == (
        "minimize_weighted_altitude_velocity_radial_insertion_residuals"
    )
    assert result.metadata["target_insertion_satisfied"] == assessment["satisfied"]
    assert result.metadata["stage_plan"] == {
        "stage_count": 2,
        "total_burn_duration_s": 120.0,
        "stages": [
            {"name": "stage-1", "start_s": 0.0, "burnout_s": 70.0},
            {"name": "stage-2", "start_s": 70.0, "burnout_s": 120.0},
        ],
    }
    assert result.metadata["multistage"] is True
    assert result.metadata["dymos_phase_covers_stage_schedule"] is None
    assert result.metadata["path_constraints"] == {
        "pitch_deg": {"lower": 0.0, "upper": 90.0},
        "pitch_program_control_points": {"count": 5, "tuned_indices": [2, 3]},
    }
    assert result.metadata["dymos_pitch_program_transcription_contract"][
        "execution_status"
    ] == "not_executed"
    assert result.metadata["dymos_pitch_program_transcription_contract"][
        "stage_phase_count"
    ] == 2


def test_optimize_launch_dymos_runs_native_pitch_program_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_pitch_program_launch_scenario()
    seen_runtime: list[DymosRuntime] = []

    def fake_pitch_solver(
        candidate: LaunchScenario,
        runtime: DymosRuntime,
    ) -> DymosPitchProgramSummary:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return DymosPitchProgramSummary(
            phase_model="stage_aware_pitch_program_ascent",
            transcription="GaussLobatto",
            num_segments=4,
            order=3,
            duration_s=120.0,
            stage_count=2,
            total_burn_duration_s=120.0,
            final_altitude_km=155.0,
            final_velocity_km_s=7.65,
            final_radial_velocity_km_s=0.05,
            final_horizontal_velocity_km_s=7.64,
            final_downrange_km=480.0,
            target_miss={
                "altitude_miss_km": -5.0,
                "velocity_miss_km_s": 0.05,
            },
            optimized_pitch_deg_by_point_index={2: 42.0, 3: 18.0},
            optimizer_success=True,
            optimizer_message="Optimization terminated successfully",
        )

    monkeypatch.setattr(
        "astro_backends.dymos.optimization._solve_dymos_pitch_program_phase",
        fake_pitch_solver,
    )

    result = optimize_launch_dymos(
        scenario,
        runtime_loader=_fake_runtime,
        optimizer_runner=run_dymos_pitch_program_optimization,
    )

    assert len(seen_runtime) == 1
    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_pitch_program"
    assert result.metadata["optimizer_status"] == "completed"
    assert result.metadata["converged"] is True
    assert result.best_case.pitch_deg_by_point_index == {"2": 42.0, "3": 18.0}
    assert result.best_case.target_miss == {
        "altitude_miss_km": -5.0,
        "velocity_miss_km_s": 0.05,
        "radial_velocity_miss_km_s": 0.05,
    }
    assert result.metadata["target_insertion_satisfied"] is True
    assert result.metadata["target_insertion_assessment"]["status"] == "within_tolerance"
    assert result.metadata["target_insertion_assessment"]["component_status"] == {
        "altitude": "within_tolerance",
        "velocity": "within_tolerance",
        "radial_velocity": "within_tolerance",
    }
    dymos_phase = result.metadata["dymos_phase"]
    assert dymos_phase["phase_model"] == "stage_aware_pitch_program_ascent"
    assert dymos_phase["pitch_program_optimization_coupling"] == (
        "native_dymos_pitch_control"
    )
    assert dymos_phase["pitch_program_optimization_scope"] == (
        "native_dymos_pitch_program_transcription"
    )
    assert dymos_phase["target_objective"] == (
        "minimize_final_normalized_target_insertion_error"
    )
    assert dymos_phase["target_score"] > 0.0
    assert set(dymos_phase["target_score_terms"]) == {
        "altitude",
        "velocity",
        "radial_velocity",
    }
    contract = result.metadata["dymos_pitch_program_transcription_contract"]
    assert contract["execution_status"] == "executed"
    assert contract["phase_coupling"] == "native_dymos_pitch_control"
    assert contract["optimized_control_points"][2]["pitch_deg"] == 42.0
    assert contract["optimized_control_points"][2]["source"] == "dymos_pitch_program_control"
    assert contract["optimized_control_points"][3]["pitch_deg"] == 18.0
    assert contract["optimized_control_points"][3]["source"] == "dymos_pitch_program_control"
    assert result.metadata["dymos_tuned_pitch_point_indices"] == [2, 3]


def test_optimize_launch_dymos_runs_native_multistage_pitch_program_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_pitch_program_launch_scenario()
    seen_runtime: list[DymosRuntime] = []

    def fake_multistage_solver(
        candidate: LaunchScenario,
        runtime: DymosRuntime,
    ) -> DymosPitchProgramSummary:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return DymosPitchProgramSummary(
            phase_model="multiphase_stage_pitch_program_ascent",
            transcription="GaussLobatto",
            num_segments=4,
            order=3,
            duration_s=120.0,
            stage_count=2,
            total_burn_duration_s=120.0,
            final_altitude_km=156.0,
            final_velocity_km_s=7.66,
            final_radial_velocity_km_s=0.04,
            final_horizontal_velocity_km_s=7.65,
            final_downrange_km=490.0,
            target_miss={
                "altitude_miss_km": -4.0,
                "velocity_miss_km_s": 0.04,
            },
            optimized_pitch_deg_by_point_index={2: 41.0, 3: 17.0},
            optimizer_success=True,
            optimizer_message="Optimization terminated successfully",
            phase_count=2,
            phase_topology="multiphase_stage_linked",
            linked_state_names=("time", "h", "downrange", "vr", "vh"),
            stage_phase_summaries=(
                {
                    "name": "stage-1",
                    "phase_name": "stage_1",
                    "duration_s": 70.0,
                    "final_altitude_km": 28.0,
                    "final_velocity_km_s": 1.8,
                },
                {
                    "name": "stage-2",
                    "phase_name": "stage_2",
                    "duration_s": 50.0,
                    "final_altitude_km": 156.0,
                    "final_velocity_km_s": 7.66,
                },
            ),
            source_backend="dymos_multistage_pitch_program",
            pitch_program_optimization_coupling="native_dymos_multiphase_pitch_control",
            pitch_program_optimization_scope="native_dymos_multiphase_stage_transcription",
        )

    monkeypatch.setattr(
        "astro_backends.dymos.optimization._solve_dymos_multistage_pitch_program_phases",
        fake_multistage_solver,
    )

    result = optimize_launch_dymos(
        scenario,
        runtime_loader=_fake_runtime,
        optimizer_runner=run_dymos_multistage_pitch_program_optimization,
    )

    assert len(seen_runtime) == 1
    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_multistage_pitch_program"
    dymos_phase = result.metadata["dymos_phase"]
    assert dymos_phase["phase_model"] == "multiphase_stage_pitch_program_ascent"
    assert dymos_phase["phase_count"] == 2
    assert dymos_phase["phase_topology"] == "multiphase_stage_linked"
    assert dymos_phase["linked_state_names"] == ("time", "h", "downrange", "vr", "vh")
    assert dymos_phase["stage_phase_summaries"][0]["phase_name"] == "stage_1"
    assert dymos_phase["stage_phase_summaries"][0]["duration_s"] == 70.0
    assert dymos_phase["stage_phase_summaries"][1]["phase_name"] == "stage_2"
    assert dymos_phase["stage_phase_summaries"][1]["duration_s"] == 50.0
    assert dymos_phase["pitch_program_optimization_coupling"] == (
        "native_dymos_multiphase_pitch_control"
    )
    assert dymos_phase["pitch_program_optimization_scope"] == (
        "native_dymos_multiphase_stage_transcription"
    )
    contract = result.metadata["dymos_pitch_program_transcription_contract"]
    assert contract["execution_status"] == "executed"
    assert contract["phase_coupling"] == "native_dymos_multiphase_pitch_control"
    assert contract["phase_topology"] == "multiphase_stage_linked"
    assert contract["native_stage_phase_count"] == 2
    assert contract["linked_state_names"] == ("time", "h", "downrange", "vr", "vh")
    assert contract["optimized_control_points"][2]["pitch_deg"] == 41.0
    assert contract["optimized_control_points"][2]["source"] == "dymos_pitch_program_control"
    assert contract["optimized_control_points"][3]["pitch_deg"] == 17.0
    assert contract["optimized_control_points"][3]["source"] == "dymos_pitch_program_control"


@pytest.mark.dymos_live
def test_live_dymos_optimization_returns_suite_product() -> None:
    if os.environ.get("ASTRO_RUN_DYMOS_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_DYMOS_LIVE=1 to run live Dymos launch optimization")
    pytest.importorskip("dymos")
    pytest.importorskip("openmdao")

    scenario = load_launch_scenario(Path("examples/launch/pitch_program_two_stage.yaml"))

    result = optimize_launch_dymos(scenario)

    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_phase"
    assert result.metadata["adapter"] == "dymos"
    assert result.metadata["converged"] is True
    assert result.metadata["stage_plan"]["stage_count"] == 2
    assert result.metadata["multistage"] is True
    assert result.metadata["dymos_phase_covers_stage_schedule"] is True
    assert result.metadata["path_constraints"] == {
        "pitch_deg": {"lower": 0.0, "upper": 90.0},
        "pitch_program_control_points": {"count": 5, "tuned_indices": [2, 3]},
    }
    assert result.metadata["dymos_tuned_pitch_point_indices"] == [2, 3]
    dymos_phase = result.metadata["dymos_phase"]
    assert dymos_phase["phase_model"] == "stage_aware_vertical_ascent"
    assert dymos_phase["transcription"] == "GaussLobatto"
    assert dymos_phase["guidance_model"] == "pitch_program"
    assert len(dymos_phase["pitch_program_control_points"]) == 5
    assert len(dymos_phase["optimized_pitch_program_control_points"]) == 5
    assert dymos_phase["pitch_program_optimization_coupling"] == (
        "dymos_phase_plus_suite_pitch_tuning"
    )
    assert dymos_phase["pitch_program_optimization_scope"] == (
        "suite_tuning_not_full_dymos_pitch_transcription"
    )
    assert result.metadata["dymos_pitch_program_transcription_contract"][
        "control_schedule_covers_stage_plan"
    ] is True
    assert result.metadata["dymos_pitch_program_transcription_contract"][
        "stage_phase_count"
    ] == 2
    assert dymos_phase["duration_s"] >= result.metadata["stage_plan"]["total_burn_duration_s"]
    assert dymos_phase["stage_count"] == result.metadata["stage_plan"]["stage_count"]
    assert dymos_phase["final_altitude_km"] > 0.0
    assert dymos_phase["final_velocity_km_s"] > 0.0
    assert result.best_case.score >= 0.0


@pytest.mark.dymos_live
def test_live_dymos_pitch_program_optimization_executes_native_transcription() -> None:
    if os.environ.get("ASTRO_RUN_DYMOS_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_DYMOS_LIVE=1 to run live Dymos launch optimization")
    pytest.importorskip("dymos")
    pytest.importorskip("openmdao")

    scenario = load_launch_scenario(Path("examples/launch/pitch_program_two_stage.yaml"))

    result = optimize_launch_dymos(
        scenario,
        optimizer_runner=run_dymos_pitch_program_optimization,
    )

    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_pitch_program"
    assert result.metadata["converged"] is True
    assert result.metadata["dymos_phase"]["phase_model"] == "stage_aware_pitch_program_ascent"
    assert result.metadata["dymos_phase"]["pitch_program_optimization_coupling"] == (
        "native_dymos_pitch_control"
    )
    assert result.metadata["dymos_phase"]["pitch_program_optimization_scope"] == (
        "native_dymos_pitch_program_transcription"
    )
    assert result.metadata["dymos_phase"]["target_objective"] == (
        "minimize_final_normalized_target_insertion_error"
    )
    assert result.metadata["dymos_phase"]["target_score"] >= 0.0
    contract = result.metadata["dymos_pitch_program_transcription_contract"]
    assert contract["execution_status"] == "executed"
    assert contract["phase_coupling"] == "native_dymos_pitch_control"
    assert contract["optimized_control_points"][2]["source"] == (
        "dymos_pitch_program_control"
    )
    assert contract["optimized_control_points"][3]["source"] == (
        "dymos_pitch_program_control"
    )


@pytest.mark.dymos_live
def test_live_dymos_multistage_pitch_program_executes_native_multiphase() -> None:
    if os.environ.get("ASTRO_RUN_DYMOS_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_DYMOS_LIVE=1 to run live Dymos launch optimization")
    pytest.importorskip("dymos")
    pytest.importorskip("openmdao")

    scenario = load_launch_scenario(Path("examples/launch/pitch_program_two_stage.yaml"))

    result = optimize_launch_dymos(
        scenario,
        optimizer_runner=run_dymos_multistage_pitch_program_optimization,
    )

    assert result.backend == "dymos"
    assert result.metadata["source_backend"] == "dymos_multistage_pitch_program"
    assert result.metadata["converged"] is True
    dymos_phase = result.metadata["dymos_phase"]
    assert dymos_phase["phase_model"] == "multiphase_stage_pitch_program_ascent"
    assert dymos_phase["phase_count"] == 2
    assert dymos_phase["phase_topology"] == "multiphase_stage_linked"
    assert dymos_phase["linked_state_names"] == ("time", "h", "downrange", "vr", "vh")
    assert dymos_phase["mass_model"] == (
        "stage_local_propellant_depletion_with_fixed_stage_initial_mass"
    )
    assert dymos_phase["mass_state_name"] == "mass"
    assert dymos_phase["mass_state_linked"] is False
    assert len(dymos_phase["stage_phase_summaries"]) == 2
    assert dymos_phase["stage_phase_summaries"][0]["duration_s"] == 70.0
    assert dymos_phase["stage_phase_summaries"][0]["initial_mass_kg"] == pytest.approx(
        scenario.vehicle.initial_mass_kg
    )
    assert dymos_phase["stage_phase_summaries"][0]["final_mass_kg"] < (
        dymos_phase["stage_phase_summaries"][0]["initial_mass_kg"]
    )
    assert dymos_phase["stage_phase_summaries"][0]["propellant_used_kg"] > 0.0
    assert dymos_phase["stage_phase_summaries"][1]["duration_s"] == 50.0
    assert dymos_phase["stage_phase_summaries"][1]["initial_mass_kg"] == pytest.approx(
        scenario.vehicle.payload_mass_kg
        + scenario.vehicle.stages[1].dry_mass_kg
        + scenario.vehicle.stages[1].propellant_mass_kg
    )
    assert dymos_phase["target_objective"] == (
        "minimize_final_normalized_target_insertion_error"
    )
    assert dymos_phase["target_score"] >= 0.0
    contract = result.metadata["dymos_pitch_program_transcription_contract"]
    assert contract["execution_status"] == "executed"
    assert contract["phase_coupling"] == "native_dymos_multiphase_pitch_control"
    assert contract["phase_topology"] == "multiphase_stage_linked"
    assert contract["native_stage_phase_count"] == 2
    assert contract["linked_state_names"] == ("time", "h", "downrange", "vr", "vh")
