import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_backends.dymos.optimization import DymosPhaseSummary, optimize_launch_dymos
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
    }


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
    }
    dymos_phase = result.metadata["dymos_phase"]
    assert dymos_phase["phase_model"] == "stage_aware_vertical_ascent"
    assert dymos_phase["transcription"] == "GaussLobatto"
    assert dymos_phase["duration_s"] >= result.metadata["stage_plan"]["total_burn_duration_s"]
    assert dymos_phase["stage_count"] == result.metadata["stage_plan"]["stage_count"]
    assert dymos_phase["final_altitude_km"] > 0.0
    assert dymos_phase["final_velocity_km_s"] > 0.0
    assert result.best_case.score >= 0.0
