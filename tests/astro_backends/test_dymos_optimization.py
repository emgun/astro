from types import SimpleNamespace

import pytest

from astro_backends.dymos.optimization import DymosPhaseSummary, optimize_launch_dymos
from astro_backends.dymos.runtime import DymosRuntime
from astro_core.errors import UnsupportedBackendError
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
            transcription="GaussLobatto",
            num_segments=3,
            order=3,
            duration_s=42.0,
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
        "transcription": "GaussLobatto",
        "num_segments": 3,
        "order": 3,
        "duration_s": 42.0,
        "final_altitude_km": 1.2,
        "final_velocity_km_s": 0.08,
        "optimizer_success": True,
        "optimizer_message": "Optimization terminated successfully",
    }


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
    assert result.metadata["path_constraints"] == {
        "pitch_deg": {"lower": 0.0, "upper": 90.0},
    }
