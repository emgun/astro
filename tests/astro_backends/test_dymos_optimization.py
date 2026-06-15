from types import SimpleNamespace

import pytest

from astro_backends.dymos.optimization import optimize_launch_dymos
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


def test_optimize_launch_dymos_requires_phase_model_runner() -> None:
    with pytest.raises(UnsupportedBackendError, match="requires a validated Dymos phase"):
        optimize_launch_dymos(
            make_pitch_program_launch_scenario(),
            runtime_loader=_fake_runtime,
        )


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
