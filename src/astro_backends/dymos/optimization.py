from __future__ import annotations

from collections.abc import Callable
from math import isfinite

from astro_backends.dymos.runtime import DymosRuntime, load_dymos_runtime
from astro_core.errors import UnsupportedBackendError
from astro_launch.models import LaunchPitchTuningResult, LaunchScenario

DymosRuntimeLoader = Callable[[], DymosRuntime]
DymosOptimizerRunner = Callable[[LaunchScenario, DymosRuntime], LaunchPitchTuningResult]


def _with_dymos_provenance(
    result: LaunchPitchTuningResult,
    runtime: DymosRuntime,
) -> LaunchPitchTuningResult:
    source_candidate_count = result.metadata.get("candidate_count")
    converged = isfinite(float(result.best_case.score))
    metadata = {
        **result.metadata,
        "adapter": "dymos",
        "source_backend": result.backend,
        "source_candidate_count": source_candidate_count,
        "dymos_version": runtime.dymos_version,
        "openmdao_version": runtime.openmdao_version,
        "optimizer_status": "completed",
        "converged": converged,
        "iteration_count": len(result.iterations),
        "candidate_count": source_candidate_count,
        "best_score": result.best_case.score,
        "target_insertion_residuals": result.best_case.target_miss,
        "path_constraints": {
            "pitch_deg": {"lower": 0.0, "upper": 90.0},
        },
    }
    return result.model_copy(update={"backend": "dymos", "metadata": metadata})


def optimize_launch_dymos(
    scenario: LaunchScenario,
    *,
    runtime_loader: DymosRuntimeLoader = load_dymos_runtime,
    optimizer_runner: DymosOptimizerRunner | None = None,
) -> LaunchPitchTuningResult:
    runtime = runtime_loader()
    if optimizer_runner is None:
        raise UnsupportedBackendError(
            "Dymos launch optimization requires a validated Dymos phase model and "
            "optimizer runner; use --backend local for the current pitch-program tuner."
        )

    result = optimizer_runner(scenario, runtime)
    return _with_dymos_provenance(result, runtime)
