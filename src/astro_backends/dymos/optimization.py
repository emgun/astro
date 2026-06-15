from __future__ import annotations

from collections.abc import Callable

from astro_backends.dymos.runtime import DymosRuntime, load_dymos_runtime
from astro_core.errors import UnsupportedBackendError
from astro_launch.models import LaunchPitchTuningResult, LaunchScenario

DymosRuntimeLoader = Callable[[], DymosRuntime]
DymosOptimizerRunner = Callable[[LaunchScenario, DymosRuntime], LaunchPitchTuningResult]


def _with_dymos_provenance(
    result: LaunchPitchTuningResult,
    runtime: DymosRuntime,
) -> LaunchPitchTuningResult:
    metadata = {
        **result.metadata,
        "adapter": "dymos",
        "source_backend": result.backend,
        "dymos_version": runtime.dymos_version,
        "openmdao_version": runtime.openmdao_version,
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
