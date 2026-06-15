from __future__ import annotations

from collections.abc import Callable

from astro_backends.tudat.runtime import TudatRuntime, load_tudat_runtime
from astro_core.errors import UnsupportedBackendError
from astro_core.models import Scenario, Trajectory

TudatRuntimeLoader = Callable[[], TudatRuntime]
TudatPropagationRunner = Callable[[Scenario, TudatRuntime], Trajectory]


def _with_tudat_provenance(
    trajectory: Trajectory,
    runtime: TudatRuntime,
) -> Trajectory:
    metadata = {
        **trajectory.metadata,
        "adapter": "tudat",
        "source_backend": trajectory.backend,
        "tudat_version": runtime.package_version,
    }
    return trajectory.model_copy(update={"backend": "tudat", "metadata": metadata})


def propagate_tudat(
    scenario: Scenario,
    *,
    runtime_loader: TudatRuntimeLoader = load_tudat_runtime,
    tudat_runner: TudatPropagationRunner | None = None,
) -> Trajectory:
    runtime = runtime_loader()
    if tudat_runner is None:
        raise UnsupportedBackendError(
            "Tudat propagation requires a validated Tudat runner and environment/body "
            "configuration; use --backend local or --backend orekit for current scenarios."
        )

    trajectory = tudat_runner(scenario, runtime)
    return _with_tudat_provenance(trajectory, runtime)
