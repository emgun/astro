from __future__ import annotations

from collections.abc import Callable

from astro_backends.rocketpy.runtime import RocketPyRuntime, load_rocketpy_runtime
from astro_core.errors import UnsupportedBackendError
from astro_launch.models import LaunchScenario, LaunchTrajectory

RocketPyRuntimeLoader = Callable[[], RocketPyRuntime]
RocketPyFlightRunner = Callable[[LaunchScenario, RocketPyRuntime], LaunchTrajectory]


def _with_rocketpy_provenance(
    trajectory: LaunchTrajectory,
    runtime: RocketPyRuntime,
) -> LaunchTrajectory:
    metadata = {
        **trajectory.metadata,
        "adapter": "rocketpy",
        "source_backend": trajectory.backend,
        "rocketpy_version": runtime.package_version,
    }
    return trajectory.model_copy(update={"backend": "rocketpy", "metadata": metadata})


def propagate_launch_rocketpy(
    scenario: LaunchScenario,
    *,
    runtime_loader: RocketPyRuntimeLoader = load_rocketpy_runtime,
    flight_runner: RocketPyFlightRunner | None = None,
) -> LaunchTrajectory:
    runtime = runtime_loader()
    if flight_runner is None:
        raise UnsupportedBackendError(
            "RocketPy launch simulation requires backend-specific rocket/motor "
            "configuration and a validated flight runner; use --backend local for "
            "the current aggregate launch scenarios."
        )

    trajectory = flight_runner(scenario, runtime)
    return _with_rocketpy_provenance(trajectory, runtime)
