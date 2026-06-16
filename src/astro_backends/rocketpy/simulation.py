from __future__ import annotations

from collections.abc import Callable

from astro_backends.rocketpy.runtime import RocketPyRuntime, load_rocketpy_runtime
from astro_core.errors import UnsupportedBackendError
from astro_launch.models import LaunchRocketPyConfig, LaunchScenario, LaunchTrajectory

RocketPyRuntimeLoader = Callable[[], RocketPyRuntime]
RocketPyFlightRunner = Callable[
    [LaunchScenario, RocketPyRuntime, LaunchRocketPyConfig],
    LaunchTrajectory,
]


def _with_rocketpy_provenance(
    trajectory: LaunchTrajectory,
    runtime: RocketPyRuntime,
    config: LaunchRocketPyConfig,
) -> LaunchTrajectory:
    metadata = {
        **trajectory.metadata,
        "adapter": "rocketpy",
        "source_backend": trajectory.backend,
        "rocketpy_version": runtime.package_version,
        "rocketpy_configured": True,
        "rocketpy_rail_length_m": config.rail_length_m,
        "rocketpy_inclination_deg": config.inclination_deg,
        "rocketpy_heading_deg": config.heading_deg,
    }
    return trajectory.model_copy(update={"backend": "rocketpy", "metadata": metadata})


def propagate_launch_rocketpy(
    scenario: LaunchScenario,
    *,
    runtime_loader: RocketPyRuntimeLoader = load_rocketpy_runtime,
    flight_runner: RocketPyFlightRunner | None = None,
) -> LaunchTrajectory:
    runtime = runtime_loader()
    config = scenario.rocketpy
    if config is None:
        raise UnsupportedBackendError(
            "RocketPy launch simulation requires scenario.rocketpy backend-specific "
            "vehicle, motor, and flight configuration; use --backend local for aggregate "
            "launch scenarios."
        )
    if flight_runner is None:
        raise UnsupportedBackendError(
            "RocketPy launch simulation requires a validated flight runner for "
            "scenario.rocketpy configuration."
        )

    trajectory = flight_runner(scenario, runtime, config)
    return _with_rocketpy_provenance(trajectory, runtime, config)
