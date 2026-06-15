from __future__ import annotations

from astro_backends.rocketpy.simulation import propagate_launch_rocketpy
from astro_core.errors import UnsupportedBackendError
from astro_launch.local import propagate_launch_local
from astro_launch.models import LaunchScenario, LaunchTrajectory

LaunchBackend = str


def propagate_launch_with_backend(
    scenario: LaunchScenario,
    backend: LaunchBackend,
) -> LaunchTrajectory:
    if backend == "local":
        return propagate_launch_local(scenario)
    if backend == "rocketpy":
        return propagate_launch_rocketpy(scenario)
    raise UnsupportedBackendError(f"unsupported launch backend: {backend}")
