from __future__ import annotations

from astro_core.errors import UnsupportedBackendError
from astro_launch.models import LaunchScenario, LaunchTrajectory


def propagate_launch_rocketpy(_scenario: LaunchScenario) -> LaunchTrajectory:
    raise UnsupportedBackendError(
        "RocketPy launch simulation requires a RocketPy adapter configuration; "
        "use --backend local for the current aggregate launch scenarios."
    )
