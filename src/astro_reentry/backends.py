from __future__ import annotations

from astro_core.errors import UnsupportedBackendError
from astro_reentry.models import ReentryResult, ReentryScenario
from astro_reentry.simulation import simulate_reentry_local


def simulate_reentry_with_backend(
    scenario: ReentryScenario,
    backend: str,
) -> ReentryResult:
    if backend == "local":
        return simulate_reentry_local(scenario)
    raise UnsupportedBackendError(f"unsupported reentry backend: {backend}")
