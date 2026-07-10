from astro_reentry.models import ReentryOptimizationConfig
from astro_reentry.optimization import optimize_reentry_guidance
from tests.astro_reentry.helpers import make_reentry_scenario, reference_target


def test_local_guidance_optimization_reduces_target_miss() -> None:
    scenario = make_reentry_scenario(
        mode="target_tracking",
        lift_to_drag_ratio=0.3,
        target=reference_target(),
    )

    result = optimize_reentry_guidance(
        scenario,
        ReentryOptimizationConfig(maximum_iterations=20),
    )

    assert result.final_objective < result.initial_objective
    assert result.reentry_result.target_miss is not None
    assert result.reentry_result.target_miss.distance_km < 10.0
    assert result.success is True
    assert result.metadata["optimizer"] == "scipy_Powell"
