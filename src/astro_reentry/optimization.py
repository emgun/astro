from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize  # type: ignore[import-untyped]

from astro_reentry.models import (
    BankSchedulePoint,
    ReentryOptimizationConfig,
    ReentryOptimizationResult,
    ReentryResult,
    ReentryScenario,
)
from astro_reentry.simulation import simulate_reentry_local


def optimize_reentry_guidance(
    scenario: ReentryScenario,
    config: ReentryOptimizationConfig | None = None,
) -> ReentryOptimizationResult:
    settings = config or ReentryOptimizationConfig()
    if scenario.guidance.mode != "target_tracking":
        raise ValueError("reentry optimization requires target_tracking guidance")
    if scenario.target is None:
        raise ValueError("reentry optimization requires a target")
    initial_angles = np.array(
        [point.bank_angle_deg for point in scenario.guidance.bank_schedule],
        dtype=np.float64,
    )
    if initial_angles.size < 2:
        raise ValueError("reentry optimization requires at least two bank schedule points")

    def objective(angles: NDArray[np.float64]) -> float:
        candidate = _scenario_with_bank_angles(scenario, angles)
        result = simulate_reentry_local(candidate)
        return _objective_value(result, settings.load_penalty_scale)

    initial_objective = objective(initial_angles)
    raw_result = minimize(
        objective,
        initial_angles,
        method="Powell",
        bounds=[
            (float(settings.bank_angle_lower_deg), float(settings.bank_angle_upper_deg))
            for _ in initial_angles
        ],
        options={
            "maxiter": settings.maximum_iterations,
            "xtol": 1.0e-3,
            "ftol": 1.0e-6,
        },
    )
    optimized = cast(OptimizeResult, raw_result)
    optimized_angles = np.asarray(optimized.x, dtype=np.float64)
    tuned_scenario = _scenario_with_bank_angles(scenario, optimized_angles)
    reentry_result = simulate_reentry_local(tuned_scenario)
    return ReentryOptimizationResult(
        scenario_id=scenario.scenario_id,
        success=bool(optimized.success),
        message=str(optimized.message),
        iterations=int(optimized.nit),
        initial_objective=initial_objective,
        final_objective=_objective_value(reentry_result, settings.load_penalty_scale),
        tuned_scenario=tuned_scenario,
        reentry_result=reentry_result,
        metadata={
            "optimizer": "scipy_Powell",
            "optimized_fields": "guidance.bank_schedule[].bank_angle_deg",
            "bank_angle_bounds_deg": [
                settings.bank_angle_lower_deg,
                settings.bank_angle_upper_deg,
            ],
            "load_penalty_scale": settings.load_penalty_scale,
        },
    )


def _scenario_with_bank_angles(
    scenario: ReentryScenario,
    angles: NDArray[np.float64],
) -> ReentryScenario:
    schedule = tuple(
        BankSchedulePoint(
            velocity_km_s=point.velocity_km_s,
            bank_angle_deg=float(angle),
        )
        for point, angle in zip(scenario.guidance.bank_schedule, angles, strict=True)
    )
    guidance = scenario.guidance.model_copy(update={"bank_schedule": schedule})
    return scenario.model_copy(update={"guidance": guidance})


def _objective_value(result: ReentryResult, penalty_scale: float) -> float:
    if result.target_miss is None:
        raise ValueError("reentry optimization result does not contain target miss")
    penalty = sum(
        max(0.0, -float(margin.margin) / abs(float(margin.threshold))) ** 2
        for margin in result.margin_report.margins
        if margin.name != "target_miss_margin_km"
    )
    return float(result.target_miss.distance_km + penalty_scale * penalty)
