from datetime import UTC, datetime

from astro_reentry.models import (
    BankSchedulePoint,
    ReentryGuidanceConfig,
    ReentryInitialState,
    ReentryLimits,
    ReentryPropagationConfig,
    ReentryScenario,
    ReentryTarget,
    ReentryVehicle,
)


def make_reentry_scenario(
    *,
    mode: str = "ballistic",
    lift_to_drag_ratio: float = 0.0,
    target: ReentryTarget | None = None,
    **overrides: object,
) -> ReentryScenario:
    if mode in {"bank_schedule", "target_tracking"}:
        guidance = ReentryGuidanceConfig(
            mode=mode,
            heading_deadband_deg=3.0,
            minimum_bank_reversal_interval_s=30.0,
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=7.8, bank_angle_deg=45.0),
                BankSchedulePoint(velocity_km_s=5.0, bank_angle_deg=55.0),
                BankSchedulePoint(velocity_km_s=2.5, bank_angle_deg=35.0),
                BankSchedulePoint(velocity_km_s=0.5, bank_angle_deg=20.0),
            ),
        )
    else:
        guidance = ReentryGuidanceConfig(
            mode=mode,
            bank_angle_deg=45.0 if mode == "constant_bank" else 0.0,
        )
    payload: dict[str, object] = {
        "scenario_id": f"test-{mode}",
        "description": "Reentry unit test scenario.",
        "initial_state": ReentryInitialState(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            altitude_km=120.0,
            velocity_km_s=7.8,
            flight_path_angle_deg=-6.0,
            heading_deg=90.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
        ),
        "vehicle": ReentryVehicle(
            name="test-entry-vehicle",
            mass_kg=5500.0,
            reference_area_m2=12.0,
            drag_coefficient=1.5,
            lift_to_drag_ratio=lift_to_drag_ratio,
            nose_radius_m=0.75,
        ),
        "guidance": guidance,
        "target": target,
        "limits": ReentryLimits(
            maximum_dynamic_pressure_pa=60000.0,
            maximum_deceleration_g=20.0,
            maximum_heat_rate_w_m2=2500000.0,
            maximum_heat_load_j_m2=150000000.0,
        ),
        "propagation": ReentryPropagationConfig(
            duration_s=1000.0,
            step_s=5.0,
            internal_step_s=1.0,
        ),
    }
    payload.update(overrides)
    return ReentryScenario(**payload)


def reference_target() -> ReentryTarget:
    return ReentryTarget(
        name="reference-target",
        latitude_deg=1.0,
        longitude_deg=10.0,
        allowable_miss_km=25.0,
    )
