import pytest
from pydantic import ValidationError

from astro_reentry.models import (
    BankSchedulePoint,
    ReentryGuidanceConfig,
    ReentryInitialState,
    ReentryPropagationConfig,
)
from tests.astro_reentry.helpers import make_reentry_scenario, reference_target


def test_ballistic_scenario_exposes_ballistic_coefficient() -> None:
    scenario = make_reentry_scenario()

    assert scenario.guidance.mode == "ballistic"
    assert scenario.vehicle.ballistic_coefficient_kg_m2 == pytest.approx(305.5555556)


def test_guidance_and_vehicle_lift_must_be_compatible() -> None:
    with pytest.raises(ValidationError, match="ballistic guidance requires"):
        make_reentry_scenario(lift_to_drag_ratio=0.2)

    with pytest.raises(ValidationError, match="lifting guidance requires"):
        make_reentry_scenario(mode="constant_bank")


def test_target_tracking_requires_target_and_positive_schedule_magnitudes() -> None:
    with pytest.raises(ValidationError, match="requires a target"):
        make_reentry_scenario(mode="target_tracking", lift_to_drag_ratio=0.3)

    with pytest.raises(ValidationError, match="non-negative magnitudes"):
        ReentryGuidanceConfig(
            mode="target_tracking",
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=7.8, bank_angle_deg=45.0),
                BankSchedulePoint(velocity_km_s=1.0, bank_angle_deg=-30.0),
            ),
        )

    scenario = make_reentry_scenario(
        mode="target_tracking",
        lift_to_drag_ratio=0.3,
        target=reference_target(),
    )
    assert scenario.target is not None


def test_bank_schedule_requires_strictly_decreasing_velocity() -> None:
    with pytest.raises(ValidationError, match="strictly decreasing"):
        ReentryGuidanceConfig(
            mode="bank_schedule",
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=5.0, bank_angle_deg=20.0),
                BankSchedulePoint(velocity_km_s=6.0, bank_angle_deg=30.0),
            ),
        )


def test_guidance_rejects_fields_that_its_mode_would_ignore() -> None:
    with pytest.raises(ValidationError, match="ballistic guidance requires bank_angle_deg = 0"):
        ReentryGuidanceConfig(mode="ballistic", bank_angle_deg=10.0)
    with pytest.raises(ValidationError, match="constant_bank guidance does not accept"):
        ReentryGuidanceConfig(
            mode="constant_bank",
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=7.0, bank_angle_deg=20.0),
                BankSchedulePoint(velocity_km_s=2.0, bank_angle_deg=30.0),
            ),
        )
    with pytest.raises(ValidationError, match="bank_schedule guidance requires bank_angle_deg = 0"):
        ReentryGuidanceConfig(
            mode="bank_schedule",
            bank_angle_deg=10.0,
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=7.0, bank_angle_deg=20.0),
                BankSchedulePoint(velocity_km_s=2.0, bank_angle_deg=30.0),
            ),
        )


def test_propagation_requires_nested_integer_step_schedule() -> None:
    with pytest.raises(ValidationError, match="integer multiple of step_s"):
        ReentryPropagationConfig(duration_s=100.0, step_s=30.0, internal_step_s=1.0)
    with pytest.raises(ValidationError, match="integer multiple of internal_step_s"):
        ReentryPropagationConfig(duration_s=100.0, step_s=10.0, internal_step_s=3.0)


def test_scenario_requires_initial_state_above_terminal_velocity() -> None:
    with pytest.raises(ValidationError, match="initial velocity must be above"):
        make_reentry_scenario(
            initial_state=ReentryInitialState(
                epoch="2026-01-01T00:00:00Z",
                altitude_km=120.0,
                velocity_km_s=0.05,
                flight_path_angle_deg=-5.0,
                heading_deg=90.0,
                latitude_deg=0.0,
                longitude_deg=0.0,
            )
        )
