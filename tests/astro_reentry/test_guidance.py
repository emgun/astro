import pytest

from astro_reentry.guidance import (
    commanded_bank_angle_deg,
    initial_bearing_deg,
    schedule_bank_angle_deg,
    signed_heading_error_deg,
)
from astro_reentry.models import BankSchedulePoint, ReentryGuidanceConfig
from tests.astro_reentry.helpers import reference_target


def test_schedule_interpolates_velocity_indexed_bank_angle() -> None:
    config = ReentryGuidanceConfig(
        mode="bank_schedule",
        bank_schedule=(
            BankSchedulePoint(velocity_km_s=8.0, bank_angle_deg=20.0),
            BankSchedulePoint(velocity_km_s=4.0, bank_angle_deg=60.0),
        ),
    )

    assert schedule_bank_angle_deg(config, 6.0) == pytest.approx(40.0)


def test_target_tracking_steers_bank_sign_toward_target_heading() -> None:
    target = reference_target()
    config = ReentryGuidanceConfig(
        mode="target_tracking",
        bank_schedule=(
            BankSchedulePoint(velocity_km_s=8.0, bank_angle_deg=40.0),
            BankSchedulePoint(velocity_km_s=1.0, bank_angle_deg=20.0),
        ),
    )
    bearing = initial_bearing_deg(0.0, 0.0, target)
    bank, sign = commanded_bank_angle_deg(
        config,
        velocity_km_s=7.0,
        latitude_deg=0.0,
        longitude_deg=0.0,
        heading_deg=bearing - 10.0,
        target=target,
        previous_bank_sign=-1,
    )

    assert signed_heading_error_deg(bearing, bearing - 10.0) == pytest.approx(10.0)
    assert bank > 0.0
    assert sign == 1


def test_target_tracking_disables_bank_below_control_velocity() -> None:
    config = ReentryGuidanceConfig(
        mode="target_tracking",
        minimum_control_velocity_km_s=0.5,
        bank_schedule=(
            BankSchedulePoint(velocity_km_s=8.0, bank_angle_deg=40.0),
            BankSchedulePoint(velocity_km_s=1.0, bank_angle_deg=20.0),
        ),
    )

    assert commanded_bank_angle_deg(
        config,
        velocity_km_s=0.4,
        latitude_deg=0.0,
        longitude_deg=0.0,
        heading_deg=90.0,
        target=reference_target(),
        previous_bank_sign=-1,
    ) == (0.0, -1)
