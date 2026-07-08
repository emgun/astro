from datetime import UTC, datetime

from astro_twin.adcs import compute_adcs_timeline
from astro_twin.models import ADCSConfig, TimelineGeometrySample


def test_compute_adcs_timeline_reports_positive_margins() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
    )
    config = ADCSConfig(
        pointing_mode="nadir",
        max_pointing_error_deg=0.08,
        pointing_requirement_deg=0.15,
        max_torque_n_m=0.08,
        required_slew_torque_n_m=0.03,
    )

    samples = compute_adcs_timeline(config, geometry)

    assert samples[0].pointing_margin_deg == 0.06999999999999999
    assert samples[0].torque_margin_n_m == 0.05
