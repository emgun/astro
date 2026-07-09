from astro_twin.coverage import access_windows_from_samples
from astro_twin.models import AccessWindow, GroundSiteConfig


def test_access_windows_from_samples_groups_contiguous_access() -> None:
    windows = access_windows_from_samples(
        ground_site=GroundSiteConfig(
            name="goldstone",
            latitude_deg=35.0,
            longitude_deg=-116.0,
            altitude_m=1000.0,
            minimum_elevation_deg=10.0,
        ),
        samples=[
            (0.0, 8.0, 2000.0),
            (60.0, 12.0, 1800.0),
            (120.0, 15.0, 1700.0),
            (180.0, 5.0, 2100.0),
        ],
    )

    assert windows == (
        AccessWindow(
            ground_site="goldstone",
            start_s=60.0,
            end_s=120.0,
            duration_s=60.0,
            max_elevation_deg=15.0,
            min_range_km=1700.0,
        ),
    )
