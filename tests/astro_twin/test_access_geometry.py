from astro_twin.geometry import elevation_and_range_km
from astro_twin.models import GroundSiteConfig


def test_elevation_and_range_km_places_overhead_spacecraft_above_mask() -> None:
    site = GroundSiteConfig(
        name="equator",
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        minimum_elevation_deg=10.0,
    )

    elevation_deg, range_km = elevation_and_range_km(
        position_km=(7000.0, 0.0, 0.0),
        site=site,
        elapsed_s=0.0,
    )

    assert elevation_deg > 80.0
    assert range_km > 600.0
