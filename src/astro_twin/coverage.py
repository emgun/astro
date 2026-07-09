from __future__ import annotations

from collections.abc import Iterable

from astro_twin.models import AccessWindow, GroundSiteConfig


def access_windows_from_samples(
    ground_site: GroundSiteConfig,
    samples: Iterable[tuple[float, float, float]],
) -> tuple[AccessWindow, ...]:
    windows: list[AccessWindow] = []
    active: list[tuple[float, float, float]] = []
    for elapsed_s, elevation_deg, range_km in samples:
        if elevation_deg >= ground_site.minimum_elevation_deg:
            active.append((elapsed_s, elevation_deg, range_km))
            continue
        if active:
            windows.append(_window_from_active(ground_site.name, active))
            active = []
    if active:
        windows.append(_window_from_active(ground_site.name, active))
    return tuple(windows)


def _window_from_active(site_name: str, active: list[tuple[float, float, float]]) -> AccessWindow:
    start_s = active[0][0]
    end_s = active[-1][0]
    return AccessWindow(
        ground_site=site_name,
        start_s=start_s,
        end_s=end_s,
        duration_s=max(1.0, end_s - start_s),
        max_elevation_deg=max(item[1] for item in active),
        min_range_km=min(item[2] for item in active),
    )
