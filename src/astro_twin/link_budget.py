from __future__ import annotations

from math import log10

from astro_twin.models import AccessWindow, LinkBudgetConfig, LinkBudgetWindow

_BOLTZMANN_DB = 228.6


def compute_link_budget_windows(
    links: tuple[LinkBudgetConfig, ...],
    access_windows: tuple[AccessWindow, ...],
) -> tuple[LinkBudgetWindow, ...]:
    results: list[LinkBudgetWindow] = []
    for link in links:
        for window in access_windows:
            if window.ground_site != link.ground_site:
                continue
            fspl_db = 92.45 + 20.0 * log10(window.min_range_km) + 20.0 * log10(
                link.frequency_ghz
            )
            cn0_db_hz = (
                link.eirp_dbw
                + link.receiver_g_over_t_db_k
                - fspl_db
                - link.implementation_loss_db
                + _BOLTZMANN_DB
            )
            ebn0_db = cn0_db_hz - 10.0 * log10(link.data_rate_bps)
            results.append(
                LinkBudgetWindow(
                    link_name=link.name,
                    ground_site=window.ground_site,
                    start_s=window.start_s,
                    end_s=window.end_s,
                    duration_s=window.duration_s,
                    worst_ebn0_margin_db=ebn0_db - link.required_ebn0_db,
                    data_volume_mbit=link.data_rate_bps * window.duration_s / 1_000_000.0,
                )
            )
    return tuple(results)
