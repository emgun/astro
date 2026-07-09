from astro_twin.link_budget import compute_link_budget_windows
from astro_twin.models import AccessWindow, LinkBudgetConfig


def test_compute_link_budget_windows_reports_positive_margin() -> None:
    windows = (
        AccessWindow(
            ground_site="goldstone",
            start_s=60.0,
            end_s=120.0,
            duration_s=60.0,
            max_elevation_deg=15.0,
            min_range_km=1700.0,
        ),
    )
    link = LinkBudgetConfig(
        name="xband-downlink",
        ground_site="goldstone",
        frequency_ghz=8.4,
        eirp_dbw=18.0,
        receiver_g_over_t_db_k=22.0,
        data_rate_bps=2_000_000.0,
        required_ebn0_db=6.5,
        implementation_loss_db=2.0,
    )

    result = compute_link_budget_windows((link,), windows)

    assert result[0].link_name == "xband-downlink"
    assert result[0].data_volume_mbit == 120.0
    assert result[0].worst_ebn0_margin_db > 0.0
