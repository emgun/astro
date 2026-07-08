import pytest

from astro_twin.constellation import (
    aggregate_access_summaries,
    aggregate_link_summaries,
    build_fleet_margin_report,
)
from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
    MemberLinkSummary,
)
from astro_twin.models import (
    AccessWindow,
    DesignMargin,
    LinkBudgetWindow,
    TwinMarginStatus,
)


def test_aggregate_access_summaries_computes_union_gaps_and_simultaneous_count() -> None:
    summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=120.0,
                    duration_s=120.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=300.0,
                    end_s=420.0,
                    duration_s=120.0,
                    max_elevation_deg=70.0,
                    min_range_km=900.0,
                ),
            ),
            "plane-b": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=60.0,
                    end_s=180.0,
                    duration_s=120.0,
                    max_elevation_deg=75.0,
                    min_range_km=800.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.ground_site == "equator-eci"
    assert summary.total_access_duration_s == 300.0
    assert summary.longest_gap_s == 180.0
    assert summary.mean_gap_s == 150.0
    assert summary.max_simultaneous_spacecraft == 2
    assert summary.coverage_fraction == 0.5


def test_aggregate_link_summaries_groups_by_site_and_member() -> None:
    fleet, members = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=60.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=120.0,
                ),
            ),
            "plane-b": (
                LinkBudgetWindow(
                    link_name="xband-b",
                    ground_site="equator-eci",
                    start_s=120.0,
                    end_s=180.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=2.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    assert fleet[0].ground_site == "equator-eci"
    assert fleet[0].total_data_volume_mbit == 240.0
    assert fleet[0].worst_ebn0_margin_db == 2.0
    assert members == (
        MemberLinkSummary(
            member_name="plane-a",
            total_data_volume_mbit=120.0,
            worst_ebn0_margin_db=4.0,
        ),
        MemberLinkSummary(
            member_name="plane-b",
            total_data_volume_mbit=120.0,
            worst_ebn0_margin_db=2.0,
        ),
    )


def test_aggregate_link_summaries_sorts_members_by_name() -> None:
    _fleet, members = aggregate_link_summaries(
        member_link_windows={
            "plane-b": (
                LinkBudgetWindow(
                    link_name="xband-b",
                    ground_site="equator-eci",
                    start_s=120.0,
                    end_s=180.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=2.0,
                    data_volume_mbit=120.0,
                ),
            ),
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=60.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    assert [member.member_name for member in members] == ["plane-a", "plane-b"]


def test_build_fleet_margin_report_uses_coverage_link_and_member_margins() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )
    link_summaries, _ = aggregate_link_summaries(
        member_link_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=300.0,
            ),
        ),
        member_limiting_margins={
            "plane-a": DesignMargin(
                name="mass_margin_fraction",
                value=0.24,
                threshold=0.2,
                margin=0.04,
                status=TwinMarginStatus.WARN,
            )
        },
    )

    margin_by_name = {margin.name: margin for margin in report.margins}
    assert "fleet_coverage_fraction_equator-eci" in margin_by_name
    assert "fleet_longest_gap_s_equator-eci" in margin_by_name
    assert "fleet_link_margin_db_equator-eci" in margin_by_name
    assert "member_plane-a_mass_margin_fraction" in margin_by_name
    assert report.limiting_margin.name == "fleet_coverage_fraction_equator-eci"
    assert report.limiting_margin.status is TwinMarginStatus.FAIL


def test_aggregation_clips_windows_to_analysis_interval() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=100.0,
                    duration_s=100.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=180.0,
                    end_s=260.0,
                    duration_s=80.0,
                    max_elevation_deg=60.0,
                    min_range_km=900.0,
                ),
            )
        },
        analysis_start_s=50.0,
        analysis_end_s=200.0,
    )
    link_summaries, member_summaries = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=100.0,
                    duration_s=100.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=200.0,
                ),
            ),
            "plane-b": (
                LinkBudgetWindow(
                    link_name="xband-b",
                    ground_site="equator-eci",
                    start_s=220.0,
                    end_s=280.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=1.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=50.0,
        analysis_end_s=200.0,
    )

    assert access_summaries[0].total_access_duration_s == 70.0
    assert access_summaries[0].coverage_fraction == pytest.approx(70.0 / 150.0)
    assert link_summaries[0].total_data_volume_mbit == 100.0
    assert link_summaries[0].worst_ebn0_margin_db == 4.0
    assert member_summaries == (
        MemberLinkSummary(
            member_name="plane-a",
            total_data_volume_mbit=100.0,
            worst_ebn0_margin_db=4.0,
        ),
        MemberLinkSummary(
            member_name="plane-b",
            total_data_volume_mbit=0.0,
            worst_ebn0_margin_db=None,
        ),
    )


def test_link_proration_uses_window_timestamps_not_declared_duration() -> None:
    link_summaries, member_summaries = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=100.0,
                    duration_s=200.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=200.0,
                ),
            ),
        },
        analysis_start_s=50.0,
        analysis_end_s=100.0,
    )

    assert link_summaries[0].total_data_volume_mbit == 100.0
    assert member_summaries[0].total_data_volume_mbit == 100.0


def test_configured_ground_sites_without_windows_stay_visible() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
        ground_sites=("equator-eci",),
    )
    link_summaries, _member_summaries = aggregate_link_summaries(
        member_link_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
        ground_sites=("equator-eci",),
    )

    assert access_summaries[0].ground_site == "equator-eci"
    assert access_summaries[0].total_access_duration_s == 0.0
    assert access_summaries[0].longest_gap_s == 600.0
    assert access_summaries[0].coverage_fraction == 0.0
    assert link_summaries[0].ground_site == "equator-eci"
    assert link_summaries[0].total_data_volume_mbit == 0.0
    assert link_summaries[0].worst_ebn0_margin_db is None


def test_build_fleet_margin_report_flags_missing_site_link_margin() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=600.0,
                    duration_s=600.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
            )
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=(),
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=700.0,
            ),
        ),
        member_limiting_margins={},
    )

    link_margin = next(
        margin
        for margin in report.margins
        if margin.name == "fleet_link_margin_db_equator-eci"
    )
    assert link_margin.status is TwinMarginStatus.FAIL


def test_build_fleet_margin_report_adds_default_coverage_for_access_sites() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=300.0,
                    duration_s=300.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
            )
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )
    link_summaries, _ = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=60.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=(),
        member_limiting_margins={},
    )

    margin_by_name = {margin.name: margin for margin in report.margins}
    coverage_margin = margin_by_name["fleet_coverage_fraction_equator-eci"]
    link_margin = margin_by_name["fleet_link_margin_db_equator-eci"]
    assert coverage_margin.value == 0.5
    assert coverage_margin.threshold == 0.0
    assert coverage_margin.status is TwinMarginStatus.PASS
    assert link_margin.value == 4.0
    assert link_margin.status is TwinMarginStatus.PASS


def test_build_fleet_margin_report_breaks_limiting_margin_ties_by_name() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=300.0,
                    end_s=600.0,
                    duration_s=300.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
            )
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )
    link_summaries, _ = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=300.0,
                    end_s=360.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=0.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.0,
                maximum_revisit_gap_s=300.0,
            ),
        ),
        member_limiting_margins={},
    )

    assert report.limiting_margin.name == "fleet_link_margin_db_equator-eci"
