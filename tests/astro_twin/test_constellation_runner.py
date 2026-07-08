from datetime import UTC, datetime

import pytest

import astro_twin.constellation as constellation
from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_io import load_constellation_twin_scenario
from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
    ConstellationMemberConfig,
    ConstellationTwinScenario,
)
from astro_twin.io import load_twin_scenario
from astro_twin.models import (
    DesignMargin,
    DesignMarginReport,
    DigitalTwinResult,
    TimelineGeometrySample,
    TwinMarginStatus,
)


def test_run_constellation_twin_returns_fleet_result() -> None:
    scenario = load_constellation_twin_scenario(
        "examples/twin/constellation_leo_observers.yaml"
    )

    result = constellation.run_constellation_twin(scenario)

    assert result.workflow == "constellation_digital_twin_v1"
    assert [member.member_name for member in result.members] == ["plane-a", "plane-b"]
    assert result.metadata["analysis_window_s"] == {
        "start_s": 0.0,
        "end_s": 600.0,
    }
    assert len(result.access_summaries) == 1
    assert result.access_summaries[0].ground_site == "equator-eci"
    assert result.access_summaries[0].total_access_duration_s == 300.0
    assert result.access_summaries[0].coverage_fraction == 0.5
    assert result.access_summaries[0].longest_gap_s == 300.0
    assert result.access_summaries[0].max_simultaneous_spacecraft == 2
    member_access_by_name = {
        member.member_name: member.result.access_windows for member in result.members
    }
    assert member_access_by_name["plane-a"][0].start_s == 0.0
    assert member_access_by_name["plane-a"][0].end_s == 240.0
    assert member_access_by_name["plane-b"][0].start_s == 0.0
    assert member_access_by_name["plane-b"][0].end_s == 300.0
    assert len(result.link_summaries) == 1
    assert result.link_summaries[0].ground_site == "equator-eci"
    assert result.link_summaries[0].total_data_volume_mbit > 0.0
    member_data_by_name = {
        summary.member_name: summary.total_data_volume_mbit
        for summary in result.member_link_summaries
    }
    assert set(member_data_by_name) == {"plane-a", "plane-b"}
    assert member_data_by_name["plane-a"] > 0.0
    assert member_data_by_name["plane-b"] > 0.0
    assert result.fleet_margin_report.limiting_margin.name
    assert any("design-screening" in warning for warning in result.warnings)


def test_run_constellation_twin_rejects_member_without_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_scenario = load_twin_scenario("examples/twin/leo_observer_plane_a.yaml")
    scenario = ConstellationTwinScenario(
        scenario_id="no-geometry",
        members=(
            ConstellationMemberConfig(name="plane-a", twin_scenario="plane-a.yaml"),
        ),
    )

    monkeypatch.setattr(
        constellation,
        "load_twin_scenario",
        lambda _path: member_scenario,
    )
    monkeypatch.setattr(
        constellation,
        "run_digital_twin",
        lambda _scenario: _digital_twin_result(elapsed_s=()),
    )

    with pytest.raises(InvalidScenarioError, match="produced no geometry samples"):
        constellation.run_constellation_twin(scenario)


def test_run_constellation_twin_validates_requirement_sites_before_running_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_scenario = load_twin_scenario("examples/twin/leo_observer_plane_a.yaml")
    scenario = ConstellationTwinScenario(
        scenario_id="bad-requirement-site",
        members=(
            ConstellationMemberConfig(name="plane-a", twin_scenario="plane-a.yaml"),
        ),
        coverage_requirements=(
            ConstellationCoverageRequirement(ground_site="unconfigured-site"),
        ),
    )

    monkeypatch.setattr(
        constellation,
        "load_twin_scenario",
        lambda _path: member_scenario,
    )

    def fail_if_called(_scenario: object) -> DigitalTwinResult:
        raise AssertionError("run_digital_twin should not run before site validation")

    monkeypatch.setattr(constellation, "run_digital_twin", fail_if_called)

    with pytest.raises(InvalidScenarioError, match="unconfigured ground sites"):
        constellation.run_constellation_twin(scenario)


def test_run_constellation_twin_reports_configured_sites_without_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_scenario = load_twin_scenario("examples/twin/leo_observer_plane_a.yaml")
    scenario = ConstellationTwinScenario(
        scenario_id="no-access",
        members=(
            ConstellationMemberConfig(name="plane-a", twin_scenario="plane-a.yaml"),
        ),
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=300.0,
            ),
        ),
    )

    monkeypatch.setattr(
        constellation,
        "load_twin_scenario",
        lambda _path: member_scenario,
    )
    monkeypatch.setattr(
        constellation,
        "run_digital_twin",
        lambda _scenario: _digital_twin_result(elapsed_s=(0.0, 600.0)),
    )

    result = constellation.run_constellation_twin(scenario)

    assert result.access_summaries[0].ground_site == "equator-eci"
    assert result.access_summaries[0].coverage_fraction == 0.0
    assert result.access_summaries[0].longest_gap_s == 600.0
    assert result.link_summaries[0].ground_site == "equator-eci"
    assert result.link_summaries[0].total_data_volume_mbit == 0.0
    assert result.link_summaries[0].worst_ebn0_margin_db is None
    margin_by_name = {
        margin.name: margin for margin in result.fleet_margin_report.margins
    }
    assert (
        margin_by_name["fleet_coverage_fraction_equator-eci"].status
        is TwinMarginStatus.FAIL
    )
    assert (
        margin_by_name["fleet_link_margin_db_equator-eci"].status
        is TwinMarginStatus.FAIL
    )


def test_run_constellation_twin_rejects_member_timelines_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_twin_scenario("examples/twin/leo_observer_plane_a.yaml")
    member_scenarios = {
        "plane-a.yaml": base_scenario.model_copy(update={"scenario_id": "plane-a"}),
        "plane-b.yaml": base_scenario.model_copy(update={"scenario_id": "plane-b"}),
    }
    scenario = ConstellationTwinScenario(
        scenario_id="no-overlap",
        members=(
            ConstellationMemberConfig(name="plane-a", twin_scenario="plane-a.yaml"),
            ConstellationMemberConfig(name="plane-b", twin_scenario="plane-b.yaml"),
        ),
    )

    monkeypatch.setattr(
        constellation,
        "load_twin_scenario",
        lambda path: member_scenarios[path],
    )
    monkeypatch.setattr(
        constellation,
        "run_digital_twin",
        lambda twin_scenario: _digital_twin_result(
            elapsed_s=(0.0, 60.0)
            if twin_scenario.scenario_id == "plane-a"
            else (120.0, 180.0)
        ),
    )

    with pytest.raises(InvalidScenarioError, match="do not overlap"):
        constellation.run_constellation_twin(scenario)


def _digital_twin_result(elapsed_s: tuple[float, ...]) -> DigitalTwinResult:
    margin = DesignMargin(
        name="mass_margin_fraction",
        value=0.3,
        threshold=0.2,
        margin=0.1,
        status=TwinMarginStatus.PASS,
    )
    return DigitalTwinResult(
        scenario_id="test-member",
        geometry=tuple(
            TimelineGeometrySample(
                epoch=datetime(2026, 1, 1, tzinfo=UTC),
                elapsed_s=elapsed,
                position_km=(7000.0, 0.0, 0.0),
                altitude_km=621.8637,
                sunlit=True,
            )
            for elapsed in elapsed_s
        ),
        power=(),
        thermal=(),
        adcs=(),
        access_windows=(),
        link_windows=(),
        margin_report=DesignMarginReport(margins=(margin,), limiting_margin=margin),
    )
