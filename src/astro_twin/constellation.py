from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import acos, cos, degrees, isfinite, radians, sin, sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_models import (
    ConstellationCoverageMapConfig,
    ConstellationCoverageRequirement,
    ConstellationCoverageSensorConfig,
    ConstellationCoverageTargetConfig,
    ConstellationTwinResult,
    ConstellationTwinScenario,
    CoverageMapSummary,
    CoverageMapTargetSummary,
    FleetAccessSummary,
    FleetLinkSummary,
    MemberLinkSummary,
    MemberTwinResult,
)
from astro_twin.geometry import elevation_and_range_km
from astro_twin.io import load_twin_scenario
from astro_twin.models import (
    AccessWindow,
    DesignMargin,
    DesignMarginReport,
    GroundSiteConfig,
    LinkBudgetWindow,
    TimelineGeometrySample,
    TwinMarginStatus,
)
from astro_twin.runner import run_digital_twin

_CONSTELLATION_DESIGN_SCREENING_WARNING = (
    "Constellation digital twin v1 is deterministic design-screening evidence "
    "for architecture trades, not operational constellation coverage authority."
)


def run_constellation_twin(
    scenario: ConstellationTwinScenario,
) -> ConstellationTwinResult:
    """Run each member twin and aggregate deterministic fleet screening evidence."""
    member_results: list[MemberTwinResult] = []
    member_access_windows: dict[str, tuple[AccessWindow, ...]] = {}
    member_link_windows: dict[str, tuple[LinkBudgetWindow, ...]] = {}
    member_geometry_samples: dict[str, tuple[TimelineGeometrySample, ...]] = {}
    member_limiting_margins: dict[str, DesignMargin] = {}
    configured_ground_sites: set[str] = set()
    analysis_starts_s: list[float] = []
    analysis_ends_s: list[float] = []
    warnings = [_CONSTELLATION_DESIGN_SCREENING_WARNING]
    loaded_member_scenarios = []

    for member in scenario.members:
        member_scenario = load_twin_scenario(member.twin_scenario)
        configured_ground_sites.update(site.name for site in member_scenario.ground_sites)
        loaded_member_scenarios.append((member, member_scenario))

    _validate_coverage_requirement_sites(scenario, configured_ground_sites)

    for member, member_scenario in loaded_member_scenarios:
        member_result = run_digital_twin(member_scenario)
        if not member_result.geometry:
            raise InvalidScenarioError(
                f"Constellation member {member.name} produced no geometry samples"
            )

        analysis_starts_s.append(member_result.geometry[0].elapsed_s)
        analysis_ends_s.append(member_result.geometry[-1].elapsed_s)
        member_results.append(
            MemberTwinResult(member_name=member.name, result=member_result)
        )
        member_access_windows[member.name] = member_result.access_windows
        member_link_windows[member.name] = member_result.link_windows
        member_geometry_samples[member.name] = member_result.geometry
        member_limiting_margins[member.name] = (
            member_result.margin_report.limiting_margin
        )
        warnings.extend(member_result.warnings)

    analysis_start_s = max(analysis_starts_s)
    analysis_end_s = min(analysis_ends_s)
    if analysis_end_s <= analysis_start_s:
        raise InvalidScenarioError(
            "Constellation member geometry timelines do not overlap"
        )

    access_summaries = aggregate_access_summaries(
        member_access_windows=member_access_windows,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
        ground_sites=configured_ground_sites,
    )
    link_summaries, member_link_summaries = aggregate_link_summaries(
        member_link_windows=member_link_windows,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
        ground_sites=configured_ground_sites,
    )
    coverage_map_summaries = aggregate_coverage_maps(
        member_geometry_samples=member_geometry_samples,
        coverage_maps=scenario.coverage_maps,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    fleet_margin_report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=scenario.coverage_requirements,
        member_limiting_margins=member_limiting_margins,
        coverage_map_summaries=coverage_map_summaries,
        coverage_maps=scenario.coverage_maps,
    )
    return ConstellationTwinResult(
        scenario_id=scenario.scenario_id,
        members=tuple(member_results),
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        member_link_summaries=member_link_summaries,
        coverage_map_summaries=coverage_map_summaries,
        fleet_margin_report=fleet_margin_report,
        metadata={
            "analysis_window_s": {
                "start_s": analysis_start_s,
                "end_s": analysis_end_s,
            }
        },
        warnings=_deduplicate_warnings(warnings),
    )


def _validate_coverage_requirement_sites(
    scenario: ConstellationTwinScenario,
    configured_ground_sites: set[str],
) -> None:
    missing_sites = sorted(
        requirement.ground_site
        for requirement in scenario.coverage_requirements
        if requirement.ground_site not in configured_ground_sites
    )
    if missing_sites:
        raise InvalidScenarioError(
            "Constellation coverage requirements reference unconfigured ground sites: "
            + ", ".join(missing_sites)
        )


def _deduplicate_warnings(warnings: Iterable[str]) -> list[str]:
    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        unique_warnings.append(warning)
    return unique_warnings


def aggregate_access_summaries(
    member_access_windows: Mapping[str, Iterable[AccessWindow]],
    analysis_start_s: float,
    analysis_end_s: float,
    ground_sites: Iterable[str] = (),
) -> tuple[FleetAccessSummary, ...]:
    """Summarize fleet-level access over the shared analysis interval."""
    _validate_analysis_interval(analysis_start_s, analysis_end_s)
    analysis_duration_s = analysis_end_s - analysis_start_s
    intervals_by_site: dict[str, list[tuple[float, float, str]]] = {}
    sites: set[str] = set(ground_sites)

    for member_name, windows in member_access_windows.items():
        for window in windows:
            sites.add(window.ground_site)
            clipped = _clip_interval(
                window.start_s,
                window.end_s,
                analysis_start_s,
                analysis_end_s,
            )
            if clipped is None:
                continue
            clipped_start_s, clipped_end_s = clipped
            intervals_by_site.setdefault(window.ground_site, []).append(
                (clipped_start_s, clipped_end_s, member_name)
            )

    summaries: list[FleetAccessSummary] = []
    for ground_site in sorted(sites):
        site_intervals = intervals_by_site.get(ground_site, [])
        merged_intervals = _merge_intervals(
            (start_s, end_s) for start_s, end_s, _member_name in site_intervals
        )
        total_access_duration_s = sum(
            end_s - start_s for start_s, end_s in merged_intervals
        )
        gaps_s = _gaps_s(
            merged_intervals,
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
        )
        summaries.append(
            FleetAccessSummary(
                ground_site=ground_site,
                total_access_duration_s=total_access_duration_s,
                longest_gap_s=max(gaps_s, default=0.0),
                mean_gap_s=sum(gaps_s) / len(gaps_s) if gaps_s else 0.0,
                max_simultaneous_spacecraft=_max_simultaneous_members(site_intervals),
                coverage_fraction=total_access_duration_s / analysis_duration_s,
            )
        )
    return tuple(summaries)


def aggregate_coverage_maps(
    member_geometry_samples: Mapping[str, Iterable[TimelineGeometrySample]],
    coverage_maps: Iterable[ConstellationCoverageMapConfig],
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[CoverageMapSummary, ...]:
    """Summarize deterministic nadir sensor coverage over configured target grids."""
    _validate_analysis_interval(analysis_start_s, analysis_end_s)
    analysis_duration_s = analysis_end_s - analysis_start_s
    summaries: list[CoverageMapSummary] = []

    member_samples = {
        member_name: tuple(sorted(samples, key=lambda sample: sample.elapsed_s))
        for member_name, samples in member_geometry_samples.items()
    }
    for coverage_map in coverage_maps:
        target_summaries: list[CoverageMapTargetSummary] = []
        for target in coverage_map.targets:
            target_intervals: list[tuple[float, float, str]] = []
            for member_name, samples in member_samples.items():
                target_intervals.extend(
                    _visible_intervals_for_target(
                        member_name=member_name,
                        samples=samples,
                        target=target,
                        sensor=coverage_map.sensor,
                        analysis_start_s=analysis_start_s,
                        analysis_end_s=analysis_end_s,
                    )
                )

            merged_intervals = _merge_intervals(
                (start_s, end_s) for start_s, end_s, _member_name in target_intervals
            )
            total_covered_duration_s = sum(
                end_s - start_s for start_s, end_s in merged_intervals
            )
            gaps_s = _gaps_s(
                merged_intervals,
                analysis_start_s=analysis_start_s,
                analysis_end_s=analysis_end_s,
            )
            target_summaries.append(
                CoverageMapTargetSummary(
                    target_name=target.name,
                    total_covered_duration_s=total_covered_duration_s,
                    longest_gap_s=max(gaps_s, default=0.0),
                    mean_gap_s=sum(gaps_s) / len(gaps_s) if gaps_s else 0.0,
                    max_simultaneous_spacecraft=_max_simultaneous_members(
                        target_intervals
                    ),
                    coverage_fraction=total_covered_duration_s / analysis_duration_s,
                )
            )

        coverage_fractions = [
            target.coverage_fraction for target in target_summaries
        ]
        summaries.append(
            CoverageMapSummary(
                name=coverage_map.name,
                sensor_name=coverage_map.sensor.name,
                target_count=len(target_summaries),
                covered_target_count=sum(
                    1
                    for target in target_summaries
                    if target.total_covered_duration_s > 0.0
                ),
                mean_coverage_fraction=(
                    sum(coverage_fractions) / len(coverage_fractions)
                    if coverage_fractions
                    else 0.0
                ),
                minimum_target_coverage_fraction=min(
                    coverage_fractions,
                    default=0.0,
                ),
                maximum_target_gap_s=max(
                    (target.longest_gap_s for target in target_summaries),
                    default=0.0,
                ),
                max_simultaneous_spacecraft=max(
                    (
                        target.max_simultaneous_spacecraft
                        for target in target_summaries
                    ),
                    default=0,
                ),
                target_summaries=tuple(target_summaries),
            )
        )
    return tuple(summaries)


def aggregate_link_summaries(
    member_link_windows: Mapping[str, Iterable[LinkBudgetWindow]],
    analysis_start_s: float,
    analysis_end_s: float,
    ground_sites: Iterable[str] = (),
) -> tuple[tuple[FleetLinkSummary, ...], tuple[MemberLinkSummary, ...]]:
    """Summarize downlink volume and worst link margin over the shared interval."""
    _validate_analysis_interval(analysis_start_s, analysis_end_s)
    sites: set[str] = set(ground_sites)
    fleet_data_by_site: dict[str, float] = {}
    fleet_worst_by_site: dict[str, float | None] = {}
    member_data: dict[str, float] = {}
    member_worst: dict[str, float | None] = {}

    for member_name, windows in member_link_windows.items():
        member_data.setdefault(member_name, 0.0)
        member_worst.setdefault(member_name, None)
        for window in windows:
            sites.add(window.ground_site)
            clipped = _clip_interval(
                window.start_s,
                window.end_s,
                analysis_start_s,
                analysis_end_s,
            )
            if clipped is None:
                continue

            clipped_start_s, clipped_end_s = clipped
            clipped_duration_s = clipped_end_s - clipped_start_s
            window_duration_s = window.end_s - window.start_s
            if window_duration_s <= 0.0:
                continue
            data_volume_mbit = (
                window.data_volume_mbit * clipped_duration_s / window_duration_s
            )
            fleet_data_by_site[window.ground_site] = (
                fleet_data_by_site.get(window.ground_site, 0.0) + data_volume_mbit
            )
            fleet_worst_by_site[window.ground_site] = _minimum_optional(
                fleet_worst_by_site.get(window.ground_site),
                window.worst_ebn0_margin_db,
            )
            member_data[member_name] += data_volume_mbit
            member_worst[member_name] = _minimum_optional(
                member_worst[member_name],
                window.worst_ebn0_margin_db,
            )

    fleet_summaries = tuple(
        FleetLinkSummary(
            ground_site=ground_site,
            total_data_volume_mbit=fleet_data_by_site.get(ground_site, 0.0),
            worst_ebn0_margin_db=fleet_worst_by_site.get(ground_site),
        )
        for ground_site in sorted(sites)
    )
    member_summaries = tuple(
        MemberLinkSummary(
            member_name=member_name,
            total_data_volume_mbit=member_data[member_name],
            worst_ebn0_margin_db=member_worst[member_name],
        )
        for member_name in sorted(member_data)
    )
    return fleet_summaries, member_summaries


def build_fleet_margin_report(
    access_summaries: Iterable[FleetAccessSummary],
    link_summaries: Iterable[FleetLinkSummary],
    coverage_requirements: Iterable[ConstellationCoverageRequirement],
    member_limiting_margins: Mapping[str, DesignMargin],
    coverage_map_summaries: Iterable[CoverageMapSummary] = (),
    coverage_maps: Iterable[ConstellationCoverageMapConfig] = (),
) -> DesignMarginReport:
    """Build a deterministic fleet margin report from aggregated twin evidence."""
    access_by_site = {summary.ground_site: summary for summary in access_summaries}
    link_by_site = {summary.ground_site: summary for summary in link_summaries}
    coverage_map_by_name = {
        summary.name: summary for summary in coverage_map_summaries
    }
    requirements_by_site = {
        requirement.ground_site: requirement for requirement in coverage_requirements
    }

    margins: list[DesignMargin] = []
    coverage_sites = set(access_by_site) | set(requirements_by_site)
    for ground_site in sorted(coverage_sites):
        requirement = requirements_by_site.get(ground_site)
        access_summary = access_by_site.get(ground_site)
        coverage_threshold = (
            requirement.minimum_coverage_fraction if requirement is not None else 0.0
        )
        coverage_fraction = (
            access_summary.coverage_fraction if access_summary is not None else 0.0
        )
        coverage_margin = coverage_fraction - coverage_threshold
        margins.append(
            DesignMargin(
                name=f"fleet_coverage_fraction_{ground_site}",
                value=coverage_fraction,
                threshold=coverage_threshold,
                margin=coverage_margin,
                status=_status(coverage_margin, warn_threshold=0.05),
            )
        )
        if requirement is not None and requirement.maximum_revisit_gap_s is not None:
            longest_gap_s = (
                access_summary.longest_gap_s
                if access_summary is not None
                else requirement.maximum_revisit_gap_s + 1.0
            )
            revisit_margin_s = requirement.maximum_revisit_gap_s - longest_gap_s
            margins.append(
                DesignMargin(
                    name=f"fleet_longest_gap_s_{ground_site}",
                    value=longest_gap_s,
                    threshold=requirement.maximum_revisit_gap_s,
                    margin=revisit_margin_s,
                    status=_status(revisit_margin_s, warn_threshold=60.0),
                )
            )

    for coverage_map in sorted(coverage_maps, key=lambda item: item.name):
        summary = coverage_map_by_name.get(coverage_map.name)
        minimum_target_coverage_fraction = (
            summary.minimum_target_coverage_fraction if summary is not None else 0.0
        )
        coverage_margin = (
            minimum_target_coverage_fraction
            - coverage_map.minimum_target_coverage_fraction
        )
        margins.append(
            DesignMargin(
                name=f"coverage_map_min_fraction_{coverage_map.name}",
                value=minimum_target_coverage_fraction,
                threshold=coverage_map.minimum_target_coverage_fraction,
                margin=coverage_margin,
                status=_status(coverage_margin, warn_threshold=0.05),
            )
        )
        if coverage_map.maximum_target_revisit_gap_s is not None:
            maximum_target_gap_s = (
                summary.maximum_target_gap_s
                if summary is not None
                else coverage_map.maximum_target_revisit_gap_s + 1.0
            )
            revisit_margin_s = (
                coverage_map.maximum_target_revisit_gap_s
                - maximum_target_gap_s
            )
            margins.append(
                DesignMargin(
                    name=f"coverage_map_max_gap_s_{coverage_map.name}",
                    value=maximum_target_gap_s,
                    threshold=coverage_map.maximum_target_revisit_gap_s,
                    margin=revisit_margin_s,
                    status=_status(revisit_margin_s, warn_threshold=60.0),
                )
            )

    represented_sites = (
        set(access_by_site)
        | set(link_by_site)
        | set(requirements_by_site)
    )
    for ground_site in sorted(represented_sites):
        margins.append(_link_margin(ground_site, link_by_site.get(ground_site)))

    for member_name in sorted(member_limiting_margins):
        margins.append(
            _member_margin(member_name, member_limiting_margins[member_name])
        )

    if not margins:
        margins.append(
            DesignMargin(
                name="fleet_aggregation_inputs",
                value=0.0,
                threshold=0.0,
                margin=0.0,
                status=TwinMarginStatus.PASS,
            )
        )

    limiting_margin = min(margins, key=_limiting_key)
    return DesignMarginReport(
        margins=tuple(margins),
        limiting_margin=limiting_margin,
    )


def _visible_intervals_for_target(
    *,
    member_name: str,
    samples: Iterable[TimelineGeometrySample],
    target: ConstellationCoverageTargetConfig,
    sensor: ConstellationCoverageSensorConfig,
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[tuple[float, float, str], ...]:
    intervals: list[tuple[float, float, str]] = []
    active_elapsed_s: list[float] = []
    for sample in samples:
        if sample.elapsed_s < analysis_start_s or sample.elapsed_s > analysis_end_s:
            continue
        if _target_visible_from_sample(sample, target, sensor):
            active_elapsed_s.append(sample.elapsed_s)
            continue
        if active_elapsed_s:
            intervals.append(
                _coverage_interval_from_active(
                    active_elapsed_s,
                    member_name=member_name,
                    analysis_end_s=analysis_end_s,
                )
            )
            active_elapsed_s = []
    if active_elapsed_s:
        intervals.append(
            _coverage_interval_from_active(
                active_elapsed_s,
                member_name=member_name,
                analysis_end_s=analysis_end_s,
            )
        )
    return tuple(
        interval for interval in intervals if interval[1] > interval[0]
    )


def _coverage_interval_from_active(
    active_elapsed_s: list[float],
    *,
    member_name: str,
    analysis_end_s: float,
) -> tuple[float, float, str]:
    start_s = active_elapsed_s[0]
    end_s = active_elapsed_s[-1]
    if end_s <= start_s:
        end_s = min(analysis_end_s, start_s + 1.0)
    return start_s, end_s, member_name


def _target_visible_from_sample(
    sample: TimelineGeometrySample,
    target: ConstellationCoverageTargetConfig,
    sensor: ConstellationCoverageSensorConfig,
) -> bool:
    target_site = GroundSiteConfig(
        name=target.name,
        latitude_deg=target.latitude_deg,
        longitude_deg=target.longitude_deg,
        altitude_m=target.altitude_m,
        minimum_elevation_deg=sensor.minimum_elevation_deg,
    )
    elevation_deg, range_km = elevation_and_range_km(
        sample.position_km,
        target_site,
        sample.elapsed_s,
    )
    if elevation_deg < sensor.minimum_elevation_deg:
        return False
    if sensor.maximum_range_km is not None and range_km > sensor.maximum_range_km:
        return False

    target_position_km = _coverage_target_position_eci_km(
        target,
        elapsed_s=sample.elapsed_s,
    )
    target_vector_km: tuple[float, float, float] = (
        target_position_km[0] - sample.position_km[0],
        target_position_km[1] - sample.position_km[1],
        target_position_km[2] - sample.position_km[2],
    )
    nadir_vector_km: tuple[float, float, float] = (
        -sample.position_km[0],
        -sample.position_km[1],
        -sample.position_km[2],
    )
    off_nadir_deg = _angle_between_deg(nadir_vector_km, target_vector_km)
    return off_nadir_deg <= sensor.field_of_view_half_angle_deg + 1.0e-9


def _coverage_target_position_eci_km(
    target: ConstellationCoverageTargetConfig,
    *,
    elapsed_s: float,
) -> tuple[float, float, float]:
    latitude = radians(target.latitude_deg)
    longitude = radians(target.longitude_deg) + 7.2921159e-5 * elapsed_s
    radius_km = R_EARTH_KM + target.altitude_m / 1000.0
    return (
        radius_km * cos(latitude) * cos(longitude),
        radius_km * cos(latitude) * sin(longitude),
        radius_km * sin(latitude),
    )


def _angle_between_deg(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    first_norm = sqrt(sum(component * component for component in first))
    second_norm = sqrt(sum(component * component for component in second))
    cosine = sum(first[index] * second[index] for index in range(3)) / (
        first_norm * second_norm
    )
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _validate_analysis_interval(analysis_start_s: float, analysis_end_s: float) -> None:
    if not isfinite(analysis_start_s) or not isfinite(analysis_end_s):
        raise ValueError("analysis interval bounds must be finite")
    if analysis_end_s <= analysis_start_s:
        raise ValueError("analysis_end_s must be greater than analysis_start_s")


def _clip_interval(
    start_s: float,
    end_s: float,
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[float, float] | None:
    clipped_start_s = max(start_s, analysis_start_s)
    clipped_end_s = min(end_s, analysis_end_s)
    if clipped_end_s <= clipped_start_s:
        return None
    return clipped_start_s, clipped_end_s


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start_s, end_s in sorted(intervals):
        if not merged or start_s > merged[-1][1]:
            merged.append((start_s, end_s))
            continue
        previous_start_s, previous_end_s = merged[-1]
        merged[-1] = (previous_start_s, max(previous_end_s, end_s))
    return tuple(merged)


def _gaps_s(
    merged_intervals: tuple[tuple[float, float], ...],
    *,
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[float, ...]:
    if not merged_intervals:
        return (analysis_end_s - analysis_start_s,)

    gaps: list[float] = []
    cursor_s = analysis_start_s
    for start_s, end_s in merged_intervals:
        if start_s > cursor_s:
            gaps.append(start_s - cursor_s)
        cursor_s = max(cursor_s, end_s)
    if analysis_end_s > cursor_s:
        gaps.append(analysis_end_s - cursor_s)
    return tuple(gaps)


def _max_simultaneous_members(
    intervals: Iterable[tuple[float, float, str]],
) -> int:
    events: list[tuple[float, int, str]] = []
    for start_s, end_s, member_name in intervals:
        events.append((start_s, 1, member_name))
        events.append((end_s, 0, member_name))

    active_counts: dict[str, int] = {}
    max_active = 0
    for _elapsed_s, event_order, member_name in sorted(events):
        if event_order == 0:
            remaining_count = active_counts[member_name] - 1
            if remaining_count == 0:
                del active_counts[member_name]
            else:
                active_counts[member_name] = remaining_count
            continue

        active_counts[member_name] = active_counts.get(member_name, 0) + 1
        max_active = max(max_active, len(active_counts))
    return max_active


def _minimum_optional(current: float | None, candidate: float) -> float:
    if current is None:
        return candidate
    return min(current, candidate)


def _link_margin(
    ground_site: str,
    link_summary: FleetLinkSummary | None,
) -> DesignMargin:
    if link_summary is None or link_summary.worst_ebn0_margin_db is None:
        return DesignMargin(
            name=f"fleet_link_margin_db_{ground_site}",
            value=0.0,
            threshold=0.0,
            margin=-1.0,
            status=TwinMarginStatus.FAIL,
        )

    margin_db = link_summary.worst_ebn0_margin_db
    return DesignMargin(
        name=f"fleet_link_margin_db_{ground_site}",
        value=margin_db,
        threshold=0.0,
        margin=margin_db,
        status=_status(margin_db, warn_threshold=3.0),
    )


def _member_margin(member_name: str, margin: DesignMargin) -> DesignMargin:
    return DesignMargin(
        name=f"member_{member_name}_{margin.name}",
        value=margin.value,
        threshold=margin.threshold,
        margin=margin.margin,
        status=margin.status,
    )


def _status(margin: float, warn_threshold: float) -> TwinMarginStatus:
    if margin < 0.0:
        return TwinMarginStatus.FAIL
    if margin <= warn_threshold:
        return TwinMarginStatus.WARN
    return TwinMarginStatus.PASS


def _limiting_key(margin: DesignMargin) -> tuple[int, float, str]:
    severity = {
        TwinMarginStatus.FAIL: 0,
        TwinMarginStatus.WARN: 1,
        TwinMarginStatus.PASS: 2,
    }[margin.status]
    normalizer = abs(margin.threshold) if margin.threshold != 0.0 else 1.0
    return severity, margin.margin / normalizer, margin.name
