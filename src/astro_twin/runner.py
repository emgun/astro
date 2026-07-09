from __future__ import annotations

from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local
from astro_twin.adcs import compute_adcs_timeline
from astro_twin.coverage import access_windows_from_samples
from astro_twin.geometry import build_geometry_timeline, elevation_and_range_km
from astro_twin.link_budget import compute_link_budget_windows
from astro_twin.margins import build_margin_report
from astro_twin.models import (
    DigitalTwinResult,
    DigitalTwinScenario,
    GroundSiteConfig,
    MissionMode,
    TimelineGeometrySample,
)
from astro_twin.power import compute_power_timeline
from astro_twin.thermal import compute_thermal_timeline

_DESIGN_SCREENING_WARNING = (
    "Digital twin v1 is deterministic design-screening evidence, not flight qualification."
)
_COVERAGE_GEOMETRY_WARNING = (
    "Coverage geometry uses spherical Earth and uniform Earth rotation screening assumptions."
)


def run_digital_twin(scenario: DigitalTwinScenario) -> DigitalTwinResult:
    orbit_scenario = load_scenario(scenario.orbit_scenario)
    trajectory = propagate_local(orbit_scenario)
    geometry = build_geometry_timeline(trajectory)
    mode_by_elapsed_s = _mode_by_elapsed_s(scenario, geometry)
    power = compute_power_timeline(scenario.power, geometry, mode_by_elapsed_s)
    thermal = compute_thermal_timeline(scenario.thermal_nodes, geometry, power)
    adcs = compute_adcs_timeline(scenario.adcs, geometry)
    access_windows = tuple(
        window
        for site in scenario.ground_sites
        for window in access_windows_from_samples(site, _access_samples_for_site(site, geometry))
    )
    link_windows = compute_link_budget_windows(scenario.links, access_windows)
    margin_report = build_margin_report(
        spacecraft=scenario.spacecraft,
        power_config=scenario.power,
        thermal_nodes=scenario.thermal_nodes,
        power=power,
        thermal=thermal,
        adcs=adcs,
        adcs_config=scenario.adcs,
        link_windows=link_windows,
    )
    return DigitalTwinResult(
        scenario_id=scenario.scenario_id,
        geometry=geometry,
        power=power,
        thermal=thermal,
        adcs=adcs,
        access_windows=access_windows,
        link_windows=link_windows,
        margin_report=margin_report,
        metadata={
            "orbit_scenario": scenario.orbit_scenario,
            "orbit_backend": trajectory.backend,
        },
        warnings=[_DESIGN_SCREENING_WARNING, _COVERAGE_GEOMETRY_WARNING],
    )


def _mode_by_elapsed_s(
    scenario: DigitalTwinScenario,
    geometry: tuple[TimelineGeometrySample, ...],
) -> dict[float, MissionMode]:
    mode_by_elapsed_s: dict[float, MissionMode] = {}
    for sample in geometry:
        mode = MissionMode.IDLE
        for scheduled in scenario.mode_schedule:
            if scheduled.start_s <= sample.elapsed_s <= scheduled.end_s:
                mode = scheduled.mode
        mode_by_elapsed_s[sample.elapsed_s] = mode
    return mode_by_elapsed_s


def _access_samples_for_site(
    site: GroundSiteConfig,
    geometry: tuple[TimelineGeometrySample, ...],
) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    for sample in geometry:
        elevation_deg, range_km = elevation_and_range_km(
            position_km=sample.position_km,
            site=site,
            elapsed_s=sample.elapsed_s,
        )
        samples.append((sample.elapsed_s, elevation_deg, range_km))
    return samples
