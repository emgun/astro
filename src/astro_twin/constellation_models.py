from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from astro_core.models import AstroModel
from astro_twin.models import DesignMarginReport, DigitalTwinResult


class ConstellationMemberConfig(AstroModel):
    name: str = Field(min_length=1)
    twin_scenario: str = Field(min_length=1)


class ConstellationCoverageRequirement(AstroModel):
    ground_site: str = Field(min_length=1)
    minimum_coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0, default=0.0)
    maximum_revisit_gap_s: FiniteFloat | None = Field(default=None, gt=0.0)


class ConstellationCoverageSensorConfig(AstroModel):
    name: str = Field(min_length=1)
    pointing_mode: Literal["nadir"] = "nadir"
    field_of_view_half_angle_deg: FiniteFloat = Field(gt=0.0, le=90.0)
    minimum_elevation_deg: FiniteFloat = Field(ge=0.0, le=90.0, default=0.0)
    maximum_range_km: FiniteFloat | None = Field(default=None, gt=0.0)


class ConstellationCoverageTargetConfig(AstroModel):
    name: str = Field(min_length=1)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    altitude_m: FiniteFloat = 0.0


class ConstellationCoverageMapConfig(AstroModel):
    name: str = Field(min_length=1)
    sensor: ConstellationCoverageSensorConfig
    targets: tuple[ConstellationCoverageTargetConfig, ...] = Field(min_length=1)
    minimum_target_coverage_fraction: FiniteFloat = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
    )
    maximum_target_revisit_gap_s: FiniteFloat | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def target_names_must_be_unique(self) -> ConstellationCoverageMapConfig:
        target_names = [target.name for target in self.targets]
        if len(set(target_names)) != len(target_names):
            raise ValueError("coverage map target names must be unique")
        return self


class ConstellationTwinScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    members: tuple[ConstellationMemberConfig, ...] = Field(min_length=1)
    coverage_requirements: tuple[ConstellationCoverageRequirement, ...] = Field(
        default_factory=tuple
    )
    coverage_maps: tuple[ConstellationCoverageMapConfig, ...] = Field(
        default_factory=tuple
    )

    @model_validator(mode="after")
    def names_must_be_unique(self) -> ConstellationTwinScenario:
        member_names = [member.name for member in self.members]
        if len(set(member_names)) != len(member_names):
            raise ValueError("member names must be unique")
        requirement_sites = [
            requirement.ground_site for requirement in self.coverage_requirements
        ]
        if len(set(requirement_sites)) != len(requirement_sites):
            raise ValueError("coverage requirement ground_site values must be unique")
        coverage_map_names = [coverage_map.name for coverage_map in self.coverage_maps]
        if len(set(coverage_map_names)) != len(coverage_map_names):
            raise ValueError("coverage map names must be unique")
        return self


class FleetAccessSummary(AstroModel):
    ground_site: str
    total_access_duration_s: FiniteFloat = Field(ge=0.0)
    longest_gap_s: FiniteFloat = Field(ge=0.0)
    mean_gap_s: FiniteFloat = Field(ge=0.0)
    max_simultaneous_spacecraft: int = Field(ge=0)
    coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0)


class FleetLinkSummary(AstroModel):
    ground_site: str
    total_data_volume_mbit: FiniteFloat = Field(ge=0.0)
    worst_ebn0_margin_db: float | None = None


class MemberLinkSummary(AstroModel):
    member_name: str
    total_data_volume_mbit: FiniteFloat = Field(ge=0.0)
    worst_ebn0_margin_db: float | None = None


class MemberTwinResult(AstroModel):
    member_name: str
    result: DigitalTwinResult


class CoverageMapTargetSummary(AstroModel):
    target_name: str
    total_covered_duration_s: FiniteFloat = Field(ge=0.0)
    longest_gap_s: FiniteFloat = Field(ge=0.0)
    mean_gap_s: FiniteFloat = Field(ge=0.0)
    max_simultaneous_spacecraft: int = Field(ge=0)
    coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0)


class CoverageMapSummary(AstroModel):
    name: str
    sensor_name: str
    target_count: int = Field(ge=0)
    covered_target_count: int = Field(ge=0)
    mean_coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    minimum_target_coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    maximum_target_gap_s: FiniteFloat = Field(ge=0.0)
    max_simultaneous_spacecraft: int = Field(ge=0)
    target_summaries: tuple[CoverageMapTargetSummary, ...]


class ConstellationTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "constellation_digital_twin_v2"
    members: tuple[MemberTwinResult, ...]
    access_summaries: tuple[FleetAccessSummary, ...]
    link_summaries: tuple[FleetLinkSummary, ...]
    member_link_summaries: tuple[MemberLinkSummary, ...]
    coverage_map_summaries: tuple[CoverageMapSummary, ...] = Field(
        default_factory=tuple
    )
    fleet_margin_report: DesignMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
