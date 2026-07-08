from __future__ import annotations

from typing import Any

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


class ConstellationTwinScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    members: tuple[ConstellationMemberConfig, ...] = Field(min_length=1)
    coverage_requirements: tuple[ConstellationCoverageRequirement, ...] = Field(
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


class ConstellationTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "constellation_digital_twin_v1"
    members: tuple[MemberTwinResult, ...]
    access_summaries: tuple[FleetAccessSummary, ...]
    link_summaries: tuple[FleetLinkSummary, ...]
    member_link_summaries: tuple[MemberLinkSummary, ...]
    fleet_margin_report: DesignMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
