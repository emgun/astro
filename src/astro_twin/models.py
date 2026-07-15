from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from astro_core.models import AstroModel


class TwinMarginStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MissionMode(StrEnum):
    IDLE = "idle"
    PAYLOAD = "payload"
    DOWNLINK = "downlink"


class MissionModeSchedule(AstroModel):
    mode: MissionMode
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> MissionModeSchedule:
        if self.end_s <= self.start_s:
            raise ValueError("Mission mode schedule end_s must be greater than start_s")
        return self


class PowerLoadSchedule(AstroModel):
    name: str = Field(min_length=1)
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    additional_load_w: FiniteFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> PowerLoadSchedule:
        if self.end_s <= self.start_s:
            raise ValueError("Power load schedule end_s must be greater than start_s")
        return self


class MassBudgetItemConfig(AstroModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    mass_kg: FiniteFloat = Field(ge=0.0)
    contingency_fraction: FiniteFloat = Field(ge=0.0, default=0.0)


class SpacecraftBusConfig(AstroModel):
    name: str = Field(min_length=1)
    dry_mass_kg: FiniteFloat = Field(gt=0.0)
    payload_mass_kg: FiniteFloat = Field(ge=0.0)
    propellant_mass_kg: FiniteFloat = Field(ge=0.0, default=0.0)
    mass_margin_fraction_required: FiniteFloat = Field(ge=0.0, default=0.2)
    mass_budget_items: tuple[MassBudgetItemConfig, ...] = Field(default_factory=tuple)


class PowerConfig(AstroModel):
    solar_array_area_m2: FiniteFloat = Field(gt=0.0)
    solar_array_efficiency: FiniteFloat = Field(gt=0.0, le=1.0)
    battery_capacity_wh: FiniteFloat = Field(gt=0.0)
    initial_battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    minimum_battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    idle_load_w: FiniteFloat = Field(ge=0.0)
    payload_load_w: FiniteFloat = Field(ge=0.0)
    downlink_load_w: FiniteFloat = Field(ge=0.0)
    battery_charge_efficiency: FiniteFloat = Field(gt=0.0, le=1.0, default=1.0)
    battery_discharge_efficiency: FiniteFloat = Field(gt=0.0, le=1.0, default=1.0)


class ThermalNodeConfig(AstroModel):
    name: str = Field(min_length=1)
    thermal_mass_j_k: FiniteFloat = Field(gt=0.0)
    radiator_area_m2: FiniteFloat = Field(gt=0.0)
    absorptivity: FiniteFloat = Field(ge=0.0, le=1.0)
    emissivity: FiniteFloat = Field(gt=0.0, le=1.0)
    initial_temperature_k: FiniteFloat = Field(gt=0.0)
    minimum_temperature_k: FiniteFloat = Field(gt=0.0)
    maximum_temperature_k: FiniteFloat = Field(gt=0.0)
    internal_heat_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    albedo_flux_w_m2: FiniteFloat = Field(ge=0.0, default=0.0)
    planet_ir_flux_w_m2: FiniteFloat = Field(ge=0.0, default=0.0)
    mode_internal_heat_scale: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def maximum_temperature_must_exceed_minimum(self) -> ThermalNodeConfig:
        if self.maximum_temperature_k <= self.minimum_temperature_k:
            raise ValueError("maximum_temperature_k must exceed minimum_temperature_k")
        return self


class ADCSConfig(AstroModel):
    pointing_mode: Literal["nadir", "inertial", "ground_station_track"]
    max_pointing_error_deg: FiniteFloat = Field(ge=0.0)
    pointing_requirement_deg: FiniteFloat = Field(gt=0.0)
    max_torque_n_m: FiniteFloat = Field(gt=0.0)
    required_slew_torque_n_m: FiniteFloat = Field(ge=0.0)
    max_slew_rate_deg_s: FiniteFloat = Field(gt=0.0, default=1.0)
    required_slew_rate_deg_s: FiniteFloat = Field(ge=0.0, default=0.0)
    maximum_actuator_utilization_fraction: FiniteFloat = Field(
        gt=0.0,
        le=1.0,
        default=1.0,
    )


class GroundSiteConfig(AstroModel):
    name: str = Field(min_length=1)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    altitude_m: FiniteFloat = 0.0
    minimum_elevation_deg: FiniteFloat = Field(ge=0.0, le=90.0)


class LinkBudgetConfig(AstroModel):
    name: str = Field(min_length=1)
    ground_site: str = Field(min_length=1)
    frequency_ghz: FiniteFloat = Field(gt=0.0)
    eirp_dbw: FiniteFloat
    receiver_g_over_t_db_k: FiniteFloat
    data_rate_bps: FiniteFloat = Field(gt=0.0)
    required_ebn0_db: FiniteFloat
    implementation_loss_db: FiniteFloat = Field(ge=0.0, default=2.0)


class DigitalTwinScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    orbit_scenario: str = Field(min_length=1)
    spacecraft: SpacecraftBusConfig
    power: PowerConfig
    thermal_nodes: tuple[ThermalNodeConfig, ...] = Field(min_length=1)
    adcs: ADCSConfig
    ground_sites: tuple[GroundSiteConfig, ...] = Field(min_length=1)
    links: tuple[LinkBudgetConfig, ...] = Field(min_length=1)
    mode_schedule: tuple[MissionModeSchedule, ...] = Field(default_factory=tuple)
    power_loads: tuple[PowerLoadSchedule, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def links_must_reference_configured_ground_sites(self) -> DigitalTwinScenario:
        site_names = {site.name for site in self.ground_sites}
        for link in self.links:
            if link.ground_site not in site_names:
                raise ValueError("link ground_site must name a configured ground site")
        return self


class TimelineGeometrySample(AstroModel):
    epoch: datetime
    elapsed_s: FiniteFloat = Field(ge=0.0)
    position_km: tuple[float, float, float]
    altitude_km: FiniteFloat
    sunlit: bool


class PowerSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    mode: MissionMode
    generated_w: FiniteFloat = Field(ge=0.0)
    load_w: FiniteFloat = Field(ge=0.0)
    scheduled_load_w: FiniteFloat = Field(ge=0.0, default=0.0)
    battery_energy_wh: FiniteFloat = Field(ge=0.0, default=0.0)
    battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    net_power_w: float
    battery_energy_change_wh: float = 0.0
    unmet_load_w: FiniteFloat = Field(ge=0.0, default=0.0)
    unmet_energy_wh: FiniteFloat = Field(ge=0.0, default=0.0)
    curtailed_energy_wh: FiniteFloat = Field(ge=0.0, default=0.0)


class ThermalSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    node_temperatures_k: dict[str, float]
    node_heat_balance_w: dict[str, float] = Field(default_factory=dict)


class ADCSSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    pointing_error_deg: FiniteFloat = Field(ge=0.0)
    pointing_margin_deg: float
    torque_margin_n_m: float
    slew_rate_margin_deg_s: float = 0.0
    actuator_utilization_fraction: FiniteFloat = Field(ge=0.0, default=0.0)
    actuator_utilization_margin_fraction: float = 0.0


class AccessWindow(AstroModel):
    ground_site: str
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    duration_s: FiniteFloat = Field(gt=0.0)
    max_elevation_deg: FiniteFloat
    min_range_km: FiniteFloat = Field(gt=0.0)


class LinkBudgetWindow(AstroModel):
    link_name: str
    ground_site: str
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    duration_s: FiniteFloat = Field(gt=0.0)
    worst_ebn0_margin_db: float
    data_volume_mbit: FiniteFloat = Field(ge=0.0)


class DesignMargin(AstroModel):
    name: str
    value: float
    threshold: float
    margin: float
    unit: str = Field(min_length=1)
    status: TwinMarginStatus


class DesignMarginReport(AstroModel):
    margins: tuple[DesignMargin, ...]
    limiting_margin: DesignMargin


class MassBudgetSummary(AstroModel):
    dry_mass_kg: FiniteFloat = Field(ge=0.0)
    payload_mass_kg: FiniteFloat = Field(ge=0.0)
    propellant_mass_kg: FiniteFloat = Field(ge=0.0)
    configured_wet_mass_kg: FiniteFloat = Field(ge=0.0)
    itemized_base_mass_kg: FiniteFloat = Field(ge=0.0)
    itemized_contingency_mass_kg: FiniteFloat = Field(ge=0.0)
    itemized_total_mass_kg: FiniteFloat = Field(ge=0.0)
    dry_payload_reference_mass_kg: FiniteFloat = Field(ge=0.0)
    dry_payload_margin_kg: float
    category_totals_kg: dict[str, float] = Field(default_factory=dict)


class DigitalTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "integrated_digital_twin_v2"
    geometry: tuple[TimelineGeometrySample, ...]
    power: tuple[PowerSample, ...]
    thermal: tuple[ThermalSample, ...]
    adcs: tuple[ADCSSample, ...]
    access_windows: tuple[AccessWindow, ...]
    link_windows: tuple[LinkBudgetWindow, ...]
    mass_budget: MassBudgetSummary | None = None
    margin_report: DesignMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
