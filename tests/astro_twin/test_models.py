import pytest
from pydantic import ValidationError

from astro_twin.models import (
    ADCSConfig,
    DigitalTwinScenario,
    GroundSiteConfig,
    LinkBudgetConfig,
    MissionMode,
    MissionModeSchedule,
    PowerConfig,
    SpacecraftBusConfig,
    ThermalNodeConfig,
)


def _valid_scenario() -> DigitalTwinScenario:
    return DigitalTwinScenario(
        scenario_id="leo-observer",
        orbit_scenario="examples/scenarios/leo_two_body.yaml",
        spacecraft=SpacecraftBusConfig(
            name="ObserverSat",
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=5.0,
            mass_margin_fraction_required=0.2,
        ),
        power=PowerConfig(
            solar_array_area_m2=2.4,
            solar_array_efficiency=0.29,
            battery_capacity_wh=1200.0,
            initial_battery_soc_fraction=0.85,
            minimum_battery_soc_fraction=0.35,
            idle_load_w=120.0,
            payload_load_w=260.0,
            downlink_load_w=360.0,
        ),
        thermal_nodes=(
            ThermalNodeConfig(
                name="bus",
                thermal_mass_j_k=45000.0,
                radiator_area_m2=1.0,
                absorptivity=0.55,
                emissivity=0.78,
                initial_temperature_k=293.0,
                minimum_temperature_k=273.0,
                maximum_temperature_k=313.0,
                internal_heat_fraction=0.45,
            ),
        ),
        adcs=ADCSConfig(
            pointing_mode="nadir",
            max_pointing_error_deg=0.08,
            pointing_requirement_deg=0.15,
            max_torque_n_m=0.08,
            required_slew_torque_n_m=0.03,
        ),
        ground_sites=(
            GroundSiteConfig(
                name="goldstone",
                latitude_deg=35.2472,
                longitude_deg=-116.7933,
                altitude_m=1000.0,
                minimum_elevation_deg=10.0,
            ),
        ),
        links=(
            LinkBudgetConfig(
                name="xband-downlink",
                ground_site="goldstone",
                frequency_ghz=8.4,
                eirp_dbw=18.0,
                receiver_g_over_t_db_k=22.0,
                data_rate_bps=2_000_000.0,
                required_ebn0_db=6.5,
                implementation_loss_db=2.0,
            ),
        ),
        mode_schedule=(
            MissionModeSchedule(mode=MissionMode.PAYLOAD, start_s=600.0, end_s=1800.0),
            MissionModeSchedule(mode=MissionMode.DOWNLINK, start_s=2400.0, end_s=3000.0),
        ),
    )


def test_digital_twin_scenario_accepts_valid_config() -> None:
    scenario = _valid_scenario()

    assert scenario.scenario_id == "leo-observer"
    assert scenario.links[0].ground_site == "goldstone"


def test_digital_twin_scenario_rejects_unknown_link_site() -> None:
    scenario = _valid_scenario()
    payload = scenario.model_dump()
    payload["links"][0]["ground_site"] = "missing"

    with pytest.raises(ValidationError, match="link ground_site must name a configured ground site"):
        DigitalTwinScenario.model_validate(payload)


def test_thermal_node_rejects_inverted_temperature_limits() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_temperature_k must exceed minimum_temperature_k",
    ):
        ThermalNodeConfig(
            name="battery",
            thermal_mass_j_k=10000.0,
            radiator_area_m2=0.4,
            absorptivity=0.5,
            emissivity=0.8,
            initial_temperature_k=293.0,
            minimum_temperature_k=300.0,
            maximum_temperature_k=290.0,
            internal_heat_fraction=0.2,
        )
