from astro_twin.margins import build_margin_report
from astro_twin.models import (
    ADCSConfig,
    ADCSSample,
    LinkBudgetWindow,
    MassBudgetSummary,
    MissionMode,
    PowerConfig,
    PowerSample,
    SpacecraftBusConfig,
    ThermalNodeConfig,
    ThermalSample,
)


def test_build_margin_report_identifies_limiting_margin() -> None:
    report = build_margin_report(
        spacecraft=SpacecraftBusConfig(
            name="ObserverSat",
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=5.0,
            mass_margin_fraction_required=0.01,
        ),
        power_config=PowerConfig(
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
        power=(
            PowerSample(
                elapsed_s=0.0,
                mode=MissionMode.IDLE,
                generated_w=600.0,
                load_w=120.0,
                battery_soc_fraction=0.5,
                net_power_w=480.0,
            ),
        ),
        thermal=(ThermalSample(elapsed_s=0.0, node_temperatures_k={"bus": 312.0}),),
        adcs=(
            ADCSSample(
                elapsed_s=0.0,
                pointing_error_deg=0.08,
                pointing_margin_deg=0.07,
                torque_margin_n_m=0.05,
                slew_rate_margin_deg_s=0.15,
                actuator_utilization_fraction=0.375,
                actuator_utilization_margin_fraction=0.325,
            ),
        ),
        adcs_config=ADCSConfig(
            pointing_mode="nadir",
            max_pointing_error_deg=0.08,
            pointing_requirement_deg=0.15,
            max_torque_n_m=0.08,
            required_slew_torque_n_m=0.03,
            max_slew_rate_deg_s=0.2,
            required_slew_rate_deg_s=0.05,
            maximum_actuator_utilization_fraction=0.7,
        ),
        link_windows=(
            LinkBudgetWindow(
                link_name="xband-downlink",
                ground_site="goldstone",
                start_s=0.0,
                end_s=60.0,
                duration_s=60.0,
                worst_ebn0_margin_db=3.0,
                data_volume_mbit=120.0,
            ),
        ),
        mass_budget=MassBudgetSummary(
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=5.0,
            configured_wet_mass_kg=150.0,
            itemized_base_mass_kg=130.0,
            itemized_contingency_mass_kg=14.0,
            itemized_total_mass_kg=144.0,
            dry_payload_reference_mass_kg=145.0,
            dry_payload_margin_kg=1.0,
            category_totals_kg={"bus": 119.0, "payload": 25.0},
        ),
    )

    assert report.limiting_margin.name == "thermal_bus_hot_margin_k"
    assert report.limiting_margin.margin == 1.0
    margin_by_name = {margin.name: margin for margin in report.margins}
    assert "slew_rate_margin_deg_s" in margin_by_name
    assert "actuator_utilization_margin_fraction" in margin_by_name
    assert "mass_budget_rollup_margin_kg" in margin_by_name
