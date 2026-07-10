from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local
from astro_twin.io import load_twin_scenario
from astro_twin.runner import run_digital_twin


def test_run_digital_twin_returns_integrated_result() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")

    result = run_digital_twin(scenario)

    assert result.workflow == "integrated_digital_twin_v1"
    assert len(result.geometry) > 1
    assert len(result.power) == len(result.geometry)
    assert len(result.thermal) == len(result.geometry)
    assert len(result.adcs) == len(result.geometry)
    assert result.access_windows
    assert result.link_windows
    assert result.mass_budget is not None
    assert result.power[0].battery_energy_wh > 0.0
    assert "bus" in result.thermal[0].node_heat_balance_w
    assert result.adcs[0].actuator_utilization_fraction > 0.0
    assert result.margin_report.limiting_margin.name


def test_run_digital_twin_accepts_suite_trajectory_override() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    trajectory = propagate_local(load_scenario("examples/scenarios/leo_two_body.yaml"))

    result = run_digital_twin(scenario, trajectory_override=trajectory)

    assert result.metadata["orbit_trajectory_source"] == "trajectory_override"
    assert result.metadata["orbit_trajectory_scenario_id"] == trajectory.scenario_id
    assert result.geometry[0].epoch == trajectory.samples[0].epoch
