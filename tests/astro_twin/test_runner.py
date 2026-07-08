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
    assert result.margin_report.limiting_margin.name
