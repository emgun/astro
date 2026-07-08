from astro_twin.constellation import run_constellation_twin
from astro_twin.constellation_io import load_constellation_twin_scenario


def test_run_constellation_twin_returns_fleet_result() -> None:
    scenario = load_constellation_twin_scenario(
        "examples/twin/constellation_leo_observers.yaml"
    )

    result = run_constellation_twin(scenario)

    assert result.workflow == "constellation_digital_twin_v1"
    assert len(result.members) == 2
    assert result.metadata["analysis_window_s"] == {
        "start_s": 0.0,
        "end_s": 600.0,
    }
    assert result.access_summaries
    assert result.link_summaries
    assert result.member_link_summaries
    assert result.fleet_margin_report.limiting_margin.name
    assert any("design-screening" in warning for warning in result.warnings)
