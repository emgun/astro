from pathlib import Path

from astro_twin.io import load_twin_scenario


def test_load_twin_scenario_reads_reference_example() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")

    assert scenario.scenario_id == "leo-observer"
    assert scenario.orbit_scenario == "examples/scenarios/leo_two_body.yaml"
    assert scenario.power.battery_capacity_wh == 1200.0


def test_load_twin_scenario_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    try:
        load_twin_scenario(path)
    except Exception as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("expected invalid scenario error")
