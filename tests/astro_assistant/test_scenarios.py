import pytest

from astro_assistant.scenarios import resolve_local_od_scenario


def test_resolves_explicit_checked_in_scenario_path() -> None:
    resolved = resolve_local_od_scenario(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml"
    )

    assert resolved.path == "examples/scenarios/leo_two_station_angles.yaml"
    assert resolved.scenario_id == "leo-two-station-angles"
    assert resolved.artifact_dir == "/tmp/astro-assistant/leo_two_station_angles"


def test_resolves_known_scenario_alias() -> None:
    resolved = resolve_local_od_scenario("Run the local OD workflow for radiometric media")

    assert resolved.path == "examples/scenarios/leo_radiometric_media.yaml"
    assert resolved.scenario_id == "leo-radiometric-media"


def test_resolves_supported_bare_scenario_filename() -> None:
    resolved = resolve_local_od_scenario("Run local OD on leo_two_station_topocentric.yaml")

    assert resolved.path == "examples/scenarios/leo_two_station_topocentric.yaml"
    assert resolved.scenario_id == "leo-two-station-topocentric"


def test_defaults_to_original_demo_when_no_scenario_is_named() -> None:
    resolved = resolve_local_od_scenario("Run the local OD demo")

    assert resolved.path == "examples/scenarios/leo_two_station_od.yaml"


def test_rejects_paths_outside_examples_scenarios() -> None:
    with pytest.raises(ValueError, match="scenario path must stay under examples/scenarios"):
        resolve_local_od_scenario("Run local OD on /tmp/custom.yaml")


def test_rejects_unknown_scenario_alias() -> None:
    with pytest.raises(ValueError, match="could not resolve a supported local OD scenario"):
        resolve_local_od_scenario("Run local OD on the secret mission scenario")
