from astro_assistant.capabilities import classify_local_od_support


def test_classifies_supported_local_od_scenario() -> None:
    report = classify_local_od_support("Run local OD on leo_two_station_topocentric.yaml")

    assert report.supported is True
    assert report.code == "supported"
    assert report.scenario_path == "examples/scenarios/leo_two_station_topocentric.yaml"
    assert report.scenario_id == "leo-two-station-topocentric"
    assert report.artifact_dir == "/tmp/astro-assistant/leo_two_station_topocentric"


def test_classifies_optional_backend_scenario() -> None:
    report = classify_local_od_support(
        "Run local orbit determination on examples/scenarios/leo_orekit_high_fidelity.yaml"
    )

    assert report.supported is False
    assert report.code == "optional_backend"
    assert report.scenario_path == "examples/scenarios/leo_orekit_high_fidelity.yaml"
    assert "optional backend" in report.message


def test_classifies_missing_measurement_geometry() -> None:
    report = classify_local_od_support(
        "Run local orbit determination on examples/scenarios/meo_two_body.yaml"
    )

    assert report.supported is False
    assert report.code == "missing_measurements"
    assert report.scenario_path == "examples/scenarios/meo_two_body.yaml"
    assert "at least one measurement" in report.message.lower()


def test_classifies_rank_deficient_geometry() -> None:
    report = classify_local_od_support(
        "Run local orbit determination on examples/scenarios/leo_two_body.yaml"
    )

    assert report.supported is False
    assert report.code == "rank_deficient_geometry"
    assert report.scenario_path == "examples/scenarios/leo_two_body.yaml"
    assert "rank deficient" in report.message


def test_classifies_path_policy_rejection() -> None:
    report = classify_local_od_support("Run local OD on /tmp/custom.yaml")

    assert report.supported is False
    assert report.code == "path_policy"
    assert report.scenario_path is None
    assert "examples/scenarios" in report.message


def test_classifies_unrelated_prompt() -> None:
    report = classify_local_od_support("Tune a launch vehicle")

    assert report.supported is False
    assert report.code == "unsupported_prompt"
    assert "local OD" in report.message
