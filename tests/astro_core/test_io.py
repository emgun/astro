from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario, load_trajectory
from astro_core.models import ForceModelName, MeasurementType
from astro_dynamics.local import propagate_local


def test_load_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    assert scenario.scenario_id == "leo-two-body"
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.propagation.sample_count == 11


def test_load_two_station_od_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_station_od.yaml"))

    assert scenario.scenario_id == "leo-two-station-od"
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.propagation.sample_count == 11
    assert len(scenario.ground_stations) == 2


def test_load_geodetic_eop_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_eop_topocentric.yaml"))

    assert scenario.scenario_id == "leo-geodetic-eop-topocentric"
    assert scenario.earth_orientation.source == "example-fixed-eop"
    assert scenario.earth_orientation.ut1_minus_utc_s == 0.12


def test_load_geodetic_eop_table_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_eop_table_topocentric.yaml"))

    assert scenario.scenario_id == "leo-geodetic-eop-table-topocentric"
    assert scenario.earth_orientation.source == "example-eop-table"
    assert len(scenario.earth_orientation.samples) == 2


def test_load_doppler_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_doppler.yaml"))

    assert scenario.scenario_id == "leo-doppler"
    assert scenario.measurements.types == (MeasurementType.RANGE, MeasurementType.DOPPLER)
    assert scenario.measurements.doppler_transmit_frequency_hz == 8.4e9
    assert scenario.measurements.noise.doppler_sigma_hz == 0.05


def test_load_velocity_aligned_burn_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_velocity_aligned_burn.yaml"))

    assert scenario.scenario_id == "leo-velocity-aligned-burn"
    assert scenario.maneuvers[0].thrust_direction_mode == "velocity_aligned"


def test_load_j2_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_j2.yaml"))

    trajectory = propagate_local(scenario)

    assert scenario.scenario_id == "leo-j2"
    assert scenario.force_model.gravity is ForceModelName.J2
    assert trajectory.backend == "local"
    assert len(trajectory.samples) == 11


def test_load_orekit_high_fidelity_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_high_fidelity.yaml"))

    assert scenario.scenario_id == "leo-orekit-high-fidelity"
    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.enabled_high_fidelity_flags() == ()


def test_load_orekit_drag_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_drag.yaml"))

    assert scenario.scenario_id == "leo-orekit-drag"
    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.enabled_high_fidelity_flags() == ("atmospheric_drag",)


def test_load_orekit_srp_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_srp.yaml"))

    assert scenario.scenario_id == "leo-orekit-srp"
    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.enabled_high_fidelity_flags() == ("solar_radiation_pressure",)


def test_load_orekit_third_body_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_third_body.yaml"))

    assert scenario.scenario_id == "leo-orekit-third-body"
    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.enabled_high_fidelity_flags() == ("third_body_gravity",)


@pytest.mark.parametrize(
    ("scenario_path", "scenario_id", "minimum_radius_km"),
    [
        ("examples/scenarios/meo_two_body.yaml", "meo-two-body", 20000.0),
        ("examples/scenarios/geo_two_body.yaml", "geo-two-body", 40000.0),
    ],
)
def test_load_medium_and_geosynchronous_examples(
    scenario_path: str,
    scenario_id: str,
    minimum_radius_km: float,
) -> None:
    scenario = load_scenario(Path(scenario_path))

    trajectory = propagate_local(scenario)

    assert scenario.scenario_id == scenario_id
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.propagation.sample_count == 7
    minimum_sample_radius_km = min(
        sample.state.position_array().dot(sample.state.position_array()) ** 0.5
        for sample in trajectory.samples
    )
    assert minimum_sample_radius_km > minimum_radius_km


def test_load_trajectory_reads_json_product(tmp_path: Path) -> None:
    trajectory = propagate_local(load_scenario(Path("examples/scenarios/leo_two_body.yaml")))
    path = tmp_path / "trajectory.json"
    path.write_text(trajectory.model_dump_json(), encoding="utf-8")

    loaded = load_trajectory(path)

    assert loaded.scenario_id == "leo-two-body"
    assert loaded.backend == "local"
    assert len(loaded.samples) == 11


def test_load_scenario_reports_yaml_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_id: [broken", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="Could not parse scenario file"):
        load_scenario(path)


def test_load_scenario_reports_utf8_decode_errors_as_read_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad-encoding.yaml"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(InvalidScenarioError, match="Could not read scenario file"):
        load_scenario(path)


def test_load_scenario_requires_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="must contain a mapping"):
        load_scenario(path)


def test_load_scenario_wraps_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-scenario.yaml"
    path.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="is invalid"):
        load_scenario(path)
