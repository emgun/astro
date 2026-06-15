import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidMeasurementFileError
from astro_core.io import load_scenario
from astro_core.models import Frame, GroundStation, Scenario
from astro_dynamics.local import propagate_local
from astro_od.io import load_measurements
from astro_od.measurements import generate_synthetic_measurements


def _observable_scenario() -> Scenario:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    station = GroundStation(
        name="measurement-file-y-axis-eci",
        position_eci_km=(0.0, 6378.1363, 0.0),
        frame=Frame.EME2000,
        elevation_mask_deg=0.0,
    )
    return scenario.model_copy(update={"ground_stations": [*scenario.ground_stations, station]})


def test_load_measurements_round_trips_synth_measurement_payload(tmp_path: Path) -> None:
    scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    path = tmp_path / "measurements.json"
    path.write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "measurements": [record.model_dump(mode="json") for record in measurements],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_measurements(path, expected_scenario_id=scenario.scenario_id)

    assert loaded == measurements


def test_load_measurements_rejects_scenario_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "measurements.json"
    path.write_text('{"scenario_id": "wrong", "measurements": []}', encoding="utf-8")

    with pytest.raises(InvalidMeasurementFileError, match="scenario_id"):
        load_measurements(path, expected_scenario_id="leo-two-body")


def test_load_measurements_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "measurements.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidMeasurementFileError, match="Could not parse"):
        load_measurements(path)


def test_load_measurements_rejects_invalid_record_payload(tmp_path: Path) -> None:
    path = tmp_path / "measurements.json"
    path.write_text(
        json.dumps(
            {
                "scenario_id": "leo-two-body",
                "measurements": [
                    {
                        "measurement_type": "range",
                        "epoch": "2026-01-01T00:00:00+00:00",
                        "observer": "equator-eci",
                        "observed_object": "demo-sat",
                        "value": 1.0,
                        "sigma": 0.01,
                        "units": "km/s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="is invalid"):
        load_measurements(path)
