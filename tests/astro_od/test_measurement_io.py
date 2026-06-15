import csv
import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidMeasurementFileError
from astro_core.io import load_scenario
from astro_core.models import Frame, GroundStation, MeasurementRecord, Scenario
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


def _write_csv_measurements(path: Path, scenario: Scenario) -> list[MeasurementRecord]:
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    fieldnames = [
        "scenario_id",
        "measurement_type",
        "epoch",
        "observer",
        "observed_object",
        "value",
        "sigma",
        "units",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as measurement_file:
        writer = csv.DictWriter(measurement_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in measurements:
            payload = record.model_dump(mode="json")
            writer.writerow(
                {"scenario_id": scenario.scenario_id}
                | {
                    fieldname: (
                        json.dumps(payload["metadata"])
                        if fieldname == "metadata_json"
                        else payload[fieldname]
                    )
                    for fieldname in fieldnames
                    if fieldname != "scenario_id"
                }
            )
    return measurements


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


def test_load_measurements_reads_csv_records(tmp_path: Path) -> None:
    scenario = _observable_scenario()
    path = tmp_path / "measurements.csv"
    measurements = _write_csv_measurements(path, scenario)

    loaded = load_measurements(path, expected_scenario_id=scenario.scenario_id)

    assert loaded == measurements


def test_load_measurements_rejects_csv_scenario_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    path.write_text(
        "\n".join(
            [
                "scenario_id,measurement_type,epoch,observer,observed_object,value,sigma,units",
                "wrong,range,2026-01-01T00:00:00+00:00,equator-eci,demo-sat,7000.0,0.01,km",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="scenario_id"):
        load_measurements(path, expected_scenario_id="leo-two-body")


def test_load_measurements_rejects_csv_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    path.write_text(
        "\n".join(
            [
                "scenario_id,measurement_type,epoch,observer,observed_object,value,units",
                "leo-two-body,range,2026-01-01T00:00:00+00:00,equator-eci,demo-sat,7000.0,km",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="missing required columns"):
        load_measurements(path)


def test_load_measurements_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "measurements.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(InvalidMeasurementFileError, match="Unsupported measurement format"):
        load_measurements(path)


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
