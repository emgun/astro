import csv
import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidMeasurementFileError
from astro_core.io import load_scenario
from astro_core.models import Frame, GroundStation, MeasurementRecord, MeasurementType, Scenario
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


def test_load_measurements_reads_tdm_range_and_doppler_records(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tdm"
    path.write_text(
        "\n".join(
            [
                "CCSDS_TDM_VERS = 2.0",
                "CREATION_DATE = 2026-01-01T00:00:00Z",
                "ORIGINATOR = ASTRO_SUITE_TEST",
                "META_START",
                "SCENARIO_ID = leo-two-body",
                "TIME_SYSTEM = UTC",
                "MODE = SEQUENTIAL",
                "PARTICIPANT_1 = measurement-file-y-axis-eci",
                "PARTICIPANT_2 = demo-sat",
                "PATH = 1,2,1",
                "RANGE_UNITS = km",
                "META_STOP",
                "DATA_START",
                "RANGE = 2026-001T00:00:00 621.8637",
                "DOPPLER_INSTANTANEOUS = 2026-01-01T00:00:00 -6.507594",
                "DATA_STOP",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_measurements(path, expected_scenario_id="leo-two-body")

    assert [record.measurement_type for record in loaded] == [
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    ]
    assert [record.observer for record in loaded] == [
        "measurement-file-y-axis-eci",
        "measurement-file-y-axis-eci",
    ]
    assert [record.observed_object for record in loaded] == ["demo-sat", "demo-sat"]
    assert loaded[0].epoch.isoformat() == "2026-01-01T00:00:00+00:00"
    assert loaded[0].value == 621.8637
    assert loaded[0].sigma == 0.01
    assert loaded[0].units == "km"
    assert loaded[0].metadata["tdm_keyword"] == "RANGE"
    assert loaded[1].value == -6.507594
    assert loaded[1].sigma == 1.0e-5
    assert loaded[1].units == "km/s"
    assert loaded[1].metadata["tdm_keyword"] == "DOPPLER_INSTANTANEOUS"


def test_load_measurements_rejects_tdm_scenario_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tdm"
    path.write_text(
        "\n".join(
            [
                "CCSDS_TDM_VERS = 2.0",
                "META_START",
                "SCENARIO_ID = wrong",
                "TIME_SYSTEM = UTC",
                "PARTICIPANT_1 = equator-eci",
                "PARTICIPANT_2 = demo-sat",
                "PATH = 1,2,1",
                "META_STOP",
                "DATA_START",
                "RANGE = 2026-01-01T00:00:00 7000.0",
                "DATA_STOP",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="scenario_id"):
        load_measurements(path, expected_scenario_id="leo-two-body")


def test_load_measurements_rejects_tdm_non_km_range_units(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tdm"
    path.write_text(
        "\n".join(
            [
                "CCSDS_TDM_VERS = 2.0",
                "META_START",
                "TIME_SYSTEM = UTC",
                "PARTICIPANT_1 = equator-eci",
                "PARTICIPANT_2 = demo-sat",
                "PATH = 1,2,1",
                "RANGE_UNITS = RU",
                "META_STOP",
                "DATA_START",
                "RANGE = 2026-01-01T00:00:00 7000.0",
                "DATA_STOP",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="RANGE_UNITS"):
        load_measurements(path)


def test_load_measurements_rejects_tdm_missing_participants(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tdm"
    path.write_text(
        "\n".join(
            [
                "CCSDS_TDM_VERS = 2.0",
                "META_START",
                "TIME_SYSTEM = UTC",
                "META_STOP",
                "DATA_START",
                "RANGE = 2026-01-01T00:00:00 7000.0",
                "DATA_STOP",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="PARTICIPANT"):
        load_measurements(path)


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
