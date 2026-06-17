import csv
import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidMeasurementFileError
from astro_core.io import load_scenario
from astro_core.models import Frame, GroundStation, MeasurementRecord, MeasurementType, Scenario
from astro_dynamics.local import propagate_local
from astro_od.io import dump_measurements_tdm, load_measurements
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


def test_load_measurements_round_trips_doppler_json_and_csv(tmp_path: Path) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_doppler.yaml"))
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    json_path = tmp_path / "doppler.json"
    csv_path = tmp_path / "doppler.csv"
    json_path.write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "measurements": [record.model_dump(mode="json") for record in measurements],
            }
        ),
        encoding="utf-8",
    )
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
    with csv_path.open("w", encoding="utf-8", newline="") as measurement_file:
        writer = csv.DictWriter(measurement_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in measurements:
            payload = record.model_dump(mode="json")
            writer.writerow(
                {
                    "scenario_id": scenario.scenario_id,
                    "measurement_type": payload["measurement_type"],
                    "epoch": payload["epoch"],
                    "observer": payload["observer"],
                    "observed_object": payload["observed_object"],
                    "value": payload["value"],
                    "sigma": payload["sigma"],
                    "units": payload["units"],
                    "metadata_json": json.dumps(payload["metadata"]),
                }
            )

    json_loaded = load_measurements(json_path, expected_scenario_id=scenario.scenario_id)
    csv_loaded = load_measurements(csv_path, expected_scenario_id=scenario.scenario_id)

    assert {record.measurement_type for record in json_loaded} == {
        MeasurementType.RANGE,
        MeasurementType.DOPPLER,
    }
    assert {
        record.units
        for record in json_loaded
        if record.measurement_type is MeasurementType.DOPPLER
    } == {"Hz"}
    assert csv_loaded == json_loaded


def test_load_measurements_round_trips_three_way_radiometric_json_and_csv(
    tmp_path: Path,
) -> None:
    scenario = _observable_scenario()
    three_way_measurements = scenario.measurements.model_copy(
        update={"types": (MeasurementType.THREE_WAY_RANGE, MeasurementType.THREE_WAY_RANGE_RATE)}
    )
    scenario = scenario.model_copy(update={"measurements": three_way_measurements})
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    json_path = tmp_path / "three_way.json"
    csv_path = tmp_path / "three_way.csv"
    json_path.write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "measurements": [record.model_dump(mode="json") for record in measurements],
            }
        ),
        encoding="utf-8",
    )
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
    with csv_path.open("w", encoding="utf-8", newline="") as measurement_file:
        writer = csv.DictWriter(measurement_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in measurements:
            payload = record.model_dump(mode="json")
            writer.writerow(
                {
                    "scenario_id": scenario.scenario_id,
                    "measurement_type": payload["measurement_type"],
                    "epoch": payload["epoch"],
                    "observer": payload["observer"],
                    "observed_object": payload["observed_object"],
                    "value": payload["value"],
                    "sigma": payload["sigma"],
                    "units": payload["units"],
                    "metadata_json": json.dumps(payload["metadata"]),
                }
            )

    json_loaded = load_measurements(json_path, expected_scenario_id=scenario.scenario_id)
    csv_loaded = load_measurements(csv_path, expected_scenario_id=scenario.scenario_id)

    assert len(json_loaded) == 22
    assert {record.measurement_type for record in json_loaded} == {
        MeasurementType.THREE_WAY_RANGE,
        MeasurementType.THREE_WAY_RANGE_RATE,
    }
    assert {record.metadata["transmitter"] for record in json_loaded} == {"equator-eci"}
    assert csv_loaded == json_loaded


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


def test_load_measurements_reads_tdm_angle_records(tmp_path: Path) -> None:
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
                "ANGLE_TYPE = RADEC",
                "ANGLE_UNITS = deg",
                "ANGLE_SIGMA_DEG = 0.002",
                "META_STOP",
                "DATA_START",
                "ANGLE_1 = 2026-001T00:00:00 12.5",
                "ANGLE_2 = 2026-001T00:00:00 -8.25",
                "DATA_STOP",
                "META_START",
                "SCENARIO_ID = leo-two-body",
                "TIME_SYSTEM = UTC",
                "MODE = SEQUENTIAL",
                "PARTICIPANT_1 = measurement-file-y-axis-eci",
                "PARTICIPANT_2 = demo-sat",
                "PATH = 1,2,1",
                "ANGLE_TYPE = AZEL",
                "ANGLE_UNITS = deg",
                "ANGLE_SIGMA_DEG = 0.003",
                "META_STOP",
                "DATA_START",
                "ANGLE_1 = 2026-001T00:01:00 95.0",
                "ANGLE_2 = 2026-001T00:01:00 22.0",
                "DATA_STOP",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_measurements(path, expected_scenario_id="leo-two-body")

    assert [record.measurement_type for record in loaded] == [
        MeasurementType.RIGHT_ASCENSION,
        MeasurementType.DECLINATION,
        MeasurementType.AZIMUTH,
        MeasurementType.ELEVATION,
    ]
    assert [record.value for record in loaded] == [12.5, -8.25, 95.0, 22.0]
    assert [record.sigma for record in loaded] == [0.002, 0.002, 0.003, 0.003]
    assert [record.units for record in loaded] == ["deg", "deg", "deg", "deg"]
    assert [record.metadata["tdm_keyword"] for record in loaded] == [
        "ANGLE_1",
        "ANGLE_2",
        "ANGLE_1",
        "ANGLE_2",
    ]
    assert [record.metadata["tdm_angle_type"] for record in loaded] == [
        "RADEC",
        "RADEC",
        "AZEL",
        "AZEL",
    ]


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


def test_dump_measurements_tdm_round_trips_angle_measurements(tmp_path: Path) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.RIGHT_ASCENSION,
            epoch=scenario.initial_state.epoch,
            observer="equator-eci",
            observed_object=scenario.spacecraft.name,
            value=12.5,
            sigma=0.001,
            units="deg",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.DECLINATION,
            epoch=scenario.initial_state.epoch,
            observer="equator-eci",
            observed_object=scenario.spacecraft.name,
            value=-8.25,
            sigma=0.001,
            units="deg",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.AZIMUTH,
            epoch=scenario.initial_state.epoch,
            observer="equator-eci",
            observed_object=scenario.spacecraft.name,
            value=95.0,
            sigma=0.002,
            units="deg",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.ELEVATION,
            epoch=scenario.initial_state.epoch,
            observer="equator-eci",
            observed_object=scenario.spacecraft.name,
            value=22.0,
            sigma=0.002,
            units="deg",
        ),
    ]

    exported = dump_measurements_tdm(scenario.scenario_id, records)
    path = tmp_path / "angles.tdm"
    path.write_text(exported, encoding="utf-8")

    assert "ANGLE_TYPE = RADEC" in exported
    assert "ANGLE_TYPE = AZEL" in exported
    assert exported.count("ANGLE_1 =") == 2
    assert exported.count("ANGLE_2 =") == 2
    assert "ANGLE_SIGMA_DEG = 0.001" in exported
    assert "ANGLE_SIGMA_DEG = 0.002" in exported
    loaded = load_measurements(path, expected_scenario_id=scenario.scenario_id)

    assert [(record.measurement_type, record.value, record.sigma) for record in loaded] == [
        (record.measurement_type, record.value, record.sigma) for record in records
    ]


def test_dump_measurements_tdm_round_trips_multi_leg_radiometric_measurements(
    tmp_path: Path,
) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_radiometric_links.yaml"))
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.TWO_WAY_RANGE,
            epoch=scenario.initial_state.epoch,
            observer="uplink-eci",
            observed_object=scenario.spacecraft.name,
            value=1243.7274,
            sigma=0.01,
            units="km",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.TWO_WAY_RANGE_RATE,
            epoch=scenario.initial_state.epoch,
            observer="uplink-eci",
            observed_object=scenario.spacecraft.name,
            value=0.0,
            sigma=1.0e-5,
            units="km/s",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.THREE_WAY_RANGE,
            epoch=scenario.initial_state.epoch,
            observer="downlink-eci",
            observed_object=scenario.spacecraft.name,
            value=10092.0,
            sigma=0.01,
            units="km",
            metadata={"transmitter": "uplink-eci"},
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.THREE_WAY_RANGE_RATE,
            epoch=scenario.initial_state.epoch,
            observer="downlink-eci",
            observed_object=scenario.spacecraft.name,
            value=-5.0,
            sigma=1.0e-5,
            units="km/s",
            metadata={"transmitter": "uplink-eci"},
        ),
    ]

    exported = dump_measurements_tdm(scenario.scenario_id, records)
    path = tmp_path / "radiometric.tdm"
    path.write_text(exported, encoding="utf-8")
    loaded = load_measurements(path, expected_scenario_id=scenario.scenario_id)

    assert "ASTRO_MEASUREMENT_TYPE = two_way" in exported
    assert "ASTRO_MEASUREMENT_TYPE = three_way" in exported
    assert "PATH = 1,2,1" in exported
    assert "PATH = 1,2,3" in exported
    assert [record.measurement_type for record in loaded] == [
        record.measurement_type for record in records
    ]
    assert [record.observer for record in loaded] == [record.observer for record in records]
    assert [record.value for record in loaded] == [record.value for record in records]
    assert [record.metadata.get("transmitter") for record in loaded[2:]] == [
        "uplink-eci",
        "uplink-eci",
    ]


def test_dump_measurements_tdm_round_trips_hz_doppler_with_suite_extension(
    tmp_path: Path,
) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.DOPPLER,
            epoch=scenario.initial_state.epoch,
            observer="equator-eci",
            observed_object=scenario.spacecraft.name,
            value=-12.5,
            sigma=0.05,
            units="Hz",
        )
    ]

    exported = dump_measurements_tdm(scenario.scenario_id, records)
    path = tmp_path / "doppler_hz.tdm"
    path.write_text(exported, encoding="utf-8")
    loaded = load_measurements(path, expected_scenario_id=scenario.scenario_id)

    assert "ASTRO_MEASUREMENT_TYPE = doppler_hz" in exported
    assert "DOPPLER_UNITS = Hz" in exported
    assert "DOPPLER_SIGMA_HZ = 0.05" in exported
    assert "DOPPLER_INSTANTANEOUS =" in exported
    loaded_values = [
        (record.measurement_type, record.value, record.sigma, record.units)
        for record in loaded
    ]
    assert loaded_values == [(MeasurementType.DOPPLER, -12.5, 0.05, "Hz")]
    assert loaded[0].metadata["astro_measurement_type"] == "doppler_hz"


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
