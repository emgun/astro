from pathlib import Path

from astro_core.models import MeasurementRecord
from astro_od.io import load_measurements

EXAMPLE_SCENARIO_ID = "leo-two-station-od"
EXAMPLE_MEASUREMENT_DIR = Path("examples/measurements")


def _measurement_signature(record: MeasurementRecord) -> tuple[object, ...]:
    return (
        record.measurement_type,
        record.epoch,
        record.observer,
        record.observed_object,
        record.value,
        record.sigma,
        record.units,
    )


def test_reference_measurement_fixtures_round_trip() -> None:
    json_records = load_measurements(
        EXAMPLE_MEASUREMENT_DIR / "leo_two_station_od_measurements.json",
        expected_scenario_id=EXAMPLE_SCENARIO_ID,
    )
    csv_records = load_measurements(
        EXAMPLE_MEASUREMENT_DIR / "leo_two_station_od_measurements.csv",
        expected_scenario_id=EXAMPLE_SCENARIO_ID,
    )
    tdm_records = load_measurements(
        EXAMPLE_MEASUREMENT_DIR / "leo_two_station_od_measurements.tdm",
        expected_scenario_id=EXAMPLE_SCENARIO_ID,
    )

    assert len(json_records) == 44
    assert [_measurement_signature(record) for record in csv_records] == [
        _measurement_signature(record) for record in json_records
    ]
    assert sorted(_measurement_signature(record) for record in tdm_records) == sorted(
        _measurement_signature(record) for record in json_records
    )
