from pathlib import Path
from struct import pack

import pytest

from astro_core.errors import InvalidMeasurementFileError
from astro_core.models import MeasurementType
from astro_od.dsn import load_dsn_binary_tracking_measurements, load_dsn_tracking_measurements


def test_load_dsn_tracking_measurements_maps_normalized_odf_tnf_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dsn_tracking.csv"
    path.write_text(
        "\n".join(
            [
                "scenario_id,tracking_format,observable,epoch,station,spacecraft,value,sigma,units,participant_path,transmitter,media_source",
                "dsn-demo,odf,two_way_range,2026-01-01T00:00:00+00:00,DSS-14,demo-sat,12345.6,0.01,km,\"DSS-14,demo-sat,DSS-14\",,calibrated-media",
                "dsn-demo,tnf,three_way_range_rate,2026-01-01T00:01:00+00:00,DSS-43,demo-sat,-0.012,0.00001,km/s,\"DSS-14,demo-sat,DSS-43\",DSS-14,calibrated-media",
            ]
        ),
        encoding="utf-8",
    )

    product = load_dsn_tracking_measurements(path)

    assert product.scenario_id == "dsn-demo"
    assert len(product.measurements) == 2
    assert product.metadata["source_format"] == "normalized_dsn_tracking_csv"
    assert product.metadata["tracking_formats"] == ["odf", "tnf"]
    assert product.measurements[0].measurement_type is MeasurementType.TWO_WAY_RANGE
    assert product.measurements[0].observer == "DSS-14"
    assert product.measurements[0].metadata["dsn_tracking_format"] == "odf"
    assert product.measurements[0].metadata["media_corrections_source"] == "calibrated-media"
    assert product.measurements[1].measurement_type is MeasurementType.THREE_WAY_RANGE_RATE
    assert product.measurements[1].metadata["transmitter"] == "DSS-14"


def test_load_dsn_tracking_measurements_rejects_mixed_scenario_ids(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        "\n".join(
            [
                "scenario_id,tracking_format,observable,epoch,station,spacecraft,value,sigma,units",
                "first,odf,two_way_range,2026-01-01T00:00:00+00:00,DSS-14,demo-sat,1.0,0.1,km",
                "second,odf,two_way_range,2026-01-01T00:00:00+00:00,DSS-14,demo-sat,1.0,0.1,km",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidMeasurementFileError, match="single scenario_id"):
        load_dsn_tracking_measurements(path)


def test_load_dsn_binary_tracking_measurements_maps_fixed_records(tmp_path: Path) -> None:
    path = tmp_path / "dsn_tracking.bin"
    path.write_bytes(
        b"ASTRODSN1"
        + pack("<I", 2)
        + _binary_tracking_record(
            tracking_format=1,
            observable=3,
            epoch_unix_s=1767225600,
            value=12345.6,
            sigma=0.01,
            units=1,
            scenario_id="dsn-binary-demo",
            station="DSS-14",
            spacecraft="demo-sat",
            participant_path="DSS-14,demo-sat,DSS-14",
            transmitter="",
            media_source="binary-media",
        )
        + _binary_tracking_record(
            tracking_format=2,
            observable=6,
            epoch_unix_s=1767225660,
            value=-0.012,
            sigma=0.00001,
            units=2,
            scenario_id="dsn-binary-demo",
            station="DSS-43",
            spacecraft="demo-sat",
            participant_path="DSS-14,demo-sat,DSS-43",
            transmitter="DSS-14",
            media_source="binary-media",
        )
    )

    product = load_dsn_binary_tracking_measurements(path)

    assert product.scenario_id == "dsn-binary-demo"
    assert product.metadata["source_format"] == "astro_dsn_binary_tracking"
    assert product.metadata["tracking_formats"] == ["odf", "tnf"]
    assert product.measurements[0].measurement_type is MeasurementType.TWO_WAY_RANGE
    assert product.measurements[0].units == "km"
    assert product.measurements[0].metadata["binary_record_index"] == 0
    assert product.measurements[1].measurement_type is MeasurementType.THREE_WAY_RANGE_RATE
    assert product.measurements[1].units == "km/s"
    assert product.measurements[1].metadata["transmitter"] == "DSS-14"


def _binary_tracking_record(
    *,
    tracking_format: int,
    observable: int,
    epoch_unix_s: int,
    value: float,
    sigma: float,
    units: int,
    scenario_id: str,
    station: str,
    spacecraft: str,
    participant_path: str,
    transmitter: str,
    media_source: str,
) -> bytes:
    payload = pack("<BBqddB", tracking_format, observable, epoch_unix_s, value, sigma, units)
    for field in (
        scenario_id,
        station,
        spacecraft,
        participant_path,
        transmitter,
        media_source,
    ):
        encoded = field.encode("utf-8")
        payload += pack("<H", len(encoded)) + encoded
    return payload
