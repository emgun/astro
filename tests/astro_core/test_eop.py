from datetime import UTC, datetime
from pathlib import Path

import pytest

from astro_core.eop import load_iers_finals_eop, parse_iers_finals_eop


def test_parse_iers_finals_eop_extracts_polar_motion_and_ut1_samples() -> None:
    payload = Path("examples/eop/finals2000A_excerpt.txt").read_text(encoding="utf-8")

    earth_orientation = parse_iers_finals_eop(payload, source="unit-test-finals2000A")

    assert earth_orientation.source == "unit-test-finals2000A"
    assert len(earth_orientation.samples) == 2
    first = earth_orientation.samples[0]
    assert first.epoch == datetime(2026, 1, 1, tzinfo=UTC)
    assert first.polar_motion_x_arcsec == pytest.approx(0.144949)
    assert first.polar_motion_y_arcsec == pytest.approx(0.316111)
    assert first.ut1_minus_utc_s == pytest.approx(0.0730000)


def test_load_iers_finals_eop_reads_fixture_file() -> None:
    earth_orientation = load_iers_finals_eop(
        Path("examples/eop/finals2000A_excerpt.txt"),
        source="file-finals2000A",
    )

    assert earth_orientation.source == "file-finals2000A"
    assert earth_orientation.at_epoch(datetime(2026, 1, 1, 12, tzinfo=UTC)).ut1_minus_utc_s == (
        pytest.approx((0.0730000 + 0.0728123) / 2.0)
    )


def test_parse_iers_finals_eop_rejects_payload_without_samples() -> None:
    with pytest.raises(ValueError, match="no usable"):
        parse_iers_finals_eop("26 1 3 61043.00\n")
