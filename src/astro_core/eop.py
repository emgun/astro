from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from astro_core.models import EarthOrientationConfig, EarthOrientationSample

DEFAULT_IERS_FINALS_SOURCE = "iers-finals"


def _four_digit_year(two_digit_year: int) -> int:
    return 1900 + two_digit_year if two_digit_year >= 50 else 2000 + two_digit_year


def _float_field(value: str, *, line_number: int, label: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"IERS finals line {line_number} has invalid {label} value {value!r}"
        ) from exc


def parse_iers_finals_eop(
    payload: str,
    *,
    source: str = DEFAULT_IERS_FINALS_SOURCE,
) -> EarthOrientationConfig:
    """Parse IERS finals/finals2000A rows into the suite EOP table subset."""
    samples: list[EarthOrientationSample] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"IERS finals line {line_number} is too short")
        if len(fields) < 11:
            continue

        try:
            year = _four_digit_year(int(fields[0]))
            month = int(fields[1])
            day = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"IERS finals line {line_number} has invalid date fields") from exc

        samples.append(
            EarthOrientationSample(
                epoch=datetime(year, month, day, tzinfo=UTC),
                polar_motion_x_arcsec=_float_field(
                    fields[5],
                    line_number=line_number,
                    label="polar motion x",
                ),
                polar_motion_y_arcsec=_float_field(
                    fields[7],
                    line_number=line_number,
                    label="polar motion y",
                ),
                ut1_minus_utc_s=_float_field(
                    fields[10],
                    line_number=line_number,
                    label="UT1-UTC",
                ),
            )
        )

    if not samples:
        raise ValueError("IERS finals payload contains no usable Earth-orientation samples")
    return EarthOrientationConfig(source=source, samples=tuple(samples))


def load_iers_finals_eop(
    path: Path | str,
    *,
    source: str = DEFAULT_IERS_FINALS_SOURCE,
) -> EarthOrientationConfig:
    eop_path = Path(path)
    try:
        payload = eop_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read IERS finals file {eop_path}: {exc}") from exc
    return parse_iers_finals_eop(payload, source=source)
