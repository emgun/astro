from datetime import datetime, timezone

from astro_backends.orekit.conversion import km_s_to_m_s, km_to_m, m_s_to_km_s, m_to_km


def test_orekit_unit_conversions_are_reversible() -> None:
    assert km_to_m(7.5) == 7500.0
    assert km_s_to_m_s(7.5) == 7500.0
    assert m_to_km(7500.0) == 7.5
    assert m_s_to_km_s(7500.0) == 7.5


def test_orekit_epoch_requires_utc() -> None:
    epoch = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert epoch.isoformat() == "2026-01-01T00:00:00+00:00"
