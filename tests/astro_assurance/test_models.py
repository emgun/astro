from __future__ import annotations

import pytest

from astro_assurance.models import InsertionDispersion


def test_insertion_dispersion_requires_a_nonzero_offset() -> None:
    with pytest.raises(ValueError, match="must be nonzero"):
        InsertionDispersion(
            position_delta_km=(0.0, 0.0, 0.0),
            velocity_delta_km_s=(0.0, 0.0, 0.0),
        )
