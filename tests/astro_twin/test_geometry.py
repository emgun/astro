from datetime import UTC, datetime

from astro_core.models import (
    CartesianState,
    ForceModelConfig,
    ForceModelName,
    Trajectory,
    TrajectorySample,
)
from astro_twin.geometry import build_geometry_timeline


def test_build_geometry_timeline_marks_sunlit_and_eclipse_samples() -> None:
    trajectory = Trajectory(
        scenario_id="test",
        backend="local",
        samples=(
            TrajectorySample(
                epoch=datetime(2026, 1, 1, tzinfo=UTC),
                state=CartesianState(
                    position_km=(7000.0, 0.0, 0.0),
                    velocity_km_s=(0.0, 7.5, 0.0),
                ),
            ),
            TrajectorySample(
                epoch=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
                state=CartesianState(
                    position_km=(-7000.0, 0.0, 0.0),
                    velocity_km_s=(0.0, -7.5, 0.0),
                ),
            ),
        ),
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        metadata={},
    )

    timeline = build_geometry_timeline(trajectory)

    assert len(timeline) == 2
    assert timeline[1].elapsed_s == 600.0
    assert timeline[0].sunlit is True
    assert timeline[1].sunlit is False
    assert timeline[0].altitude_km > 600.0
