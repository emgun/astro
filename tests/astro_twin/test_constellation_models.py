import pytest
from pydantic import ValidationError

from astro_twin.constellation_models import (
    ConstellationCoverageMapConfig,
    ConstellationCoverageRequirement,
    ConstellationCoverageSensorConfig,
    ConstellationCoverageTargetConfig,
    ConstellationMemberConfig,
    ConstellationTwinScenario,
)


def test_constellation_twin_scenario_accepts_members_and_requirements() -> None:
    scenario = ConstellationTwinScenario(
        scenario_id="leo-observers",
        members=(
            ConstellationMemberConfig(
                name="plane-a",
                twin_scenario="examples/twin/leo_observer_plane_a.yaml",
            ),
            ConstellationMemberConfig(
                name="plane-b",
                twin_scenario="examples/twin/leo_observer_plane_b.yaml",
            ),
        ),
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=300.0,
            ),
        ),
        coverage_maps=(
            ConstellationCoverageMapConfig(
                name="equatorial-targets",
                sensor=ConstellationCoverageSensorConfig(
                    name="nadir-imager",
                    field_of_view_half_angle_deg=25.0,
                ),
                targets=(
                    ConstellationCoverageTargetConfig(
                        name="prime-meridian",
                        latitude_deg=0.0,
                        longitude_deg=0.0,
                    ),
                ),
                minimum_target_coverage_fraction=0.2,
                maximum_target_revisit_gap_s=600.0,
            ),
        ),
    )

    assert scenario.scenario_id == "leo-observers"
    assert len(scenario.members) == 2
    assert scenario.coverage_requirements[0].maximum_revisit_gap_s == 300.0
    assert scenario.coverage_maps[0].sensor.name == "nadir-imager"
    assert scenario.coverage_maps[0].targets[0].name == "prime-meridian"


def test_constellation_twin_scenario_rejects_duplicate_member_names() -> None:
    with pytest.raises(ValidationError, match="member names must be unique"):
        ConstellationTwinScenario(
            scenario_id="leo-observers",
            members=(
                ConstellationMemberConfig(name="plane-a", twin_scenario="a.yaml"),
                ConstellationMemberConfig(name="plane-a", twin_scenario="b.yaml"),
            ),
        )


def test_constellation_twin_scenario_rejects_duplicate_requirements() -> None:
    with pytest.raises(
        ValidationError,
        match="coverage requirement ground_site values must be unique",
    ):
        ConstellationTwinScenario(
            scenario_id="leo-observers",
            members=(
                ConstellationMemberConfig(name="plane-a", twin_scenario="a.yaml"),
            ),
            coverage_requirements=(
                ConstellationCoverageRequirement(ground_site="equator-eci"),
                ConstellationCoverageRequirement(ground_site="equator-eci"),
            ),
        )


def test_constellation_twin_scenario_rejects_duplicate_coverage_maps() -> None:
    with pytest.raises(
        ValidationError,
        match="coverage map names must be unique",
    ):
        ConstellationTwinScenario(
            scenario_id="leo-observers",
            members=(
                ConstellationMemberConfig(name="plane-a", twin_scenario="a.yaml"),
            ),
            coverage_maps=(
                ConstellationCoverageMapConfig(
                    name="equatorial-targets",
                    sensor=ConstellationCoverageSensorConfig(
                        name="wide-imager",
                        field_of_view_half_angle_deg=25.0,
                    ),
                    targets=(
                        ConstellationCoverageTargetConfig(
                            name="prime-meridian",
                            latitude_deg=0.0,
                            longitude_deg=0.0,
                        ),
                    ),
                ),
                ConstellationCoverageMapConfig(
                    name="equatorial-targets",
                    sensor=ConstellationCoverageSensorConfig(
                        name="narrow-imager",
                        field_of_view_half_angle_deg=10.0,
                    ),
                    targets=(
                        ConstellationCoverageTargetConfig(
                            name="east-equator",
                            latitude_deg=0.0,
                            longitude_deg=10.0,
                        ),
                    ),
                ),
            ),
        )


def test_constellation_coverage_map_rejects_duplicate_targets() -> None:
    with pytest.raises(
        ValidationError,
        match="coverage map target names must be unique",
    ):
        ConstellationCoverageMapConfig(
            name="equatorial-targets",
            sensor=ConstellationCoverageSensorConfig(
                name="nadir-imager",
                field_of_view_half_angle_deg=25.0,
            ),
            targets=(
                ConstellationCoverageTargetConfig(
                    name="prime-meridian",
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                ),
                ConstellationCoverageTargetConfig(
                    name="prime-meridian",
                    latitude_deg=5.0,
                    longitude_deg=0.0,
                ),
            ),
        )
