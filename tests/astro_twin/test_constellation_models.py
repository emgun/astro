import pytest
from pydantic import ValidationError

from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
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
    )

    assert scenario.scenario_id == "leo-observers"
    assert len(scenario.members) == 2
    assert scenario.coverage_requirements[0].maximum_revisit_gap_s == 300.0


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
