import json
from pathlib import Path

import pytest
import yaml

from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_io import (
    format_constellation_summary,
    load_constellation_twin_result,
    load_constellation_twin_scenario,
    write_constellation_twin_result,
)
from astro_twin.constellation_models import ConstellationTwinResult
from astro_twin.models import DesignMargin, DesignMarginReport, TwinMarginStatus


def test_load_constellation_twin_scenario_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "constellation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenario_id": "leo-observers",
                "members": [
                    {"name": "plane-a", "twin_scenario": "a.yaml"},
                    {"name": "plane-b", "twin_scenario": "b.yaml"},
                ],
                "coverage_requirements": [
                    {
                        "ground_site": "equator-eci",
                        "minimum_coverage_fraction": 0.25,
                        "maximum_revisit_gap_s": 300.0,
                    }
                ],
                "coverage_maps": [
                    {
                        "name": "equatorial-targets",
                        "sensor": {
                            "name": "nadir-imager",
                            "field_of_view_half_angle_deg": 25.0,
                        },
                        "targets": [
                            {
                                "name": "prime-meridian",
                                "latitude_deg": 0.0,
                                "longitude_deg": 0.0,
                            }
                        ],
                        "minimum_target_coverage_fraction": 0.25,
                        "maximum_target_revisit_gap_s": 600.0,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    scenario = load_constellation_twin_scenario(path)

    assert scenario.scenario_id == "leo-observers"
    assert scenario.members[1].name == "plane-b"
    assert scenario.coverage_requirements[0].minimum_coverage_fraction == 0.25
    assert scenario.coverage_maps[0].targets[0].name == "prime-meridian"


def test_load_constellation_twin_scenario_rejects_non_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="must contain a mapping"):
        load_constellation_twin_scenario(path)


def test_write_load_and_format_constellation_result(tmp_path: Path) -> None:
    result = ConstellationTwinResult(
        scenario_id="leo-observers",
        members=(),
        access_summaries=(),
        link_summaries=(),
        member_link_summaries=(),
        coverage_map_summaries=(),
        fleet_margin_report=DesignMarginReport(
            margins=(
                DesignMargin(
                    name="fleet_link_margin_db_equator-eci",
                    value=3.0,
                    threshold=0.0,
                    margin=3.0,
                    unit="dB",
                    status=TwinMarginStatus.WARN,
                ),
            ),
            limiting_margin=DesignMargin(
                name="fleet_link_margin_db_equator-eci",
                value=3.0,
                threshold=0.0,
                margin=3.0,
                unit="dB",
                status=TwinMarginStatus.WARN,
            ),
        ),
        metadata={"analysis_window_s": {"start_s": 0.0, "end_s": 600.0}},
        warnings=["screening only"],
    )
    output = tmp_path / "result.json"

    write_constellation_twin_result(output, result)
    loaded = load_constellation_twin_result(output)
    summary = format_constellation_summary(loaded)

    assert json.loads(output.read_text(encoding="utf-8"))["workflow"] == (
        "constellation_digital_twin_v2"
    )
    assert loaded.scenario_id == "leo-observers"
    assert "Constellation twin: leo-observers" in summary
    assert "Coverage maps: 0" in summary
    assert "Limiting fleet margin:" in summary
    assert "screening only" in summary


def test_write_constellation_twin_result_wraps_write_errors(
    tmp_path: Path,
) -> None:
    result = ConstellationTwinResult(
        scenario_id="leo-observers",
        members=(),
        access_summaries=(),
        link_summaries=(),
        member_link_summaries=(),
        coverage_map_summaries=(),
        fleet_margin_report=DesignMarginReport(
            margins=(
                DesignMargin(
                    name="fleet_link_margin_db_equator-eci",
                    value=3.0,
                    threshold=0.0,
                    margin=3.0,
                    unit="dB",
                    status=TwinMarginStatus.WARN,
                ),
            ),
            limiting_margin=DesignMargin(
                name="fleet_link_margin_db_equator-eci",
                value=3.0,
                threshold=0.0,
                margin=3.0,
                unit="dB",
                status=TwinMarginStatus.WARN,
            ),
        ),
    )

    with pytest.raises(
        InvalidScenarioError,
        match="Could not write constellation twin result",
    ):
        write_constellation_twin_result(tmp_path, result)


def test_load_constellation_twin_result_rejects_non_object_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="must contain a JSON object"):
        load_constellation_twin_result(path)
