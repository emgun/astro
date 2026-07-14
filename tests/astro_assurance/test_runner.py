from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import astro_assurance.runner as assurance_runner
from astro_assurance.errors import MissionAssuranceError
from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.models import AssuranceStatus
from astro_assurance.runner import run_post_launch_assurance
from astro_od.measurements import elevation_deg
from astro_twin.io import load_twin_scenario


def test_reference_post_launch_assurance_closes_the_truth_replay() -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )

    result = run_post_launch_assurance(scenario)

    assert result.passed
    assert result.continuity_report.all_passed
    assert {check.name for check in result.continuity_report.checks} >= {
        "launch_tracking_mass",
        "launch_twin_mass",
        "correction_twin_mass",
    }
    assert result.estimate.converged
    assert all(
        measurement.epoch <= result.correction_maneuver.epoch
        for measurement in result.measurements
    )
    assert len(result.measurements) == result.metadata["measurement_count"]
    assert result.metadata["generated_measurement_count"] == 1452
    assert result.metadata["visible_measurement_count"] == (
        result.metadata["measurement_count"]
        + result.metadata["rejected_after_decision_measurement_count"]
    )
    assert result.metadata["generated_measurement_count"] == (
        result.metadata["visible_measurement_count"]
        + result.metadata["rejected_below_mask_measurement_count"]
    )
    assert result.correction_maneuver.metadata["disposition"] == "candidate_for_manual_review"
    assert result.corrected_digital_twin.metadata["orbit_trajectory_source"] == (
        "trajectory_override"
    )
    assert result.metadata["tracking_source"] == "synthetic_simulation_truth"
    assert result.metadata["truth_position_error_reduction_fraction"] > 0.8
    assert result.metadata["truth_recovery_position_error_km"] < 1.0
    assert result.corrected_digital_twin.mass_budget.propellant_mass_kg == pytest.approx(
        50.0 - result.metadata["candidate_propellant_used_kg"]
    )
    assert len(result.manifest.entries) == 16
    assert {reference.role for reference in result.manifest.inputs} == {
        "assurance_scenario",
        "launch_scenario",
        "tracking_scenario",
        "twin_scenario",
    }
    assert all(len(entry.source_digest) == 64 for entry in result.manifest.entries)
    assert not any(
        margin.status.value == "fail"
        for margin in result.corrected_digital_twin.margin_report.margins
    )
    states = {sample.epoch: sample.state for sample in result.truth_trajectory.samples}
    stations = {station.name: station for station in result.truth_scenario.ground_stations}
    for measurement in result.measurements:
        station = stations[measurement.observer]
        station_position = station.position_array(
            measurement.epoch,
            result.truth_scenario.earth_orientation,
        )
        assert elevation_deg(states[measurement.epoch].position_km, station_position) >= float(
            station.elevation_mask_deg
        )
    assert any("not a flight command" in warning for warning in result.warnings)
    assert any("do not prove RF contact" in warning for warning in result.warnings)


def test_reference_assurance_does_not_apply_unrequested_force_role_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )

    def reject_unrequested_override(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("force-role override should not run")

    monkeypatch.setattr(
        assurance_runner,
        "_scenario_with_force_model",
        reject_unrequested_override,
    )

    result = run_post_launch_assurance(scenario)

    assert result.passed


def test_assurance_completes_with_failed_status_when_requirement_is_missed() -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    strict_requirements = scenario.requirements.model_copy(
        update={"maximum_truth_recovery_position_error_km": 0.01}
    )

    result = run_post_launch_assurance(
        scenario.model_copy(update={"requirements": strict_requirements})
    )

    margin = next(
        item
        for item in result.margin_report.margins
        if item.name == "truth_recovery_position_error"
    )
    assert not result.passed
    assert result.continuity_report.all_passed
    assert margin.status is AssuranceStatus.FAIL
    assert result.margin_report.overall_status is AssuranceStatus.FAIL


def test_assurance_exposes_launch_to_twin_mass_discontinuity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    twin = load_twin_scenario(scenario.twin_scenario)
    spacecraft = twin.spacecraft.model_copy(
        update={"dry_mass_kg": float(twin.spacecraft.dry_mass_kg) + 1.0}
    )
    mismatched_twin = twin.model_copy(update={"spacecraft": spacecraft})
    monkeypatch.setattr(assurance_runner, "load_twin_scenario", lambda _: mismatched_twin)

    result = run_post_launch_assurance(scenario)

    check = next(
        item for item in result.continuity_report.checks if item.name == "launch_twin_mass"
    )
    assert not result.passed
    assert not check.passed
    assert check.error == pytest.approx(1.0)


def test_assurance_fails_when_corrected_twin_has_a_failed_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    twin = load_twin_scenario(scenario.twin_scenario)
    thermal_node = twin.thermal_nodes[0].model_copy(update={"maximum_temperature_k": 294.0})
    failing_twin = twin.model_copy(update={"thermal_nodes": (thermal_node,)})
    monkeypatch.setattr(assurance_runner, "load_twin_scenario", lambda _: failing_twin)

    result = run_post_launch_assurance(scenario)

    top_level = next(
        item
        for item in result.margin_report.margins
        if item.name == "digital_twin_failed_margin_count"
    )
    assert not result.passed
    assert top_level.status is AssuranceStatus.FAIL
    assert top_level.value >= 1.0
    assert any(
        margin.status.value == "fail"
        for margin in result.corrected_digital_twin.margin_report.margins
    )


def test_assurance_rejects_schedule_that_does_not_align_with_tracking_step() -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    correction = scenario.correction.model_copy(update={"correction_elapsed_s": 901.0})

    with pytest.raises(
        MissionAssuranceError, match="must align with the tracking propagation step"
    ):
        run_post_launch_assurance(scenario.model_copy(update={"correction": correction}))


def test_assurance_rejects_referenced_input_drift_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    tracking_source = Path(source.tracking_scenario)
    tracking_path = tmp_path / "tracking.yaml"
    tracking_path.write_bytes(tracking_source.read_bytes())
    scenario_path = tmp_path / "assurance.yaml"
    scenario_payload = source.model_dump(mode="json")
    scenario_payload["tracking_scenario"] = str(tracking_path)
    scenario_path.write_text(
        yaml.safe_dump(scenario_payload, sort_keys=False),
        encoding="utf-8",
    )
    scenario = load_post_launch_assurance_scenario(scenario_path)
    original_load = assurance_runner.load_scenario

    def load_then_mutate(path: str) -> object:
        loaded = original_load(path)
        tracking_path.write_text(
            tracking_path.read_text(encoding="utf-8") + "# drift\n",
            encoding="utf-8",
        )
        return loaded

    monkeypatch.setattr(assurance_runner, "load_scenario", load_then_mutate)

    with pytest.raises(MissionAssuranceError, match="tracking_scenario input changed"):
        run_post_launch_assurance(scenario)
