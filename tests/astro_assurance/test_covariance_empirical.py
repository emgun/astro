from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from astro_assurance.covariance_empirical import run_empirical_covariance_campaign
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import Scenario, Trajectory
from astro_dynamics.local import propagate_local


def _write_inputs(tmp_path: Path, *, process_noise: float = 0.0) -> tuple[Path, Path]:
    scenario = load_scenario("examples/scenarios/leo_covariance.yaml").model_copy(
        update={
            "scenario_id": "empirical-covariance-test",
            "covariance_process_noise_acceleration_km_s2": process_noise,
        }
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    predictor = propagate_local(scenario).model_copy(
        update={
            "metadata": {
                "covariance_implementation": "local_test_predictor",
                "covariance_units_policy": {
                    "frame": "EME2000",
                    "representation": "cartesian",
                    "time_scale": "UTC",
                    "state_order": ["x", "y", "z", "vx", "vy", "vz"],
                    "state_units": ["km", "km", "km", "km/s", "km/s", "km/s"],
                    "covariance_units_policy": "outer_product_of_state_units",
                },
            }
        }
    )
    predictor_path = tmp_path / "predictor.json"
    predictor_path.write_text(predictor.model_dump_json(indent=2), encoding="utf-8")
    return scenario_path, predictor_path


def _fake_tudat_truth(scenario: Scenario, backend: str) -> Trajectory:
    assert backend == "tudat"
    return propagate_local(scenario).model_copy(update={"backend": "tudat"})


def test_empirical_campaign_is_seeded_and_preserves_raw_truth(tmp_path: Path) -> None:
    scenario, predictor = _write_inputs(tmp_path)

    first = run_empirical_covariance_campaign(
        scenario,
        predictor,
        truth_backend="tudat",
        samples=8,
        seed=817,
        truth_propagator=_fake_tudat_truth,
    )
    second = run_empirical_covariance_campaign(
        scenario,
        predictor,
        truth_backend="tudat",
        samples=8,
        seed=817,
        truth_propagator=_fake_tudat_truth,
    )

    assert first == second
    assert first.campaign_provenance.sample_count == 8
    assert first.campaign_provenance.truth_backend == "tudat"
    assert len({sample.initial_state_perturbation for sample in first.samples}) == 8
    for sample in first.samples:
        assert sample.state_error == pytest.approx(
            tuple(
                realized - nominal
                for realized, nominal in zip(
                    sample.realized_truth_state,
                    sample.nominal_truth_state,
                    strict=True,
                )
            )
        )


def test_empirical_campaign_rejects_process_noise_without_realizations(
    tmp_path: Path,
) -> None:
    scenario, predictor = _write_inputs(tmp_path, process_noise=1.0e-9)

    with pytest.raises(InvalidScenarioError, match="zero covariance process noise"):
        run_empirical_covariance_campaign(
            scenario,
            predictor,
            truth_backend="tudat",
            samples=8,
            seed=817,
            truth_propagator=_fake_tudat_truth,
        )


def test_empirical_campaign_rejects_nonindependent_backend(tmp_path: Path) -> None:
    scenario, predictor = _write_inputs(tmp_path)

    with pytest.raises(InvalidScenarioError, match="must be independent"):
        run_empirical_covariance_campaign(
            scenario,
            predictor,
            truth_backend="local",
            samples=8,
            seed=817,
        )


def test_empirical_campaign_cli_reports_invalid_population(tmp_path: Path) -> None:
    scenario, predictor = _write_inputs(tmp_path, process_noise=1.0e-9)

    result = CliRunner().invoke(
        app,
        [
            "run-empirical-covariance-campaign",
            str(scenario),
            str(predictor),
            "--truth-backend",
            "tudat",
            "--samples",
            "8",
            "--output",
            str(tmp_path / "empirical.json"),
        ],
    )

    assert result.exit_code == 2
    assert "zero covariance process noise" in result.output
