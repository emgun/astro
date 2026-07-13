from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from astro_cli.main import app

runner = CliRunner()


def _campaign(
    tmp_path: Path,
    *,
    extractor: str = "orbit.final_altitude_km",
    samples: int = 2,
    parameter_id: str = "mass",
    parameter_target: str = "orbit.spacecraft.mass_kg",
    parameter_unit: str = "kg",
    parameter_low: float = 900.0,
    parameter_high: float = 901.0,
) -> Path:
    source = Path("examples/scenarios/leo_two_body.yaml").resolve()
    definition = {
        "campaign_id": "cli-orbit",
        "workflow": {"kind": "orbit", "scenario": str(source)},
        "uncertainty": {
            "parameters": [
                {
                    "parameter_id": parameter_id,
                    "target": parameter_target,
                    "unit": parameter_unit,
                    "uncertainty_kind": "epistemic",
                    "distribution": {
                        "kind": "uniform",
                        "low": parameter_low,
                        "high": parameter_high,
                    },
                }
            ]
        },
        "sampler": {"kind": "pseudorandom", "samples": samples, "seed": 7},
        "evaluator": {
            "evaluator_id": "local-orbit",
            "kind": "authoritative",
            "workflow": "orbit",
            "implementation_version": "1",
            "backend": "local",
            "claim_boundary": "test evidence",
        },
        "metrics": [
            {
                "metric_id": "altitude",
                "extractor": extractor,
                "value_kind": "numeric",
                "unit": "km",
            }
        ],
        "requirements": [
            {
                "requirement_id": "above_earth",
                "metric_id": "altitude",
                "operator": "gt",
                "value": 0.0,
            }
        ],
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    return path


def test_validate_campaign_resolves_without_propagating(tmp_path: Path, monkeypatch) -> None:
    path = _campaign(tmp_path)
    monkeypatch.setattr(
        "astro_uq.cli.propagate_with_backend",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    result = runner.invoke(app, ["validate-campaign", str(path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"campaign_id": "cli-orbit", "valid": True}


def test_validate_campaign_rejects_unknown_metric(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-campaign", str(_campaign(tmp_path, extractor="bad"))])

    assert result.exit_code == 2
    assert "unregistered metric extractor" in result.output


def test_run_resume_and_regenerate_summary(tmp_path: Path) -> None:
    path = _campaign(tmp_path)
    output = tmp_path / "output"

    first = runner.invoke(app, ["run-campaign", str(path), "--output-dir", str(output)])
    resumed = runner.invoke(
        app, ["run-campaign", str(path), "--output-dir", str(output), "--resume"]
    )
    (output / "statistics.json").unlink()
    (output / "summary.txt").unlink()
    summarized = runner.invoke(app, ["summarize-campaign", str(output)])

    assert first.exit_code == 0
    assert resumed.exit_code == 0
    assert len((output / "cases.jsonl").read_text().splitlines()) == 2
    assert summarized.exit_code == 0
    assert "Completed: 2/2" in summarized.stdout
    assert (output / "statistics.json").exists()
    assert (output / "summary.txt").exists()


def test_profile_campaign_writes_machine_scoped_timing_product(tmp_path: Path) -> None:
    output = tmp_path / "campaign-output"
    profile_path = tmp_path / "profile.json"
    run = runner.invoke(
        app,
        ["run-campaign", str(_campaign(tmp_path)), "--output-dir", str(output)],
    )

    profiled = runner.invoke(
        app,
        ["profile-campaign", str(output), "--output", str(profile_path)],
    )

    assert run.exit_code == 0
    assert profiled.exit_code == 0
    profile = json.loads(profiled.stdout)
    assert profile["evidence_scope"] == "machine_scoped"
    assert profile["timing"]["case_count"] == 2
    assert profile["timing"]["fully_instrumented_case_count"] == 2
    assert profile["cases_digest"]
    assert profile["software_compatibility"]["campaign-runtime"] == "1.1"
    assert profile["machine"]["architecture"]
    assert profile["runtime"]["python_version"]
    assert profile["timing"]["evaluation_share_of_instrumented_time"] is not None
    assert json.loads(profile_path.read_text()) == profile


def test_profile_campaign_rejects_output_inside_source_evidence(tmp_path: Path) -> None:
    output = tmp_path / "campaign-output"
    run = runner.invoke(
        app,
        ["run-campaign", str(_campaign(tmp_path)), "--output-dir", str(output)],
    )

    profiled = runner.invoke(
        app,
        ["profile-campaign", str(output), "--output", str(output / "statistics.json")],
    )
    integrity_check = runner.invoke(app, ["profile-campaign", str(output)])

    assert run.exit_code == 0
    assert profiled.exit_code == 2
    assert "outside the source campaign directory" in profiled.output
    assert integrity_check.exit_code == 0


def test_profile_campaign_rejects_interrupted_evidence(tmp_path: Path) -> None:
    output = tmp_path / "campaign-output"
    run = runner.invoke(
        app,
        ["run-campaign", str(_campaign(tmp_path)), "--output-dir", str(output)],
    )
    manifest_path = output / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = "interrupted"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    profiled = runner.invoke(app, ["profile-campaign", str(output)])

    assert run.exit_code == 0
    assert profiled.exit_code == 2
    assert "only completed campaigns can be profiled" in profiled.output


def test_analyze_campaign_sensitivity_writes_digest_bound_report(tmp_path: Path) -> None:
    campaign = _campaign(
        tmp_path,
        samples=32,
        parameter_id="position_x",
        parameter_target="orbit.initial_state.cartesian.position_x_km",
        parameter_unit="km",
        parameter_low=6999.0,
        parameter_high=7001.0,
    )
    output = tmp_path / "campaign-output"
    report_path = tmp_path / "sensitivity.json"
    run = runner.invoke(app, ["run-campaign", str(campaign), "--output-dir", str(output)])

    analyzed = runner.invoke(
        app,
        [
            "analyze-campaign-sensitivity",
            str(output),
            "--metric",
            "altitude",
            "--requirement-margin",
            "above_earth",
            "--output",
            str(report_path),
        ],
    )

    assert run.exit_code == 0
    assert analyzed.exit_code == 0, analyzed.output
    report = json.loads(analyzed.stdout)
    assert report["evidence_scope"] == "campaign_design_space"
    assert report["sample_count"] == 32
    assert report["samples_digest"]
    assert report["cases_digest"]
    assert {target["target_id"] for target in report["targets"]} == {
        "altitude",
        "above_earth",
    }
    assert all(
        target["largest_absolute_prcc_parameter_id"] == "position_x"
        for target in report["targets"]
    )
    assert json.loads(report_path.read_text()) == report


def test_sensitivity_output_cannot_overwrite_campaign_evidence(tmp_path: Path) -> None:
    campaign = _campaign(
        tmp_path,
        samples=32,
        parameter_id="position_x",
        parameter_target="orbit.initial_state.cartesian.position_x_km",
        parameter_unit="km",
        parameter_low=6999.0,
        parameter_high=7001.0,
    )
    output = tmp_path / "campaign-output"
    run = runner.invoke(app, ["run-campaign", str(campaign), "--output-dir", str(output)])

    analyzed = runner.invoke(
        app,
        [
            "analyze-campaign-sensitivity",
            str(output),
            "--metric",
            "altitude",
            "--output",
            str(output / "statistics.json"),
        ],
    )
    integrity_check = runner.invoke(app, ["profile-campaign", str(output)])

    assert run.exit_code == 0
    assert analyzed.exit_code == 2
    assert "outside the source campaign directory" in analyzed.output
    assert integrity_check.exit_code == 0


def test_run_campaign_dry_run_and_max_cases_do_not_write_evidence(tmp_path: Path) -> None:
    path = _campaign(tmp_path)
    output = tmp_path / "dry-run"

    result = runner.invoke(
        app,
        [
            "run-campaign",
            str(path),
            "--output-dir",
            str(output),
            "--max-cases",
            "1",
            "--workers",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "campaign_id": "cli-orbit",
        "configured_samples": 2,
        "planned_samples": 1,
        "valid": True,
        "workers": 2,
    }
    assert not output.exists()


def test_run_campaign_supports_two_workers_and_max_cases(tmp_path: Path) -> None:
    output = tmp_path / "parallel"
    result = runner.invoke(
        app,
        [
            "run-campaign",
            str(_campaign(tmp_path)),
            "--output-dir",
            str(output),
            "--max-cases",
            "1",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metadata"]["workers"] == 2
    assert payload["statistics"]["completed_samples"] == 1


def test_malformed_campaign_is_json_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    result = runner.invoke(app, ["validate-campaign", str(path)])

    assert result.exit_code == 2
    assert json.loads(result.output)["error"] == "campaign definition must contain a mapping"


def test_execution_failure_reports_sample_and_nonzero(tmp_path: Path, monkeypatch) -> None:
    path = _campaign(tmp_path)
    monkeypatch.setattr(
        "astro_uq.cli.propagate_with_backend",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    result = runner.invoke(
        app, ["run-campaign", str(path), "--output-dir", str(tmp_path / "failed")]
    )

    assert result.exit_code == 1
    errors = [json.loads(line) for line in result.output.splitlines()]
    assert len(errors) == 2
    assert all(error["sample_id"] for error in errors)
