from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.conditional_campaign import (
    ConditionalCampaignBinding,
    ConditionalCampaignExecutionSpec,
    build_conditional_campaign_outcome,
)
from astro_operator.conditional_campaign_io import (
    _verify_campaign_evidence,
    load_conditional_campaign_spec,
    run_conditional_campaign,
    verify_conditional_campaign,
)
from astro_uq.io import (
    CAMPAIGN_FILE,
    CASES_FILE,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
)
from astro_uq.models import CampaignDefinition, OutcomeStatus

DIRECTOR_SPEC = Path("examples/design/leo_mission_design_director.yaml")
REPLAY = Path("examples/operator/leo_lifecycle_trade_study_replay.yaml")
EXECUTION_SPEC = Path("examples/design/leo_mission_design_conditional_campaign.yaml")


def _digest(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run_director(tmp_path: Path) -> Path:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    director = tmp_path / "director"
    runner = make_cli_runner()
    result = runner.invoke(
        app,
        [
            "run-mission-design-director",
            str(DIRECTOR_SPEC),
            "--reasoner-replay",
            str(REPLAY),
            "--output-dir",
            str(director),
        ],
    )
    assert result.exit_code == 0, result.output
    return director


def _run_bundle(tmp_path: Path) -> Path:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    director = _run_director(tmp_path)
    runner = make_cli_runner()
    output = tmp_path / "conditional"
    result = runner.invoke(
        app,
        [
            "run-mission-design-conditional-campaign",
            str(director),
            str(EXECUTION_SPEC),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    return output


def test_conditional_campaign_executes_retains_and_verifies_after_relocation(
    tmp_path: Path,
) -> None:
    output = _run_bundle(tmp_path)

    outcome = verify_conditional_campaign(output)
    assert outcome.disposition == "retain"
    assert outcome.reason == "all_declared_design_space_gates_passed"
    assert outcome.candidate_id == "higher-reserve"
    assert outcome.baseline_id == "leo-mission-design:baseline"
    assert outcome.completed_samples == outcome.requested_samples == 8
    assert outcome.outcome_counts == {"success": 8}
    assert {
        item.director_requirement_id: item.observed_pass_fraction
        for item in outcome.gate_assessments
    } == {
        "deorbit-propellant-reserve": 1.0,
        "entry-interface-continuity": 1.0,
    }

    relocated = tmp_path / "relocated"
    shutil.copytree(output, relocated)
    shutil.rmtree(output)
    assert verify_conditional_campaign(relocated) == outcome


def test_conditional_campaign_rejects_extra_artifact(
    tmp_path: Path,
) -> None:
    output = _run_bundle(tmp_path)
    (output / "extra.txt").write_text("not inventoried\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="inventory"):
        verify_conditional_campaign(output)


def test_conditional_campaign_rejects_requirement_boolean_not_derived_from_metric(
    tmp_path: Path,
) -> None:
    output = _run_bundle(tmp_path)
    cases_path = output / "campaign" / CASES_FILE
    cases = read_jsonl(cases_path)
    requirement = next(
        item
        for item in cases[0]["requirements"]
        if item["requirement_id"] == "deorbit_propellant_reserve"
    )
    assert cases[0]["metric_values"]["propellant_reserve_margin"] > 0.0
    requirement["passed"] = False
    atomic_write_jsonl(cases_path, cases)
    manifest_path = output / "campaign" / CAMPAIGN_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases_digest"] = canonical_hash(cases)
    atomic_write_json(manifest_path, manifest)
    definition = CampaignDefinition.model_validate(manifest["definition"])

    with pytest.raises(
        ValueError,
        match="requirements do not match captured metric values",
    ):
        _verify_campaign_evidence(output / "campaign", definition)


def test_conditional_campaign_rejects_symlink_root(tmp_path: Path) -> None:
    output = _run_bundle(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(InvalidScenarioError, match="symbolic link"):
        verify_conditional_campaign(linked)


def test_conditional_campaign_resume_preserves_completed_cases_after_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import astro_uq.runner as uq_runner

    director = _run_director(tmp_path)
    output = tmp_path / "interrupted"
    original = uq_runner._evaluate_sample
    calls = 0

    def interrupt_after_two(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(uq_runner, "_evaluate_sample", interrupt_after_two)
    with pytest.raises(KeyboardInterrupt):
        run_conditional_campaign(
            director_root=director,
            spec_path=EXECUTION_SPEC,
            output_dir=output,
        )
    assert len(read_jsonl(output / "campaign" / CASES_FILE)) == 2

    monkeypatch.setattr(uq_runner, "_evaluate_sample", original)
    outcome = run_conditional_campaign(
        director_root=director,
        spec_path=EXECUTION_SPEC,
        output_dir=output,
        resume=True,
    )
    assert outcome.disposition == "retain"
    assert outcome.completed_samples == 8


def test_conditional_campaign_resume_republishes_completed_evidence(
    tmp_path: Path,
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = _run_bundle(tmp_path)
    expected = verify_conditional_campaign(output)
    cases_before = (output / "campaign/cases.jsonl").read_bytes()
    (output / "conditional-campaign-manifest.json").unlink()
    (output / "conditional-campaign-outcome.json").unlink()

    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-design-conditional-campaign",
            str(tmp_path / "director"),
            str(EXECUTION_SPEC),
            "--output-dir",
            str(output),
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "campaign/cases.jsonl").read_bytes() == cases_before
    assert verify_conditional_campaign(output) == expected


def test_conditional_campaign_reconstructs_self_consistent_outcome(
    tmp_path: Path,
) -> None:
    output = _run_bundle(tmp_path)
    outcome_path = output / "conditional-campaign-outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["disposition"] = "revise"
    outcome["reason"] = "declared_design_space_gate_failed"
    outcome["outcome_sha256"] = _digest(
        {key: value for key, value in outcome.items() if key != "outcome_sha256"}
    )
    outcome_path.write_text(
        json.dumps(outcome, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = output / "conditional-campaign-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["path"] == "conditional-campaign-outcome.json"
    )
    artifact["sha256"] = sha256(outcome_path.read_bytes()).hexdigest()
    artifact["size_bytes"] = outcome_path.stat().st_size
    manifest["outcome_sha256"] = outcome["outcome_sha256"]
    manifest["disposition"] = "revise"
    manifest["manifest_sha256"] = _digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidScenarioError, match="outcome does not match"):
        verify_conditional_campaign(output)


def test_conditional_campaign_reducer_revises_and_abstains(
    tmp_path: Path,
) -> None:
    output = _run_bundle(tmp_path)
    spec = load_conditional_campaign_spec(output / "inputs/execution-spec.yaml")
    binding = ConditionalCampaignBinding.model_validate_json(
        (output / "inputs/binding.json").read_text(encoding="utf-8")
    )
    definition = CampaignDefinition.model_validate_json(
        (output / "inputs/bound-campaign-definition.json").read_text(encoding="utf-8")
    )
    evidence = _verify_campaign_evidence(output / "campaign", definition)
    first = evidence.cases[0]
    revised_requirements = tuple(
        item.model_copy(update={"passed": False, "margin": -1.0})
        if item.requirement_id == "entry_interface_continuity"
        else item
        for item in first.requirements
    )
    revised_cases = (
        first.model_copy(update={"requirements": revised_requirements}),
        *evidence.cases[1:],
    )
    revised = build_conditional_campaign_outcome(
        spec=spec,
        binding=binding,
        definition=definition,
        campaign_state=evidence.campaign_state,
        samples=evidence.samples,
        cases=revised_cases,
        statistics=evidence.statistics,
        samples_sha256=evidence.samples_sha256,
        cases_sha256="0" * 64,
        statistics_sha256=evidence.statistics_sha256,
    )
    assert revised.disposition == "revise"
    assert revised.reason == "declared_design_space_gate_failed"

    failed_cases = (
        first.model_copy(
            update={
                "outcome_status": OutcomeStatus.EXECUTION_FAILURE,
                "requirements": (),
            }
        ),
        *evidence.cases[1:],
    )
    abstained = build_conditional_campaign_outcome(
        spec=spec,
        binding=binding,
        definition=definition,
        campaign_state=evidence.campaign_state,
        samples=evidence.samples,
        cases=failed_cases,
        statistics=evidence.statistics,
        samples_sha256=evidence.samples_sha256,
        cases_sha256="0" * 64,
        statistics_sha256=evidence.statistics_sha256,
    )
    assert abstained.disposition == "abstain"
    assert abstained.reason == "campaign_case_failure"


@pytest.mark.parametrize("fraction", [True, "1.0"])
def test_conditional_campaign_rejects_coercive_acceptance_fraction(
    fraction: object,
) -> None:
    import yaml

    payload = yaml.safe_load(EXECUTION_SPEC.read_text(encoding="utf-8"))
    payload["acceptance_gates"][0]["minimum_pass_fraction"] = fraction
    with pytest.raises(ValueError, match="pass fraction must be numeric"):
        ConditionalCampaignExecutionSpec.model_validate(payload)
