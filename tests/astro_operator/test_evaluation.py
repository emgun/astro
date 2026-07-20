from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from astro_core.errors import InvalidScenarioError
from astro_operator.evaluation import (
    DecisionDisposition,
    load_adversarial_corpus,
    score_adversarial_corpus,
)

CORPUS = Path("examples/operator/reasoner_adversarial_corpus.yaml")


def test_checked_adversarial_corpus_promotes_contract() -> None:
    corpus = load_adversarial_corpus(CORPUS)

    first = score_adversarial_corpus(corpus)
    second = score_adversarial_corpus(corpus)

    assert first == second
    assert first.promoted
    assert first.total_cases == 8
    assert first.matched_cases == 8
    assert first.safety_critical_cases == 7
    assert first.safety_critical_matched == 7
    assert first.coverage_complete
    assert {result.actual_disposition for result in first.results} == {
        DecisionDisposition.ACCEPTED,
        DecisionDisposition.SCHEMA_REJECTED,
        DecisionDisposition.POLICY_REJECTED,
    }


def test_mismatched_expectation_blocks_promotion() -> None:
    corpus = load_adversarial_corpus(CORPUS)
    cases = list(corpus.cases)
    cases[0] = cases[0].model_copy(
        update={"expected_disposition": DecisionDisposition.POLICY_REJECTED}
    )

    score = score_adversarial_corpus(corpus.model_copy(update={"cases": tuple(cases)}))

    assert not score.promoted
    assert score.matched_cases == 7
    assert not score.results[0].matched


def test_incomplete_corpus_cannot_promote() -> None:
    corpus = load_adversarial_corpus(CORPUS)

    score = score_adversarial_corpus(corpus.model_copy(update={"cases": corpus.cases[:1]}))

    assert score.matched_cases == 1
    assert not score.coverage_complete
    assert not score.promoted


def test_case_id_cannot_substitute_for_canonical_behavior() -> None:
    corpus = load_adversarial_corpus(CORPUS)
    cases = list(corpus.cases)
    cases[4] = cases[4].model_copy(update={"response": cases[5].response})

    score = score_adversarial_corpus(corpus.model_copy(update={"cases": tuple(cases)}))

    assert score.results[4].actual_disposition == DecisionDisposition.POLICY_REJECTED
    assert score.results[4].matched
    assert not score.coverage_complete
    assert not score.promoted


def test_loader_rejects_inconsistent_operator_state(tmp_path: Path) -> None:
    payload = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    payload["cases"][0]["state"]["remaining_steps"] = 2
    invalid = tmp_path / "invalid-state.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="remaining step budget is inconsistent"):
        load_adversarial_corpus(invalid)


def test_loader_rejects_mismatched_historical_observation(tmp_path: Path) -> None:
    payload = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    state = payload["cases"][0]["state"]
    state["remaining_steps"] = 2
    state["remaining_candidate_evaluations"] = 1
    state["steps"] = [
        {
            "sequence": 1,
            "action": {
                "action_id": "evaluate-a",
                "kind": "evaluate_candidate",
                "rationale": "Evaluate candidate A.",
                "candidate": {
                    "candidate_id": "candidate-a",
                    "assignments": {"wet_mass_kg": 500.0},
                },
            },
            "observation": {
                "candidate": {
                    "candidate_id": "candidate-b",
                    "assignments": {"wet_mass_kg": 500.0},
                },
                "evaluation_status": "evaluated",
                "passed": True,
                "metrics": [{"metric_id": "reserve", "value": 1.0, "unit": "kg"}],
            },
        }
    ]
    invalid = tmp_path / "mismatched-history.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="observation does not match its action"):
        load_adversarial_corpus(invalid)


def test_corpus_cli_writes_report_and_fails_closed(tmp_path: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    report_path = tmp_path / "score.json"
    passed = make_cli_runner().invoke(
        app,
        ["score-mission-reasoner-corpus", str(CORPUS), "--output", str(report_path)],
    )
    assert passed.exit_code == 0, passed.output
    assert json.loads(report_path.read_text(encoding="utf-8"))["promoted"] is True

    payload = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_disposition"] = "policy_rejected"
    failing_corpus = tmp_path / "failing.yaml"
    failing_corpus.write_text(yaml.safe_dump(payload), encoding="utf-8")
    blocked = make_cli_runner().invoke(
        app,
        ["score-mission-reasoner-corpus", str(failing_corpus)],
    )
    assert blocked.exit_code == 1
    assert json.loads(blocked.output)["promoted"] is False
