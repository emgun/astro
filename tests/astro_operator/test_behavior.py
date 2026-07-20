from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from astro_core.errors import InvalidScenarioError
from astro_operator.behavior import (
    BehaviorDisposition,
    load_reasoner_behavior_corpus,
    load_reasoner_behavior_replay,
    score_reasoner_behavior_corpus,
)

CORPUS = Path("examples/operator/reasoner_behavior_corpus.yaml")
REPLAY = Path("examples/operator/reasoner_behavior_baseline_replay.yaml")


def test_checked_behavior_corpus_is_research_contract_ready() -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS)
    replay = load_reasoner_behavior_replay(REPLAY)

    first = score_reasoner_behavior_corpus(corpus, replay)
    second = score_reasoner_behavior_corpus(corpus, replay)

    assert first == second
    assert first.total_cases == 4
    assert first.matched_cases == 4
    assert first.coverage_complete
    assert first.behavior_gate_passed
    assert [result.actual_disposition for result in first.results] == [
        BehaviorDisposition.COMPLETED,
        BehaviorDisposition.COMPLETED,
        BehaviorDisposition.REASONER_INVALID_RESPONSE,
        BehaviorDisposition.BUDGET_EXHAUSTED,
    ]
    assert first.results[1].actual_evaluated_candidate_ids == (
        "candidate-a",
        "candidate-b",
    )


def test_behavior_substitution_blocks_readiness() -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS)
    replay = load_reasoner_behavior_replay(REPLAY)
    cases = list(corpus.cases)
    cases[0] = cases[0].model_copy(update={"description": "Description may change."})
    description_only = score_reasoner_behavior_corpus(
        corpus.model_copy(update={"cases": tuple(cases)}), replay
    )
    assert description_only.behavior_gate_passed

    cases[0] = cases[0].model_copy(
        update={"expected_selected_candidate_id": None}
    )
    substituted = score_reasoner_behavior_corpus(
        corpus.model_copy(update={"cases": tuple(cases)}), replay
    )
    assert not substituted.coverage_complete
    assert not substituted.behavior_gate_passed


def test_external_replay_is_the_candidate_under_test() -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS)
    replay = load_reasoner_behavior_replay(REPLAY)
    replay_cases = list(replay.cases)
    replay_cases[0] = replay_cases[0].model_copy(
        update={"decisions": replay_cases[0].decisions[:1]}
    )

    score = score_reasoner_behavior_corpus(
        corpus, replay.model_copy(update={"cases": tuple(replay_cases)})
    )

    assert score.results[0].actual_disposition == BehaviorDisposition.REASONER_INVALID_RESPONSE
    assert not score.results[0].matched
    assert not score.behavior_gate_passed


def test_replay_rejects_unbound_provider_identity(tmp_path: Path) -> None:
    payload = yaml.safe_load(REPLAY.read_text(encoding="utf-8"))
    payload["adapter"] = "openrouter"
    payload["model"] = "unverified-model"
    relabeled = tmp_path / "relabeled.yaml"
    relabeled.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="Extra inputs are not permitted"):
        load_reasoner_behavior_replay(relabeled)


def test_behavior_cli_writes_checked_score(tmp_path: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = tmp_path / "behavior-score.json"
    result = make_cli_runner().invoke(
        app,
        [
            "score-mission-reasoner-behavior",
            str(CORPUS),
            "--reasoner-replay",
            str(REPLAY),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["behavior_gate_passed"] is True
    assert payload["matched_cases"] == 4
