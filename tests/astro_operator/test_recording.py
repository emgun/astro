from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.behavior import (
    load_reasoner_behavior_corpus,
    load_reasoner_behavior_replay,
)
from astro_operator.models import OperatorActionKind
from astro_operator.reasoner import invocation_digest
from astro_operator.recording import (
    RecordedFailureKind,
    load_reasoner_behavior_recording,
    record_reasoner_behavior_replay,
    recorded_decision_digest,
    score_recorded_reasoner_behavior_corpus,
    write_reasoner_behavior_recording,
)

CORPUS_PATH = Path("examples/operator/reasoner_behavior_corpus.yaml")
REPLAY_PATH = Path("examples/operator/reasoner_behavior_baseline_replay.yaml")


def _recording():
    return record_reasoner_behavior_replay(
        load_reasoner_behavior_corpus(CORPUS_PATH),
        load_reasoner_behavior_replay(REPLAY_PATH),
        recording_id="checked-bound-baseline",
    )


def test_invocation_bound_recording_round_trips_and_scores(tmp_path: Path) -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS_PATH)
    recording = _recording()
    path = tmp_path / "recording.json"
    write_reasoner_behavior_recording(path, recording)

    loaded = load_reasoner_behavior_recording(path)
    score = score_recorded_reasoner_behavior_corpus(corpus, loaded)

    assert loaded == recording
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert score.matched_cases == 4
    assert score.recording_complete
    assert score.coverage_complete
    assert score.behavior_gate_passed
    assert not score.mixed_identity
    assert [item.model_dump() for item in score.attributions] == [
        {
            "adapter": "conditional-replay",
            "provider": "deterministic-replay",
            "model": "checked-fixture",
        }
    ]
    unmatched = next(case for case in loaded.cases if case.case_id == "unmatched-branch")
    assert unmatched.terminal_failure is not None
    assert unmatched.terminal_failure.kind == RecordedFailureKind.INVALID_RESPONSE


@pytest.mark.parametrize("field", ["provider", "model", "adapter"])
def test_loader_rejects_relabelled_invocation(tmp_path: Path, field: str) -> None:
    path = tmp_path / "recording.json"
    write_reasoner_behavior_recording(path, _recording())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["decisions"][0]["decision"]["invocation"][field] = "forged"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="record digest"):
        load_reasoner_behavior_recording(path)


def test_loader_rejects_forged_entry_digest(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    write_reasoner_behavior_recording(path, _recording())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["decisions"][0]["entry_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="entry digest"):
        load_reasoner_behavior_recording(path)


def test_loader_rejects_relabelled_terminal_failure(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    write_reasoner_behavior_recording(path, _recording())
    payload = json.loads(path.read_text(encoding="utf-8"))
    unmatched = next(
        case for case in payload["cases"] if case["case_id"] == "unmatched-branch"
    )
    unmatched["terminal_failure"]["kind"] = "unavailable"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="terminal failure entry digest"):
        load_reasoner_behavior_recording(path)


def test_recording_persists_state_digest_not_secret_state(tmp_path: Path) -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS_PATH)
    cases = list(corpus.cases)
    cases[0] = cases[0].model_copy(
        update={
            "objective": cases[0].objective.model_copy(
                update={"metadata": {"private_token": "SENTINEL-SECRET"}}
            )
        }
    )
    recording = record_reasoner_behavior_replay(
        corpus.model_copy(update={"cases": tuple(cases)}),
        load_reasoner_behavior_replay(REPLAY_PATH),
        recording_id="secret-free",
    )
    path = tmp_path / "recording.json"
    write_reasoner_behavior_recording(path, recording)

    encoded = path.read_text(encoding="utf-8")
    assert "SENTINEL-SECRET" not in encoded
    assert "private_token" not in encoded
    assert "state_sha256" in encoded
    assert '"state"' not in encoded


def test_policy_rejected_decision_is_recordable_and_complete() -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS_PATH)
    cases = list(corpus.cases)
    first = cases[0]
    cases[0] = first.model_copy(
        update={
            "authority": first.authority.model_copy(
                update={"allowed_actions": (OperatorActionKind.FINISH,)}
            )
        }
    )
    modified = corpus.model_copy(update={"cases": tuple(cases)})

    recording = record_reasoner_behavior_replay(
        modified,
        load_reasoner_behavior_replay(REPLAY_PATH),
        recording_id="policy-rejected",
    )
    score = score_recorded_reasoner_behavior_corpus(modified, recording)

    assert score.results[0].actual_disposition.value == "policy_rejected"
    assert score.recording_complete


def test_mixed_valid_identity_is_reported_and_not_promoted() -> None:
    corpus = load_reasoner_behavior_corpus(CORPUS_PATH)
    recording = _recording()
    cases = list(recording.cases)
    entries = list(cases[0].decisions)
    entry = entries[0]
    invocation = entry.decision.invocation.model_copy(update={"model": "other-fixture"})
    invocation = invocation.model_copy(
        update={"record_sha256": invocation_digest(invocation)}
    )
    decision = entry.decision.model_copy(update={"invocation": invocation})
    entries[0] = entry.model_copy(
        update={
            "decision": decision,
            "entry_sha256": recorded_decision_digest(
                entry.sequence, entry.state_sha256, decision
            ),
        }
    )
    cases[0] = cases[0].model_copy(update={"decisions": tuple(entries)})
    mixed = recording.model_copy(update={"cases": tuple(cases)})

    score = score_recorded_reasoner_behavior_corpus(corpus, mixed)

    assert score.mixed_identity
    assert len(score.attributions) == 2
    assert not score.behavior_gate_passed


def test_recording_cli_binds_and_scores_without_provider(tmp_path: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    recording = tmp_path / "recording.json"
    bound = make_cli_runner().invoke(
        app,
        [
            "bind-mission-reasoner-replay",
            str(CORPUS_PATH),
            "--reasoner-replay",
            str(REPLAY_PATH),
            "--recording-id",
            "cli-bound",
            "--output",
            str(recording),
        ],
    )
    assert bound.exit_code == 0, bound.output

    scored = make_cli_runner().invoke(
        app,
        [
            "score-mission-reasoner-behavior",
            str(CORPUS_PATH),
            "--reasoner-recording",
            str(recording),
        ],
    )
    assert scored.exit_code == 0, scored.output
    assert json.loads(scored.output)["behavior_gate_passed"] is True

    ambiguous = make_cli_runner().invoke(
        app,
        [
            "score-mission-reasoner-behavior",
            str(CORPUS_PATH),
            "--reasoner-replay",
            str(REPLAY_PATH),
            "--reasoner-recording",
            str(recording),
        ],
    )
    assert ambiguous.exit_code == 2
