"""Invocation-bound recording artifacts for provider-neutral reasoner decisions."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_operator.behavior import (
    BehaviorDisposition,
    BehaviorFixtureEvaluator,
    ReasonerBehaviorCase,
    ReasonerBehaviorCorpus,
    ReasonerBehaviorReplay,
    ReasonerBehaviorResult,
    behavior_coverage_complete,
)
from astro_operator.engine import run_operator
from astro_operator.errors import (
    OperatorPolicyError,
    ReasonerCancelledError,
    ReasonerConfigurationError,
    ReasonerError,
    ReasonerInvalidResponseError,
    ReasonerUnavailableError,
)
from astro_operator.models import (
    OperatorActionKind,
    OperatorState,
    ReasonerAttemptProvenance,
    ReasonerDecision,
)
from astro_operator.reasoner import (
    ConditionalReplayReasoner,
    InvocationRecordingReasoner,
    invocation_digest,
    model_digest,
)


class RecordedFailureKind(StrEnum):
    INVALID_RESPONSE = "invalid_response"
    UNAVAILABLE = "unavailable"
    CONFIGURATION = "configuration"
    CANCELLED = "cancelled"


class RecordedReasonerFailure(AstroModel):
    sequence: int = Field(ge=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: RecordedFailureKind
    attempt: ReasonerAttemptProvenance | None = None
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def entry_digest_must_match(self) -> RecordedReasonerFailure:
        if self.entry_sha256 != recorded_failure_digest(
            self.sequence, self.input_sha256, self.kind, self.attempt
        ):
            raise ValueError("recorded terminal failure entry digest does not match")
        return self


class RecordedReasonerDecision(AstroModel):
    sequence: int = Field(ge=1)
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: ReasonerDecision
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_must_bind_to_state(self) -> RecordedReasonerDecision:
        invocation = self.decision.invocation
        if invocation.input_sha256 != self.state_sha256:
            raise ValueError("recorded decision input digest does not match state digest")
        if invocation.output_sha256 != model_digest(self.decision.action):
            raise ValueError("recorded decision output digest does not match action")
        if (
            invocation.record_sha256 is None
            or invocation.record_sha256 != invocation_digest(invocation)
        ):
            raise ValueError("recorded decision invocation record digest does not match")
        if self.entry_sha256 != recorded_decision_digest(
            self.sequence, self.state_sha256, self.decision
        ):
            raise ValueError("recorded decision entry digest does not match")
        return self


class ReasonerBehaviorRecordingCase(AstroModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    decisions: tuple[RecordedReasonerDecision, ...] = ()
    terminal_failure: RecordedReasonerFailure | None = None

    @model_validator(mode="after")
    def sequence_must_be_contiguous(self) -> ReasonerBehaviorRecordingCase:
        if [entry.sequence for entry in self.decisions] != list(
            range(1, len(self.decisions) + 1)
        ):
            raise ValueError("recorded decision sequences must be contiguous and one-based")
        if (
            self.terminal_failure is not None
            and self.terminal_failure.sequence != len(self.decisions) + 1
        ):
            raise ValueError("recorded terminal failure sequence must follow decisions")
        return self


class ReasonerBehaviorRecording(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    recording_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    call_cap: int | None = Field(default=None, ge=1)
    calls_attempted: int | None = Field(default=None, ge=0)
    complete: bool = True
    cases: tuple[ReasonerBehaviorRecordingCase, ...] = ()

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> ReasonerBehaviorRecording:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("reasoner recording case IDs must be unique")
        if (self.call_cap is None) != (self.calls_attempted is None):
            raise ValueError("recording call cap and attempted count must appear together")
        if (
            self.call_cap is not None
            and self.calls_attempted is not None
            and self.calls_attempted > self.call_cap
        ):
            raise ValueError("recording attempted calls exceed its call cap")
        return self


class ReasonerAttribution(AstroModel):
    adapter: str
    provider: str
    model: str


def load_reasoner_behavior_recording(path: Path | str) -> ReasonerBehaviorRecording:
    recording_path = Path(path)
    try:
        raw: Any = yaml.safe_load(recording_path.read_text(encoding="utf-8"))
        return ReasonerBehaviorRecording.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load reasoner behavior recording {recording_path}: {exc}"
        ) from exc


def write_reasoner_behavior_recording(
    path: Path | str, recording: ReasonerBehaviorRecording
) -> None:
    recording_path = Path(path)
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{recording_path.name}.", suffix=".tmp", dir=recording_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(recording.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(recording_path)
        recording_path.chmod(0o600)
        _fsync_directory(recording_path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def reserve_reasoner_behavior_recording(
    path: Path | str, recording: ReasonerBehaviorRecording
) -> None:
    """Exclusively reserve a mode-0600 output with an initial valid checkpoint."""

    recording_path = Path(path)
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{recording_path.name}.", suffix=".reserve", dir=recording_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(recording.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, recording_path)
        _fsync_directory(recording_path.parent)
    except BaseException:
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def recording_case_from_capture(
    case_id: str, recorder: InvocationRecordingReasoner
) -> ReasonerBehaviorRecordingCase:
    decisions = tuple(
        RecordedReasonerDecision(
            sequence=index,
            state_sha256=model_digest(state),
            decision=decision,
            entry_sha256=recorded_decision_digest(
                index, model_digest(state), decision
            ),
        )
        for index, (state, decision) in enumerate(
            zip(recorder.states, recorder.decisions, strict=True), start=1
        )
    )
    terminal_failure = None
    if recorder.failure is not None:
        failure_state = recorder.failure_state
        assert failure_state is not None
        attempt = (
            recorder.failure.attempt
            if isinstance(recorder.failure.attempt, ReasonerAttemptProvenance)
            else None
        )
        terminal_failure = RecordedReasonerFailure(
            sequence=len(decisions) + 1,
            input_sha256=model_digest(failure_state),
            kind=_failure_kind(recorder.failure),
            attempt=attempt,
            entry_sha256=recorded_failure_digest(
                len(decisions) + 1,
                model_digest(failure_state),
                _failure_kind(recorder.failure),
                attempt,
            ),
        )
    return ReasonerBehaviorRecordingCase(
        case_id=case_id,
        decisions=decisions,
        terminal_failure=terminal_failure,
    )


def record_reasoner_behavior_replay(
    corpus: ReasonerBehaviorCorpus,
    replay: ReasonerBehaviorReplay,
    *,
    recording_id: str,
) -> ReasonerBehaviorRecording:
    """Run a synthetic replay through the validated recorder without provider calls."""

    replay_by_id = {case.case_id: case for case in replay.cases}
    if {case.case_id for case in corpus.cases} != set(replay_by_id):
        raise ValueError("behavior replay cases must exactly match the corpus")
    recorded_cases: list[ReasonerBehaviorRecordingCase] = []
    for case in corpus.cases:
        replay_case = replay_by_id[case.case_id]
        recorder = InvocationRecordingReasoner(
            ConditionalReplayReasoner(replay_case.decisions)
        )
        evaluator = BehaviorFixtureEvaluator(case.observations)
        with suppress(ReasonerError, OperatorPolicyError):
            run_operator(
                objective=case.objective,
                authority=case.authority,
                reasoner=recorder,
                evaluator=evaluator,
            )
        recorded_cases.append(recording_case_from_capture(case.case_id, recorder))
    return ReasonerBehaviorRecording(
        schema_version="1.0",
        recording_id=recording_id,
        corpus_id=corpus.corpus_id,
        cases=tuple(recorded_cases),
    )


class RecordedBehaviorReasoner:
    """Replay a recording while retaining and revalidating original provenance."""

    def __init__(self, recording: ReasonerBehaviorRecordingCase) -> None:
        self._recording = recording
        self._cursor = 0
        self.terminal_failure_consumed = False
        self.exhausted_without_terminal = False
        self.action_kinds: list[OperatorActionKind] = []

    @property
    def complete(self) -> bool:
        return self._cursor == len(self._recording.decisions) and (
            self._recording.terminal_failure is None or self.terminal_failure_consumed
        ) and not self.exhausted_without_terminal

    def decide(self, state: OperatorState) -> ReasonerDecision:
        if self._cursor < len(self._recording.decisions):
            entry = self._recording.decisions[self._cursor]
            if entry.state_sha256 != model_digest(state):
                raise ReasonerInvalidResponseError(
                    "recorded reasoner state does not match playback state"
                )
            self._cursor += 1
            self.action_kinds.append(entry.decision.action.kind)
            return entry.decision.model_copy(deep=True)
        failure = self._recording.terminal_failure
        if failure is not None and not self.terminal_failure_consumed:
            if failure.sequence != self._cursor + 1:
                raise ReasonerInvalidResponseError(
                    "recorded terminal failure sequence does not match playback"
                )
            if failure.input_sha256 != model_digest(state):
                raise ReasonerInvalidResponseError(
                    "recorded terminal failure does not match playback state"
                )
            self.terminal_failure_consumed = True
            raise _failure_exception(failure.kind)
        self.exhausted_without_terminal = True
        raise ReasonerInvalidResponseError(
            "invocation-bound recording exhausted before finishing"
        )


def recording_attributions(
    recording: ReasonerBehaviorRecording,
) -> tuple[ReasonerAttribution, ...]:
    identities = {
        (
            entry.decision.invocation.adapter,
            entry.decision.invocation.provider,
            entry.decision.invocation.model,
        )
        for case in recording.cases
        for entry in case.decisions
    }
    return tuple(
        ReasonerAttribution(adapter=adapter, provider=provider, model=model)
        for adapter, provider, model in sorted(identities)
    )


class RecordedReasonerBehaviorScore(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    corpus_id: str
    recording_id: str
    recording_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attributions: tuple[ReasonerAttribution, ...]
    mixed_identity: bool
    total_cases: int = Field(ge=1)
    matched_cases: int = Field(ge=0)
    coverage_complete: bool
    recording_complete: bool
    behavior_gate_passed: bool
    results: tuple[ReasonerBehaviorResult, ...]


def score_recorded_reasoner_behavior_corpus(
    corpus: ReasonerBehaviorCorpus, recording: ReasonerBehaviorRecording
) -> RecordedReasonerBehaviorScore:
    recording_by_id = {case.case_id: case for case in recording.cases}
    scored = tuple(
        _score_recorded_case(case, recording_by_id.get(case.case_id))
        for case in corpus.cases
    )
    results = tuple(item[0] for item in scored)
    recording_complete = recording.complete and all(item[1] for item in scored)
    coverage_complete = (
        recording.corpus_id == corpus.corpus_id
        and behavior_coverage_complete(corpus, results, set(recording_by_id))
    )
    matched = sum(result.matched for result in results)
    attributions = recording_attributions(recording)
    return RecordedReasonerBehaviorScore(
        schema_version="1.0",
        corpus_id=corpus.corpus_id,
        recording_id=recording.recording_id,
        recording_sha256=reasoner_behavior_recording_digest(recording),
        attributions=attributions,
        mixed_identity=len(attributions) > 1,
        total_cases=len(results),
        matched_cases=matched,
        coverage_complete=coverage_complete,
        recording_complete=recording_complete,
        behavior_gate_passed=(
            matched == len(results)
            and coverage_complete
            and recording_complete
            and len(attributions) == 1
        ),
        results=results,
    )


def _score_recorded_case(
    case: ReasonerBehaviorCase,
    recording: ReasonerBehaviorRecordingCase | None,
) -> tuple[ReasonerBehaviorResult, bool]:
    recording = recording or ReasonerBehaviorRecordingCase(case_id=case.case_id)
    reasoner = RecordedBehaviorReasoner(recording)
    evaluator = BehaviorFixtureEvaluator(case.observations)
    selected_candidate_id: str | None = None
    diagnostic: str | None = None
    try:
        run = run_operator(
            objective=case.objective,
            authority=case.authority,
            reasoner=reasoner,
            evaluator=evaluator,
        )
    except ReasonerUnavailableError as exc:
        disposition = BehaviorDisposition.REASONER_UNAVAILABLE
        diagnostic = str(exc)
    except ReasonerConfigurationError as exc:
        disposition = BehaviorDisposition.REASONER_CONFIGURATION
        diagnostic = str(exc)
    except ReasonerCancelledError as exc:
        disposition = BehaviorDisposition.REASONER_CANCELLED
        diagnostic = str(exc)
    except ReasonerInvalidResponseError as exc:
        disposition = BehaviorDisposition.REASONER_INVALID_RESPONSE
        diagnostic = str(exc)
    except OperatorPolicyError as exc:
        disposition = BehaviorDisposition.POLICY_REJECTED
        diagnostic = str(exc)
    else:
        disposition = BehaviorDisposition(run.status.value)
        selected_candidate_id = run.selected_candidate_id
    actual_actions = tuple(reasoner.action_kinds)
    actual_candidates = tuple(evaluator.evaluated_candidate_ids)
    matched = (
        disposition == case.expected_disposition
        and actual_actions == case.expected_action_kinds
        and actual_candidates == case.expected_evaluated_candidate_ids
        and selected_candidate_id == case.expected_selected_candidate_id
    )
    return (
        ReasonerBehaviorResult(
            case_id=case.case_id,
            expected_disposition=case.expected_disposition,
            actual_disposition=disposition,
            expected_action_kinds=case.expected_action_kinds,
            actual_action_kinds=actual_actions,
            expected_evaluated_candidate_ids=case.expected_evaluated_candidate_ids,
            actual_evaluated_candidate_ids=actual_candidates,
            expected_selected_candidate_id=case.expected_selected_candidate_id,
            actual_selected_candidate_id=selected_candidate_id,
            matched=matched,
            diagnostic=diagnostic,
        ),
        reasoner.complete,
    )


def reasoner_behavior_recording_digest(recording: ReasonerBehaviorRecording) -> str:
    return sha256(_canonical_json(recording.model_dump(mode="json"))).hexdigest()


def recorded_decision_digest(
    sequence: int, state_sha256: str, decision: ReasonerDecision
) -> str:
    assert decision.invocation.record_sha256 is not None
    return sha256(
        _canonical_json(
            {
                "sequence": sequence,
                "state_sha256": state_sha256,
                "action_sha256": model_digest(decision.action),
                "invocation_sha256": invocation_digest(decision.invocation),
            }
        )
    ).hexdigest()


def recorded_failure_digest(
    sequence: int,
    input_sha256: str,
    kind: RecordedFailureKind,
    attempt: ReasonerAttemptProvenance | None,
) -> str:
    return sha256(
        _canonical_json(
            {
                "sequence": sequence,
                "input_sha256": input_sha256,
                "kind": kind.value,
                "attempt": attempt.model_dump(mode="json") if attempt is not None else None,
            }
        )
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _failure_kind(error: ReasonerError) -> RecordedFailureKind:
    if isinstance(error, ReasonerUnavailableError):
        return RecordedFailureKind.UNAVAILABLE
    if isinstance(error, ReasonerConfigurationError):
        return RecordedFailureKind.CONFIGURATION
    if isinstance(error, ReasonerCancelledError):
        return RecordedFailureKind.CANCELLED
    return RecordedFailureKind.INVALID_RESPONSE


def _failure_exception(kind: RecordedFailureKind) -> ReasonerError:
    message = f"recorded reasoner terminal failure: {kind.value}"
    if kind == RecordedFailureKind.UNAVAILABLE:
        return ReasonerUnavailableError(message)
    if kind == RecordedFailureKind.CONFIGURATION:
        return ReasonerConfigurationError(message)
    if kind == RecordedFailureKind.CANCELLED:
        return ReasonerCancelledError(message)
    return ReasonerInvalidResponseError(message)
