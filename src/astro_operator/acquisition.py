"""Bounded provider-neutral acquisition of invocation-bound behavior recordings."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress

from astro_operator.behavior import BehaviorFixtureEvaluator, ReasonerBehaviorCorpus
from astro_operator.engine import run_operator
from astro_operator.errors import (
    OperatorPolicyError,
    ReasonerCancelledError,
    ReasonerError,
)
from astro_operator.models import OperatorState, ReasonerDecision
from astro_operator.reasoner import InvocationRecordingReasoner, MissionReasoner
from astro_operator.recording import (
    ReasonerBehaviorRecording,
    ReasonerBehaviorRecordingCase,
    recording_case_from_capture,
)


class CallCappedReasoner:
    """Enforce one hard call budget across every benchmark case."""

    def __init__(self, reasoner: MissionReasoner, *, max_calls: int) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("reasoner max_calls must be a positive integer")
        self._reasoner = reasoner
        self.max_calls = max_calls
        self.calls_attempted = 0

    def decide(self, state: OperatorState) -> ReasonerDecision:
        if self.calls_attempted >= self.max_calls:
            raise ReasonerCancelledError("reasoner acquisition call cap exhausted")
        self.calls_attempted += 1
        return self._reasoner.decide(state)


def acquire_reasoner_behavior_recording(
    corpus: ReasonerBehaviorCorpus,
    reasoner: MissionReasoner,
    *,
    recording_id: str,
    max_calls: int,
    checkpoint: Callable[[ReasonerBehaviorRecording], None] | None = None,
) -> ReasonerBehaviorRecording:
    """Run a reasoner under a global cap and retain validated portable provenance."""

    capped = CallCappedReasoner(reasoner, max_calls=max_calls)
    recorded_cases: list[ReasonerBehaviorRecordingCase] = []
    if checkpoint is not None:
        checkpoint(
            _recording_snapshot(
                corpus, recording_id, capped, tuple(recorded_cases), complete=False
            )
        )
    for case in corpus.cases:
        recorder = InvocationRecordingReasoner(capped)
        evaluator = BehaviorFixtureEvaluator(case.observations)
        with suppress(ReasonerError, OperatorPolicyError):
            run_operator(
                objective=case.objective,
                authority=case.authority,
                reasoner=recorder,
                evaluator=evaluator,
            )
        recorded_cases.append(recording_case_from_capture(case.case_id, recorder))
        if checkpoint is not None:
            checkpoint(
                _recording_snapshot(
                    corpus, recording_id, capped, tuple(recorded_cases), complete=False
                )
            )
    recording = _recording_snapshot(
        corpus, recording_id, capped, tuple(recorded_cases), complete=True
    )
    if checkpoint is not None:
        checkpoint(recording)
    return recording


def _recording_snapshot(
    corpus: ReasonerBehaviorCorpus,
    recording_id: str,
    capped: CallCappedReasoner,
    cases: tuple[ReasonerBehaviorRecordingCase, ...],
    *,
    complete: bool,
) -> ReasonerBehaviorRecording:
    return ReasonerBehaviorRecording(
        schema_version="1.0",
        recording_id=recording_id,
        corpus_id=corpus.corpus_id,
        call_cap=capped.max_calls,
        calls_attempted=capped.calls_attempted,
        complete=complete,
        cases=cases,
    )
