from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, Protocol, TypeVar

from astro_core.errors import InvalidScenarioError, NumericalConvergenceError
from astro_core.models import AstroModel
from astro_uq.models import (
    EvaluationOutcome,
    EvaluationTiming,
    OutcomeStatus,
    ScenarioRealization,
)
from astro_uq.parameters import ParameterBindingError

ScenarioT = TypeVar("ScenarioT", bound=AstroModel)
ScenarioT_contra = TypeVar("ScenarioT_contra", bound=AstroModel, contravariant=True)
ResultT = TypeVar("ResultT", bound=AstroModel)


class EvaluationError(Exception):
    """Base class for failures with an explicit campaign outcome meaning."""


class InvalidRealizationError(EvaluationError):
    """The resolved scenario is invalid for authoritative evaluation."""


class PolicyRejectionError(EvaluationError):
    """Evaluation is disallowed by an explicit workflow or execution policy."""


class NumericalEvaluationError(EvaluationError):
    """The evaluator failed because its numerical method was not usable."""


class ExecutionEvaluationError(EvaluationError):
    """The evaluator or artifact serializer could not execute."""


class Evaluator(Protocol[ScenarioT_contra, ResultT]):
    """Typed boundary between a resolved scenario and a suite-owned result."""

    @property
    def evaluator_id(self) -> str: ...

    def setup(self) -> None: ...

    def evaluate(self, scenario: ScenarioT_contra) -> ResultT: ...

    def serialize(self, result: ResultT) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class AuthoritativeCallableEvaluator(Generic[ScenarioT, ResultT]):  # noqa: UP046
    """Authoritative evaluator assembled from allow-listed suite callables."""

    evaluator_id: str
    evaluate_callable: Callable[[ScenarioT], ResultT]
    serialize_callable: Callable[[ResultT], tuple[str, ...]]
    setup_callable: Callable[[], None] = lambda: None

    def setup(self) -> None:
        self.setup_callable()

    def evaluate(self, scenario: ScenarioT) -> ResultT:
        return self.evaluate_callable(scenario)

    def serialize(self, result: ResultT) -> tuple[str, ...]:
        return self.serialize_callable(result)


Clock = Callable[[], float]


def evaluate_authoritatively(  # noqa: UP047
    evaluator: Evaluator[ScenarioT, ResultT],
    scenario: ScenarioT,
    realization: ScenarioRealization,
    *,
    clock: Clock = perf_counter,
    serialize_result: bool = True,
) -> tuple[EvaluationOutcome, ResultT | None]:
    """Run one sample and produce exactly one outcome, including on failure."""
    started = clock()
    setup_s = 0.0
    evaluation_s = 0.0
    serialization_s = 0.0
    phase = "realization"

    if not realization.valid:
        message = "; ".join(realization.validation_errors)
        outcome = _failure_outcome(
            evaluator_id=evaluator.evaluator_id,
            realization=realization,
            status=OutcomeStatus.INVALID_REALIZATION,
            error_type="InvalidRealization",
            error_message=message,
            phase=phase,
            timing=_timing(started, setup_s, evaluation_s, serialization_s, clock),
            traceback_summary=(),
        )
        return outcome, None

    try:
        phase = "setup"
        phase_started = clock()
        try:
            evaluator.setup()
        finally:
            setup_s = _elapsed(phase_started, clock())

        phase = "evaluation"
        phase_started = clock()
        try:
            result = evaluator.evaluate(scenario)
        finally:
            evaluation_s = _elapsed(phase_started, clock())

        artifact_refs: tuple[str, ...] = ()
        if serialize_result:
            phase = "serialization"
            phase_started = clock()
            try:
                artifact_refs = evaluator.serialize(result)
            finally:
                serialization_s = _elapsed(phase_started, clock())
    except Exception as exc:
        status = _status_for_exception(exc)
        exception_metadata: dict[str, object] = {}
        workflow_phase = getattr(exc, "lifecycle_phase", None)
        if isinstance(workflow_phase, str) and workflow_phase:
            exception_metadata["workflow_phase"] = workflow_phase
        outcome = _failure_outcome(
            evaluator_id=evaluator.evaluator_id,
            realization=realization,
            status=status,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
            phase=phase,
            timing=_timing(started, setup_s, evaluation_s, serialization_s, clock),
            traceback_summary=tuple(traceback.format_exception(exc)),
            extra_metadata=exception_metadata,
        )
        return outcome, None

    timing = _timing(started, setup_s, evaluation_s, serialization_s, clock)
    return (
        EvaluationOutcome(
            sample_id=realization.sample_id,
            evaluator_id=evaluator.evaluator_id,
            status=OutcomeStatus.SUCCESS,
            timing=timing,
            artifact_refs=artifact_refs,
            metadata={"resolved_scenario_digest": realization.resolved_scenario_digest},
        ),
        result,
    )


def _status_for_exception(exc: Exception) -> OutcomeStatus:
    if isinstance(exc, (InvalidRealizationError, InvalidScenarioError, ParameterBindingError)):
        return OutcomeStatus.INVALID_REALIZATION
    if isinstance(exc, PolicyRejectionError):
        return OutcomeStatus.POLICY_REJECTION
    if isinstance(
        exc,
        (
            NumericalEvaluationError,
            NumericalConvergenceError,
            ArithmeticError,
            FloatingPointError,
        ),
    ):
        return OutcomeStatus.NUMERICAL_FAILURE
    return OutcomeStatus.EXECUTION_FAILURE


def _elapsed(started: float, finished: float) -> float:
    return max(0.0, finished - started)


def _timing(
    started: float,
    setup_s: float,
    evaluation_s: float,
    serialization_s: float,
    clock: Clock,
) -> EvaluationTiming:
    components = setup_s + evaluation_s + serialization_s
    total_s = max(components, _elapsed(started, clock()))
    return EvaluationTiming(
        setup_s=setup_s,
        evaluation_s=evaluation_s,
        serialization_s=serialization_s,
        total_s=total_s,
    )


def _failure_outcome(
    *,
    evaluator_id: str,
    realization: ScenarioRealization,
    status: OutcomeStatus,
    error_type: str,
    error_message: str,
    phase: str,
    timing: EvaluationTiming,
    traceback_summary: tuple[str, ...],
    extra_metadata: dict[str, object] | None = None,
) -> EvaluationOutcome:
    return EvaluationOutcome(
        sample_id=realization.sample_id,
        evaluator_id=evaluator_id,
        status=status,
        timing=timing,
        error_type=error_type,
        error_message=error_message,
        metadata={
            "phase": phase,
            "sample_id": realization.sample_id,
            "resolved_scenario_digest": realization.resolved_scenario_digest,
            "validation_errors": list(realization.validation_errors),
            "traceback_summary": list(traceback_summary),
            **(extra_metadata or {}),
        },
    )
