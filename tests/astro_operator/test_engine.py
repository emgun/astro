from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from astro_operator.engine import run_operator
from astro_operator.errors import OperatorPolicyError, ReasonerInvalidResponseError
from astro_operator.models import (
    ActionApproval,
    AuthorityGrant,
    AuthorityLevel,
    CandidateObservation,
    CandidateProposal,
    CommandRequest,
    DesignVariable,
    MetricGoal,
    MissionObjective,
    ObservedMetric,
    OperatorAction,
    OperatorActionKind,
    OperatorRunStatus,
    ReasonerDecision,
    ReasonerInvocation,
)
from astro_operator.policy import action_digest
from astro_operator.reasoner import (
    ConditionalReplayDecision,
    ConditionalReplayReasoner,
    ReplayCondition,
    ScriptedReasoner,
    invocation_digest,
    model_digest,
)


def _objective() -> MissionObjective:
    return MissionObjective(
        objective_id="test-objective",
        summary="Exercise the adaptive operator kernel.",
        design_variables=(
            DesignVariable(
                variable_id="mass",
                target="spacecraft_wet_mass_kg",
                lower_bound=400.0,
                upper_bound=600.0,
                unit="kg",
            ),
        ),
        metric_goals=(
            MetricGoal(metric_id="reserve", objective="maximize", unit="kg"),
        ),
    )


def _authority(**updates: object) -> AuthorityGrant:
    payload: dict[str, object] = {
        "grant_id": "research-grant",
        "level": AuthorityLevel.RESEARCH,
        "mission_scope": "unit test",
        "allowed_actions": (
            OperatorActionKind.EVALUATE_CANDIDATE,
            OperatorActionKind.FINISH,
        ),
        "max_steps": 3,
        "max_candidate_evaluations": 1,
    }
    payload.update(updates)
    return AuthorityGrant.model_validate(payload)


class _Evaluator:
    def __init__(self) -> None:
        self.candidates: list[CandidateProposal] = []

    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation:
        self.candidates.append(candidate)
        return CandidateObservation(
            candidate=candidate,
            evaluation_status="evaluated",
            passed=True,
            metrics=(ObservedMetric(metric_id="reserve", value=12.0, unit="kg"),),
        )


def _decision(action: OperatorAction, state: object) -> ReasonerDecision:
    from astro_operator.models import OperatorState

    assert isinstance(state, OperatorState)
    invocation = ReasonerInvocation(
        adapter="unit-test",
        provider="deterministic",
        model="fixture",
        input_sha256=model_digest(state),
        output_sha256=model_digest(action),
    )
    invocation = invocation.model_copy(
        update={"record_sha256": invocation_digest(invocation)}
    )
    return ReasonerDecision(action=action, invocation=invocation)


def _run_actions(actions: Sequence[OperatorAction], authority: AuthorityGrant) -> object:
    return run_operator(
        objective=_objective(),
        authority=authority,
        reasoner=ScriptedReasoner(actions),
        evaluator=_Evaluator(),
    )


def test_reasoner_adapts_through_typed_observation_and_finishes() -> None:
    candidate = CandidateProposal(candidate_id="candidate-a", assignments={"mass": 510.0})
    actions = (
        OperatorAction(
            action_id="evaluate-a",
            kind=OperatorActionKind.EVALUATE_CANDIDATE,
            rationale="Evaluate a bounded candidate.",
            candidate=candidate,
        ),
        OperatorAction(
            action_id="finish",
            kind=OperatorActionKind.FINISH,
            rationale="Use the observed passing candidate.",
            selected_candidate_id="candidate-a",
            conclusion="Candidate A passed.",
        ),
    )

    run = _run_actions(actions, _authority())

    assert run.status == OperatorRunStatus.COMPLETED
    assert run.selected_candidate_id == "candidate-a"
    assert run.steps[0].observation is not None
    assert run.steps[0].observation.metrics[0].value == 12.0


def test_conditional_replay_skips_failure_branch_when_candidate_passes() -> None:
    candidate_a = CandidateProposal(candidate_id="candidate-a", assignments={"mass": 510.0})
    candidate_b = CandidateProposal(candidate_id="candidate-b", assignments={"mass": 520.0})
    decisions = (
        ConditionalReplayDecision(
            when=ReplayCondition(step_count=0),
            action=OperatorAction(
                action_id="evaluate-a",
                kind=OperatorActionKind.EVALUATE_CANDIDATE,
                rationale="Evaluate the first branch.",
                candidate=candidate_a,
            ),
        ),
        ConditionalReplayDecision(
            when=ReplayCondition(
                last_candidate_id="candidate-a",
                last_evaluation_status="evaluated",
                last_candidate_passed=True,
            ),
            action=OperatorAction(
                action_id="finish-a",
                kind=OperatorActionKind.FINISH,
                rationale="Finish because candidate A passed.",
                selected_candidate_id="candidate-a",
                conclusion="Candidate A passed.",
            ),
        ),
        ConditionalReplayDecision(
            when=ReplayCondition(
                last_candidate_id="candidate-a",
                last_evaluation_status="evaluation_failed",
                last_candidate_passed=False,
            ),
            action=OperatorAction(
                action_id="evaluate-b",
                kind=OperatorActionKind.EVALUATE_CANDIDATE,
                rationale="Use the recovery branch.",
                candidate=candidate_b,
            ),
        ),
    )
    evaluator = _Evaluator()

    run = run_operator(
        objective=_objective(),
        authority=_authority(),
        reasoner=ConditionalReplayReasoner(decisions),
        evaluator=evaluator,
    )

    assert run.selected_candidate_id == "candidate-a"
    assert evaluator.candidates == [candidate_a]


def test_candidate_outside_declared_envelope_is_rejected_before_evaluation() -> None:
    action = OperatorAction(
        action_id="escape",
        kind=OperatorActionKind.EVALUATE_CANDIDATE,
        rationale="Attempt an out-of-envelope value.",
        candidate=CandidateProposal(candidate_id="escape", assignments={"mass": 601.0}),
    )

    with pytest.raises(OperatorPolicyError, match="outside"):
        _run_actions((action,), _authority())


def test_research_grant_cannot_include_command_execution() -> None:
    with pytest.raises(ValidationError, match="cannot grant execute_command"):
        _authority(
            allowed_actions=(OperatorActionKind.EXECUTE_COMMAND,),
            allowed_command_types=("burn",),
        )


def test_supervised_command_requires_action_approval() -> None:
    command = CommandRequest(command_id="burn-1", command_type="burn")
    actions = (
        OperatorAction(
            action_id="propose",
            kind=OperatorActionKind.PROPOSE_COMMAND,
            rationale="Propose the bounded command.",
            command=command,
        ),
        OperatorAction(
            action_id="execute",
            kind=OperatorActionKind.EXECUTE_COMMAND,
            rationale="Execute the approved proposal.",
            command=command,
        ),
    )
    authority = _authority(
        level=AuthorityLevel.SUPERVISED_AUTONOMY,
        allowed_actions=(
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
            OperatorActionKind.FINISH,
        ),
        allowed_command_types=("burn",),
        approval_required_for=(OperatorActionKind.EXECUTE_COMMAND,),
        max_candidate_evaluations=0,
    )

    with pytest.raises(OperatorPolicyError, match="requires approval"):
        _run_actions(actions, authority)


def test_approval_requirement_applies_to_non_command_actions_too() -> None:
    finish = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Attempt an unapproved conclusion.",
        conclusion="Not approved.",
    )
    authority = _authority(
        allowed_actions=(OperatorActionKind.FINISH,),
        approval_required_for=(OperatorActionKind.FINISH,),
        max_candidate_evaluations=0,
    )

    with pytest.raises(OperatorPolicyError, match="requires approval"):
        _run_actions((finish,), authority)


def test_approval_is_bound_to_the_exact_action_content() -> None:
    finish = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Approved conclusion.",
        conclusion="Approved.",
    )
    approval = ActionApproval(
        approval_id="approval-1",
        grant_version=1,
        action_id=finish.action_id,
        action_sha256=action_digest(finish),
    )
    authority = _authority(
        allowed_actions=(OperatorActionKind.FINISH,),
        approval_required_for=(OperatorActionKind.FINISH,),
        approvals=(approval,),
        max_candidate_evaluations=0,
    )

    assert _run_actions((finish,), authority).status == OperatorRunStatus.COMPLETED
    changed = finish.model_copy(update={"conclusion": "Changed after approval."})
    with pytest.raises(OperatorPolicyError, match="requires approval"):
        _run_actions((changed,), authority)


def test_delegated_grant_stages_execution_until_commit_protocol_exists() -> None:
    command = CommandRequest(
        command_id="burn-1",
        command_type="burn",
        parameters={"delta_v_m_s": 0.2},
    )
    actions = (
        OperatorAction(
            action_id="propose",
            kind=OperatorActionKind.PROPOSE_COMMAND,
            rationale="Propose a command inside the grant.",
            command=command,
        ),
        OperatorAction(
            action_id="execute",
            kind=OperatorActionKind.EXECUTE_COMMAND,
            rationale="Execute the exact proposal.",
            command=command,
        ),
        OperatorAction(
            action_id="finish",
            kind=OperatorActionKind.FINISH,
            rationale="The delegated task is complete.",
            conclusion="The scoped command was executed.",
        ),
    )
    authority = _authority(
        level=AuthorityLevel.DELEGATED_AUTONOMY,
        allowed_actions=(
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
            OperatorActionKind.FINISH,
        ),
        allowed_command_types=("burn",),
        max_candidate_evaluations=0,
    )
    with pytest.raises(OperatorPolicyError, match="does not support command commit"):
        run_operator(
            objective=_objective(),
            authority=authority,
            reasoner=ScriptedReasoner(actions),
            evaluator=_Evaluator(),
        )


def test_revoked_grant_stops_before_reasoner_or_tools_run() -> None:
    with pytest.raises(OperatorPolicyError, match="revoked"):
        _run_actions((), _authority(revoked=True))


@pytest.mark.parametrize(
    "update, message",
    (
        ({"expires_at": datetime.now(UTC) - timedelta(seconds=1)}, "expired"),
        ({"valid_from": datetime.now(UTC) + timedelta(hours=1)}, "not yet valid"),
    ),
)
def test_authority_validity_window_applies_to_every_operator_action(
    update: dict[str, datetime], message: str
) -> None:
    finish = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Must not run outside the authority window.",
        conclusion="Blocked.",
    )
    with pytest.raises(OperatorPolicyError, match=message):
        _run_actions((finish,), _authority(max_candidate_evaluations=0, **update))


def test_run_records_budget_exhaustion_without_inventing_a_conclusion() -> None:
    candidate = CandidateProposal(candidate_id="candidate-a", assignments={"mass": 510.0})
    run = _run_actions(
        (
            OperatorAction(
                action_id="evaluate-a",
                kind=OperatorActionKind.EVALUATE_CANDIDATE,
                rationale="Use the only available step.",
                candidate=candidate,
            ),
        ),
        _authority(max_steps=1),
    )

    assert run.status == OperatorRunStatus.BUDGET_EXHAUSTED
    assert run.selected_candidate_id is None
    assert "step budget" in run.conclusion


def test_authority_monitor_can_revoke_between_adaptive_calls() -> None:
    class _Monitor:
        def __init__(self) -> None:
            self.checks = 0

        def check(self, authority: AuthorityGrant) -> None:
            del authority
            self.checks += 1
            if self.checks >= 4:
                raise OperatorPolicyError("authority revoked by monitor")

    candidate = CandidateProposal(candidate_id="candidate-a", assignments={"mass": 510.0})
    actions = (
        OperatorAction(
            action_id="evaluate-a",
            kind=OperatorActionKind.EVALUATE_CANDIDATE,
            rationale="Evaluate before external revocation.",
            candidate=candidate,
        ),
        OperatorAction(
            action_id="finish",
            kind=OperatorActionKind.FINISH,
            rationale="This action must not be accepted after revocation.",
            conclusion="Should not complete.",
        ),
    )

    with pytest.raises(OperatorPolicyError, match="revoked by monitor"):
        run_operator(
            objective=_objective(),
            authority=_authority(),
            reasoner=ScriptedReasoner(actions),
            evaluator=_Evaluator(),
            authority_monitor=_Monitor(),
        )


def test_reasoner_cannot_mutate_the_policy_objective_to_escape_bounds() -> None:
    class _MutatingReasoner:
        def decide(self, state: object) -> ReasonerDecision:
            from astro_operator.models import OperatorState

            assert isinstance(state, OperatorState)
            state.objective.design_variables[0].upper_bound = 1_000.0
            return _decision(
                OperatorAction(
                    action_id="escape",
                    kind=OperatorActionKind.EVALUATE_CANDIDATE,
                    rationale="Attempt to widen the copied provider context.",
                    candidate=CandidateProposal(
                        candidate_id="escape",
                        assignments={"mass": 999.0},
                    ),
                ),
                state,
            )

    with pytest.raises(ReasonerInvalidResponseError, match="input digest"):
        run_operator(
            objective=_objective(),
            authority=_authority(),
            reasoner=_MutatingReasoner(),
            evaluator=_Evaluator(),
        )


def test_reasoner_cannot_retroactively_mutate_an_accepted_action() -> None:
    class _RetainingReasoner:
        def __init__(self) -> None:
            self.proposal = OperatorAction(
                action_id="proposal",
                kind=OperatorActionKind.PROPOSE_COMMAND,
                rationale="Original rationale.",
                command=CommandRequest(command_id="burn-1", command_type="burn"),
            )

        def decide(self, state: object) -> ReasonerDecision:
            from astro_operator.models import OperatorState

            assert isinstance(state, OperatorState)
            if not state.steps:
                return _decision(self.proposal, state)
            self.proposal.rationale = "Mutated after acceptance."
            return _decision(
                OperatorAction(
                    action_id="finish",
                    kind=OperatorActionKind.FINISH,
                    rationale="Finish after proposal.",
                    conclusion="Proposal recorded.",
                ),
                state,
            )

    run = run_operator(
        objective=_objective(),
        authority=_authority(
            level=AuthorityLevel.DECISION_SUPPORT,
            allowed_actions=(
                OperatorActionKind.PROPOSE_COMMAND,
                OperatorActionKind.FINISH,
            ),
            allowed_command_types=("burn",),
            max_candidate_evaluations=0,
        ),
        reasoner=_RetainingReasoner(),
        evaluator=_Evaluator(),
    )

    assert run.steps[0].action.rationale == "Original rationale."


def test_reasoner_decision_provenance_is_digest_bound_and_journaled() -> None:
    action = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Finish deterministically.",
        conclusion="No candidate required.",
    )

    run = _run_actions((action,), _authority(max_candidate_evaluations=0))

    invocation = run.steps[0].reasoner_invocation
    assert invocation is not None
    assert invocation.provider == "deterministic-replay"
    assert invocation.output_sha256 == model_digest(action)

    legacy_payload = run.model_dump(mode="json")
    legacy_payload["schema_version"] = "1.0"
    for step in legacy_payload["steps"]:
        step.pop("reasoner_invocation")
    legacy = type(run).model_validate(legacy_payload)
    assert legacy.schema_version == "1.0"
    assert legacy.steps[0].reasoner_invocation is None


@pytest.mark.parametrize("field", ("input_sha256", "output_sha256"))
def test_engine_rejects_forged_reasoner_provenance(field: str) -> None:
    class _ForgedReasoner:
        def decide(self, state: object) -> ReasonerDecision:
            action = OperatorAction(
                action_id="finish",
                kind=OperatorActionKind.FINISH,
                rationale="Return forged provenance.",
                conclusion="Must not be accepted.",
            )
            decision = _decision(action, state)
            setattr(decision.invocation, field, "f" * 64)
            return decision

    with pytest.raises(ReasonerInvalidResponseError, match="digest"):
        run_operator(
            objective=_objective(),
            authority=_authority(max_candidate_evaluations=0),
            reasoner=_ForgedReasoner(),
            evaluator=_Evaluator(),
        )


def test_engine_rejects_unenveloped_reasoner_output() -> None:
    class _BareActionReasoner:
        def decide(self, state: object) -> OperatorAction:
            del state
            return OperatorAction(
                action_id="finish",
                kind=OperatorActionKind.FINISH,
                rationale="Return a bare action.",
                conclusion="Must not be accepted.",
            )

    with pytest.raises(ReasonerInvalidResponseError, match="ReasonerDecision"):
        run_operator(
            objective=_objective(),
            authority=_authority(max_candidate_evaluations=0),
            reasoner=_BareActionReasoner(),  # type: ignore[arg-type]
            evaluator=_Evaluator(),
        )


def test_engine_normalizes_mutated_decision_model() -> None:
    class _MutatedDecisionReasoner:
        def decide(self, state: object) -> ReasonerDecision:
            action = OperatorAction(
                action_id="finish",
                kind=OperatorActionKind.FINISH,
                rationale="Construct then corrupt a decision.",
                conclusion="Must not be accepted.",
            )
            decision = _decision(action, state)
            return decision.model_copy(update={"invocation": "not-an-invocation"})

    with pytest.raises(ReasonerInvalidResponseError, match="invalid ReasonerDecision"):
        run_operator(
            objective=_objective(),
            authority=_authority(max_candidate_evaluations=0),
            reasoner=_MutatedDecisionReasoner(),
            evaluator=_Evaluator(),
        )


def test_reasoner_invocation_rejects_unsafe_metadata_and_timestamps() -> None:
    payload = {
        "adapter": "test",
        "provider": "test",
        "model": "test",
        "input_sha256": "0" * 64,
        "output_sha256": "1" * 64,
    }
    with pytest.raises(ValidationError, match="timezone"):
        ReasonerInvocation.model_validate(
            {**payload, "started_at": datetime(2026, 7, 17, 12, 0)}
        )
    with pytest.raises(ValidationError):
        ReasonerInvocation.model_validate({**payload, "metadata": {"unsafe": object()}})
    with pytest.raises(ValidationError, match="16384"):
        ReasonerInvocation.model_validate(
            {**payload, "metadata": {"oversized": "x" * 16_385}}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_steps", True), ("max_candidate_evaluations", "1")),
)
def test_authority_integer_fields_reject_coercive_inputs(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="must be integers"):
        _authority(**{field: value})


def test_candidate_assignments_reject_boolean_values() -> None:
    with pytest.raises(ValidationError, match="numeric scalars"):
        CandidateProposal(candidate_id="invalid", assignments={"mass": True})
