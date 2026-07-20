from __future__ import annotations

import json
from hashlib import sha256

from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    CandidateProposal,
    CommandTerminalStatus,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
    OperatorState,
    OperatorStep,
)
from astro_operator.reasoner import invocation_digest, model_digest
from astro_operator.world_state import (
    assertion_digest,
    reduce_world_state,
    validate_conclusion_claims,
)


def action_digest(action: OperatorAction) -> str:
    payload = json.dumps(
        action.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _record_digest(record: object) -> str:
    if not hasattr(record, "model_dump"):
        raise OperatorPolicyError("command record is not serializable")
    payload = record.model_dump(mode="json", exclude={"record_sha256"})
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_command_record(
    step: OperatorStep, proposal: OperatorAction, authority: AuthorityGrant
) -> None:
    execution = step.action.command_execution
    record = step.command_execution_record
    result = step.command_result
    assert execution is not None and record is not None and result is not None
    assert proposal.command is not None
    prepared = record.prepared
    terminal = record.terminal
    if prepared.idempotency_key != execution.idempotency_key:
        raise OperatorPolicyError("command preparation idempotency key mismatch")
    if prepared.proposal_action_id != proposal.action_id:
        raise OperatorPolicyError("command preparation proposal mismatch")
    if prepared.proposal_sha256 != action_digest(proposal):
        raise OperatorPolicyError("command preparation proposal digest mismatch")
    if prepared.execution_action_sha256 != action_digest(step.action):
        raise OperatorPolicyError("command preparation execution digest mismatch")
    if prepared.command_sha256 != model_digest(proposal.command):
        raise OperatorPolicyError("command preparation command digest mismatch")
    if not prepared.simulation_only:
        raise OperatorPolicyError("operator journal contains a non-simulation command tool")
    envelope = next(
        (
            item
            for item in authority.command_envelopes
            if item.command_type == proposal.command.command_type
        ),
        None,
    )
    if envelope is None:
        raise OperatorPolicyError("command record has no authority qualification envelope")
    if (
        prepared.tool_version != envelope.tool_version
        or prepared.tool_qualification_sha256 != envelope.tool_qualification_sha256
        or prepared.simulation_only != envelope.simulation_only
    ):
        raise OperatorPolicyError("command record tool qualification receipt mismatch")
    if (
        prepared.grant_id != authority.grant_id
        or prepared.grant_version != authority.grant_version
        or prepared.authority_sha256 != model_digest(authority)
    ):
        raise OperatorPolicyError("command preparation authority receipt mismatch")
    if prepared.world_state_sha256 != execution.expected_world_state_sha256:
        raise OperatorPolicyError("command preparation world-state receipt mismatch")
    if prepared.approval_id != execution.approval_id:
        raise OperatorPolicyError("command preparation approval receipt mismatch")
    approval_required = (
        OperatorActionKind.EXECUTE_COMMAND in authority.approval_required_for
    )
    if approval_required:
        if execution.approval_id is None:
            raise OperatorPolicyError("command execution lacks its named approval")
        approval = next(
            (
                item
                for item in authority.approvals
                if item.approval_id == execution.approval_id
            ),
            None,
        )
        if (
            approval is None
            or approval.grant_version != authority.grant_version
            or approval.action_id != step.action.action_id
            or approval.action_sha256 != action_digest(step.action)
        ):
            raise OperatorPolicyError(
                "command execution named approval is not bound to the exact action"
            )
    elif execution.approval_id is not None:
        raise OperatorPolicyError("command execution names an approval not required by its grant")
    if prepared.record_sha256 != _record_digest(prepared):
        raise OperatorPolicyError("command preparation record digest mismatch")
    if (
        authority.valid_from is not None
        and prepared.prepared_at < authority.valid_from
    ):
        raise OperatorPolicyError("command preparation predates authority validity")
    if authority.expires_at is not None and prepared.prepared_at >= authority.expires_at:
        raise OperatorPolicyError("command preparation occurred after authority expiry")
    if terminal.record_sha256 != _record_digest(terminal):
        raise OperatorPolicyError("command terminal record digest mismatch")
    if authority.expires_at is not None and terminal.completed_at >= authority.expires_at:
        raise OperatorPolicyError("command commit occurred after authority expiry")
    if terminal.status != CommandTerminalStatus.COMMITTED:
        raise OperatorPolicyError("completed operator state requires a committed command")
    if terminal.result_sha256 != model_digest(result):
        raise OperatorPolicyError("command terminal result digest mismatch")
    if result.command != proposal.command:
        raise OperatorPolicyError("command result does not match its proposal")


def validate_action_against_grant(
    action: OperatorAction, authority: AuthorityGrant
) -> None:
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    if action.kind not in authority.allowed_actions:
        raise OperatorPolicyError(
            f"action {action.kind.value} is outside grant {authority.grant_id}"
        )
    if action.kind in authority.approval_required_for:
        digest = action_digest(action)
        if not any(
            approval.action_id == action.action_id
            and approval.grant_version == authority.grant_version
            and approval.action_sha256 == digest
            for approval in authority.approvals
        ):
            raise OperatorPolicyError(
                f"action {action.action_id} requires approval bound to its content under this grant"
            )
    if (
        action.command is not None
        and action.command.command_type not in authority.allowed_command_types
    ):
        raise OperatorPolicyError(
            f"command type {action.command.command_type} is outside grant {authority.grant_id}"
        )
    if action.evidence_request is not None:
        request = action.evidence_request
        allowed = next(
            (
                item
                for item in authority.allowed_evidence_tools
                if item.tool_id == request.tool_id
                and item.tool_version == request.tool_version
            ),
            None,
        )
        if allowed is None or request.request_kind not in allowed.request_kinds:
            raise OperatorPolicyError(
                f"evidence tool {request.tool_id}@{request.tool_version} "
                f"cannot serve {request.request_kind} under grant {authority.grant_id}"
            )


def validate_candidate_against_objective(
    candidate: CandidateProposal, objective: MissionObjective
) -> None:
    variables = {item.variable_id: item for item in objective.design_variables}
    unknown = set(candidate.assignments) - set(variables)
    if unknown:
        raise OperatorPolicyError(
            f"candidate uses unknown design variables: {', '.join(sorted(unknown))}"
        )
    for variable_id, value in candidate.assignments.items():
        variable = variables[variable_id]
        if not variable.lower_bound <= value <= variable.upper_bound:
            raise OperatorPolicyError(
                f"candidate {candidate.candidate_id} sets {variable_id} outside "
                f"[{variable.lower_bound}, {variable.upper_bound}] {variable.unit}"
            )


def validate_action_against_state(action: OperatorAction, state: OperatorState) -> None:
    """Validate one proposed action without executing it or calling a provider."""

    if state.remaining_steps == 0:
        raise OperatorPolicyError("operator step budget is exhausted")
    if action.action_id in {step.action.action_id for step in state.steps}:
        raise OperatorPolicyError(f"action ID {action.action_id} has already been used")
    validate_action_against_grant(action, state.authority)
    known_evidence_ids = {evidence.evidence_id for evidence in state.known_evidence}
    unknown_evidence = set(action.evidence_ids) - known_evidence_ids
    if unknown_evidence:
        raise OperatorPolicyError(
            f"action cites unknown evidence: {', '.join(sorted(unknown_evidence))}"
        )
    evaluated = {
        step.observation.candidate.candidate_id
        for step in state.steps
        if step.observation is not None
    }
    if action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
        if state.remaining_candidate_evaluations == 0:
            raise OperatorPolicyError("candidate evaluation budget exhausted")
        assert action.candidate is not None
        if action.candidate.candidate_id in evaluated:
            raise OperatorPolicyError(
                f"candidate {action.candidate.candidate_id} has already been evaluated"
            )
        validate_candidate_against_objective(action.candidate, state.objective)
    elif action.kind == OperatorActionKind.REQUEST_EVIDENCE:
        if state.remaining_evidence_acquisitions == 0:
            raise OperatorPolicyError("evidence acquisition budget exhausted")
        assert action.evidence_request is not None
        request_ids = {
            step.action.evidence_request.request_id
            for step in state.steps
            if step.action.evidence_request is not None
        }
        if action.evidence_request.request_id in request_ids:
            raise OperatorPolicyError(
                f"evidence request {action.evidence_request.request_id} has already been used"
            )
    elif action.kind == OperatorActionKind.PROPOSE_COMMAND:
        assert action.command is not None
        proposed = {
            step.action.command.command_id
            for step in state.steps
            if step.action.kind == OperatorActionKind.PROPOSE_COMMAND
            and step.action.command is not None
        }
        if action.command.command_id in proposed:
            raise OperatorPolicyError(
                f"command {action.command.command_id} has already been proposed"
            )
    elif action.kind == OperatorActionKind.EXECUTE_COMMAND:
        if action.command_execution is None:
            raise OperatorPolicyError(
                "legacy command payload does not support command commit; use the "
                "transactional execution request"
            )
        proposal = next(
            (
                step.action
                for step in state.steps
                if step.action.action_id == action.command_execution.proposal_action_id
                and step.action.kind == OperatorActionKind.PROPOSE_COMMAND
            ),
            None,
        )
        if proposal is None or proposal.command is None:
            raise OperatorPolicyError("command execution must reference a prior proposal")
        if proposal.command.command_id != action.command_execution.command_id:
            raise OperatorPolicyError("command execution does not match its proposal")
        if state.world_state is None:
            raise OperatorPolicyError("command execution requires a world-state snapshot")
        if (
            action.command_execution.expected_world_state_sha256
            != state.world_state.state_sha256
        ):
            raise OperatorPolicyError("command execution expected world state is stale")
    elif (
        action.kind == OperatorActionKind.FINISH
        and action.selected_candidate_id is not None
        and action.selected_candidate_id not in evaluated
    ):
        raise OperatorPolicyError("selected candidate must have been evaluated")


def validate_operator_state(state: OperatorState) -> None:
    """Reject states whose counters or evidence cannot be reconstructed from history."""

    if len(state.steps) > state.authority.max_steps:
        raise OperatorPolicyError("operator state exceeds its step budget")
    expected_remaining_steps = state.authority.max_steps - len(state.steps)
    if state.remaining_steps != expected_remaining_steps:
        raise OperatorPolicyError("operator state remaining step budget is inconsistent")
    evaluation_count = sum(
        step.action.kind == OperatorActionKind.EVALUATE_CANDIDATE for step in state.steps
    )
    if evaluation_count > state.authority.max_candidate_evaluations:
        raise OperatorPolicyError("operator state exceeds its candidate evaluation budget")
    if (
        state.remaining_candidate_evaluations
        != state.authority.max_candidate_evaluations - evaluation_count
    ):
        raise OperatorPolicyError("operator state remaining evaluation budget is inconsistent")
    acquisition_count = sum(
        step.action.kind == OperatorActionKind.REQUEST_EVIDENCE for step in state.steps
    )
    if acquisition_count > state.authority.max_evidence_acquisitions:
        raise OperatorPolicyError("operator state exceeds its evidence acquisition budget")
    if (
        state.remaining_evidence_acquisitions
        != state.authority.max_evidence_acquisitions - acquisition_count
    ):
        raise OperatorPolicyError(
            "operator state remaining evidence acquisition budget is inconsistent"
        )
    if [step.sequence for step in state.steps] != list(range(1, len(state.steps) + 1)):
        raise OperatorPolicyError("operator state step sequences must be contiguous and one-based")
    known = list(state.objective.base_evidence)
    known_ids = {evidence.evidence_id for evidence in known}
    assertions = list(state.objective.base_assertions)
    assertion_ids = {assertion.assertion_id for assertion in assertions}
    action_ids: set[str] = set()
    evaluated: set[str] = set()
    proposed_commands: set[str] = set()
    proposals_by_action_id: dict[str, OperatorAction] = {}
    idempotency_keys: set[str] = set()
    consumed_approvals: set[str] = set()
    request_ids: set[str] = set()
    for step in state.steps:
        if step.action.kind == OperatorActionKind.FINISH:
            raise OperatorPolicyError("operator state cannot continue after a finish action")
        if step.action.action_id in action_ids:
            raise OperatorPolicyError("operator state action IDs must be unique")
        action_ids.add(step.action.action_id)
        validate_action_against_grant(step.action, state.authority)
        if not set(step.action.evidence_ids).issubset(known_ids):
            raise OperatorPolicyError("operator state action cites unavailable evidence")
        if step.action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            assert step.action.candidate is not None
            assert step.observation is not None
            candidate_id = step.action.candidate.candidate_id
            if candidate_id in evaluated:
                raise OperatorPolicyError("operator state evaluates a candidate more than once")
            if step.observation.candidate != step.action.candidate:
                raise OperatorPolicyError("operator state observation does not match its action")
            validate_candidate_against_objective(step.action.candidate, state.objective)
            validate_observation_against_objective(step.observation, state.objective)
            evaluated.add(candidate_id)
        elif step.action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert step.action.command is not None
            command_id = step.action.command.command_id
            if command_id in proposed_commands:
                raise OperatorPolicyError("operator state proposes a command more than once")
            proposed_commands.add(command_id)
            proposals_by_action_id[step.action.action_id] = step.action
        elif step.action.kind == OperatorActionKind.REQUEST_EVIDENCE:
            assert step.action.evidence_request is not None
            request_id = step.action.evidence_request.request_id
            if request_id in request_ids:
                raise OperatorPolicyError("operator state evidence request IDs must be unique")
            request_ids.add(request_id)
            if step.acquisition_result is None:
                raise OperatorPolicyError("operator state lacks a typed acquisition result")
        elif step.action.kind == OperatorActionKind.EXECUTE_COMMAND:
            execution = step.action.command_execution
            if execution is None or step.command_execution_record is None:
                raise OperatorPolicyError("operator state has legacy command execution history")
            proposal = proposals_by_action_id.get(execution.proposal_action_id)
            if proposal is None or proposal.command is None:
                raise OperatorPolicyError("operator state execution lacks its proposal")
            if proposal.command.command_id != execution.command_id:
                raise OperatorPolicyError("operator state execution mismatches its proposal")
            current_world_state = reduce_world_state(tuple(assertions))
            if execution.expected_world_state_sha256 != current_world_state.state_sha256:
                raise OperatorPolicyError("operator state execution world-state receipt is stale")
            if execution.idempotency_key in idempotency_keys:
                raise OperatorPolicyError("operator state reuses a command idempotency key")
            idempotency_keys.add(execution.idempotency_key)
            if execution.approval_id is not None:
                if execution.approval_id in consumed_approvals:
                    raise OperatorPolicyError("operator state reuses a command approval")
                consumed_approvals.add(execution.approval_id)
            _validate_command_record(step, proposal, state.authority)
        additions = step.acquired_evidence
        if step.observation is not None:
            additions += step.observation.evidence
        if step.command_result is not None:
            additions += step.command_result.evidence
        if step.acquisition_result is not None:
            additions += step.acquisition_result.evidence
        for evidence in additions:
            if evidence.evidence_id in known_ids:
                raise OperatorPolicyError("operator state evidence IDs must be unique")
            known.append(evidence)
            known_ids.add(evidence.evidence_id)
        if step.acquisition_result is not None:
            for assertion in step.acquisition_result.assertions:
                if assertion.assertion_id in assertion_ids:
                    raise OperatorPolicyError("operator state assertion IDs must be unique")
                if not set(assertion.source_evidence_ids).issubset(known_ids):
                    raise OperatorPolicyError("operator state assertion cites unavailable evidence")
                assertions.append(assertion)
                assertion_ids.add(assertion.assertion_id)
        if step.command_result is not None:
            for assertion in step.command_result.assertions:
                if assertion.assertion_id in assertion_ids:
                    raise OperatorPolicyError("operator state assertion IDs must be unique")
                if not set(assertion.source_evidence_ids).issubset(known_ids):
                    raise OperatorPolicyError("operator state assertion cites unavailable evidence")
                assertions.append(assertion)
                assertion_ids.add(assertion.assertion_id)
    if tuple(known) != state.known_evidence:
        raise OperatorPolicyError("operator state evidence inventory is inconsistent")
    expected_world_state = reduce_world_state(tuple(assertions))
    if state.world_state is None:
        if assertions or any(step.acquisition_result is not None for step in state.steps):
            raise OperatorPolicyError("operator state is missing its world-state reduction")
    elif state.world_state != expected_world_state:
        raise OperatorPolicyError("operator state world-state reduction is inconsistent")


def validate_observation_against_objective(
    observation: CandidateObservation, objective: MissionObjective
) -> None:
    metrics = {item.metric_id: item for item in observation.metrics}
    if observation.evaluation_status == "evaluated":
        missing = {goal.metric_id for goal in objective.metric_goals} - set(metrics)
        if missing:
            raise OperatorPolicyError(
                f"evaluator omitted objective metrics: {', '.join(sorted(missing))}"
            )
    for goal in objective.metric_goals:
        metric = metrics.get(goal.metric_id)
        if metric is not None and metric.unit != goal.unit:
            raise OperatorPolicyError(
                f"evaluator metric {goal.metric_id} uses {metric.unit}, expected {goal.unit}"
            )


def validate_operator_run_policy(run: OperatorRun) -> None:
    authority = run.authority
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    if len(run.steps) > authority.max_steps:
        raise OperatorPolicyError("operator journal exceeds its step budget")

    evaluated: set[str] = set()
    proposed_commands: dict[str, object] = {}
    proposals_by_action_id: dict[str, OperatorAction] = {}
    idempotency_keys: set[str] = set()
    consumed_approvals: set[str] = set()
    evaluation_count = 0
    acquisition_count = 0
    known_evidence = list(run.objective.base_evidence)
    assertions = list(run.objective.base_assertions)
    if run.schema_version == "1.2" and any(
        item.assertion_sha256 != assertion_digest(item) for item in assertions
    ):
        raise OperatorPolicyError("operator journal base assertions are not digest-bound")
    world_state = reduce_world_state(tuple(assertions))
    prior_steps: list[OperatorStep] = []
    request_ids: set[str] = set()
    for step_index, step in enumerate(run.steps):
        action = step.action
        if run.schema_version in {"1.1", "1.2"}:
            invocation = step.reasoner_invocation
            assert invocation is not None
            state = OperatorState(
                objective=run.objective,
                authority=run.authority,
                steps=tuple(prior_steps),
                known_evidence=tuple(known_evidence),
                world_state=(world_state if run.schema_version == "1.2" else None),
                remaining_steps=authority.max_steps - step_index,
                remaining_candidate_evaluations=(
                    authority.max_candidate_evaluations - evaluation_count
                ),
                remaining_evidence_acquisitions=(
                    authority.max_evidence_acquisitions - acquisition_count
                ),
            )
            if invocation.input_sha256 != model_digest(state):
                raise OperatorPolicyError(
                    "reasoner invocation input digest does not match journal state"
                )
            if invocation.output_sha256 != model_digest(action):
                raise OperatorPolicyError(
                    "reasoner invocation output digest does not match journal action"
                )
            if (
                invocation.record_sha256 is None
                or invocation.record_sha256 != invocation_digest(invocation)
            ):
                raise OperatorPolicyError(
                    "reasoner invocation record digest does not match journal provenance"
                )
            validate_action_against_state(action, state)
        validate_action_against_grant(action, authority)
        if action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            assert action.candidate is not None
            if action.candidate.candidate_id in evaluated:
                raise OperatorPolicyError("operator journal evaluates a candidate more than once")
            validate_candidate_against_objective(action.candidate, run.objective)
            assert step.observation is not None
            validate_observation_against_objective(step.observation, run.objective)
            evaluated.add(action.candidate.candidate_id)
            evaluation_count += 1
        elif action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert action.command is not None
            if action.command.command_id in proposed_commands:
                raise OperatorPolicyError("operator journal proposes a command more than once")
            proposed_commands[action.command.command_id] = action.command
            proposals_by_action_id[action.action_id] = action
        elif action.kind == OperatorActionKind.REQUEST_EVIDENCE:
            assert action.evidence_request is not None
            request_id = action.evidence_request.request_id
            if request_id in request_ids:
                raise OperatorPolicyError("operator journal reuses an evidence request ID")
            request_ids.add(request_id)
            if step.acquisition_result is None:
                raise OperatorPolicyError("operator journal lacks a typed acquisition result")
            acquisition_count += 1
        elif action.kind == OperatorActionKind.EXECUTE_COMMAND:
            if run.schema_version != "1.2":
                raise OperatorPolicyError("legacy operator schemas cannot commit commands")
            execution = action.command_execution
            if execution is None or step.command_execution_record is None:
                raise OperatorPolicyError("operator journal has legacy command execution history")
            proposal = proposals_by_action_id.get(execution.proposal_action_id)
            if proposal is None or proposal.command is None:
                raise OperatorPolicyError("operator journal execution lacks its proposal")
            if proposal.command.command_id != execution.command_id:
                raise OperatorPolicyError("operator journal execution mismatches its proposal")
            if execution.idempotency_key in idempotency_keys:
                raise OperatorPolicyError("operator journal reuses a command idempotency key")
            idempotency_keys.add(execution.idempotency_key)
            if execution.approval_id is not None:
                if execution.approval_id in consumed_approvals:
                    raise OperatorPolicyError("operator journal reuses a command approval")
                consumed_approvals.add(execution.approval_id)
            _validate_command_record(step, proposal, authority)
        additions = step.acquired_evidence
        if step.observation is not None:
            additions += step.observation.evidence
        if step.command_result is not None:
            if run.schema_version == "1.2" and any(
                item.assertion_sha256 != assertion_digest(item)
                for item in step.command_result.assertions
            ):
                raise OperatorPolicyError(
                    "operator journal command assertions are not digest-bound"
                )
            additions += step.command_result.evidence
        if step.acquisition_result is not None:
            if run.schema_version == "1.2" and any(
                item.assertion_sha256 != assertion_digest(item)
                for item in step.acquisition_result.assertions
            ):
                raise OperatorPolicyError(
                    "operator journal acquired assertions are not digest-bound"
                )
            additions += step.acquisition_result.evidence
        known_evidence.extend(additions)
        if step.acquisition_result is not None:
            known_ids = {item.evidence_id for item in known_evidence}
            known_assertion_ids = {item.assertion_id for item in assertions}
            for assertion in step.acquisition_result.assertions:
                if assertion.assertion_id in known_assertion_ids:
                    raise OperatorPolicyError("operator journal assertion IDs must be unique")
                if not set(assertion.source_evidence_ids).issubset(known_ids):
                    raise OperatorPolicyError(
                        "operator journal assertion cites unavailable evidence"
                    )
                assertions.append(assertion)
                known_assertion_ids.add(assertion.assertion_id)
            world_state = reduce_world_state(tuple(assertions))
        if step.command_result is not None:
            known_ids = {item.evidence_id for item in known_evidence}
            known_assertion_ids = {item.assertion_id for item in assertions}
            for assertion in step.command_result.assertions:
                if assertion.assertion_id in known_assertion_ids:
                    raise OperatorPolicyError("operator journal assertion IDs must be unique")
                if not set(assertion.source_evidence_ids).issubset(known_ids):
                    raise OperatorPolicyError(
                        "operator journal assertion cites unavailable evidence"
                    )
                assertions.append(assertion)
                known_assertion_ids.add(assertion.assertion_id)
            world_state = reduce_world_state(tuple(assertions))
        prior_steps.append(step)

    if evaluation_count > authority.max_candidate_evaluations:
        raise OperatorPolicyError("operator journal exceeds its candidate evaluation budget")
    if acquisition_count > authority.max_evidence_acquisitions:
        raise OperatorPolicyError("operator journal exceeds its evidence acquisition budget")
    if run.schema_version == "1.2":
        if run.world_state != world_state:
            raise OperatorPolicyError("operator journal world state does not match replay")
        if run.status == OperatorRunStatus.COMPLETED:
            try:
                validate_conclusion_claims(
                    run.steps[-1].action.conclusion_claims, world_state
                )
            except ValueError as exc:
                raise OperatorPolicyError(f"invalid conclusion claims: {exc}") from exc
    if run.status == OperatorRunStatus.BUDGET_EXHAUSTED and len(run.steps) != authority.max_steps:
        raise OperatorPolicyError("budget-exhausted journal does not consume its step budget")
