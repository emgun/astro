from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astro_operator.command_execution import (
    CommandExecutionCoordinator,
    CommandToolRegistry,
    SimulatedBurnTool,
    SQLiteCommandExecutionStore,
)
from astro_operator.engine import run_operator
from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    ActionApproval,
    AuthorityGrant,
    AuthorityLevel,
    CandidateObservation,
    CandidateProposal,
    CommandEnvelope,
    CommandExecutionRequest,
    CommandParameterLimit,
    CommandRequest,
    DesignVariable,
    MetricGoal,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
)
from astro_operator.policy import action_digest, validate_operator_run_policy
from astro_operator.reasoner import ScriptedReasoner
from astro_operator.world_state import reduce_world_state


class _UnusedEvaluator:
    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation:
        raise AssertionError(f"unexpected candidate evaluation: {candidate.candidate_id}")


def _objective() -> MissionObjective:
    return MissionObjective(
        objective_id="simulated-command-test",
        summary="Exercise supervised simulated command execution.",
        design_variables=(
            DesignVariable(
                variable_id="mass",
                target="spacecraft_wet_mass_kg",
                lower_bound=1.0,
                upper_bound=2.0,
                unit="kg",
            ),
        ),
        metric_goals=(MetricGoal(metric_id="margin", objective="maximize", unit="kg"),),
    )


def _actions() -> tuple[OperatorAction, OperatorAction, OperatorAction]:
    world = reduce_world_state(())
    proposal = OperatorAction(
        action_id="propose-burn",
        kind=OperatorActionKind.PROPOSE_COMMAND,
        rationale="Propose a simulation-only burn.",
        command=CommandRequest(
            command_id="burn-1",
            command_type="simulated_burn",
            asset_id="sat-1",
            parameters={"delta_v_m_s": 0.5, "duration_s": 5.0, "frame": "TNW"},
        ),
    )
    execute = OperatorAction(
        action_id="execute-burn",
        kind=OperatorActionKind.EXECUTE_COMMAND,
        rationale="Execute the exact supervised simulation proposal.",
        command_execution=CommandExecutionRequest(
            proposal_action_id=proposal.action_id,
            command_id="burn-1",
            idempotency_key="burn-1-attempt-1",
            expected_world_state_sha256=world.state_sha256,
            approval_id="approve-burn-1",
        ),
    )
    finish = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="The simulated command committed.",
        conclusion="The supervised simulation-only burn committed exactly once.",
    )
    return proposal, execute, finish


def _authority(execute: OperatorAction) -> AuthorityGrant:
    return AuthorityGrant(
        grant_id="supervised-simulation",
        level=AuthorityLevel.SUPERVISED_AUTONOMY,
        mission_scope="sat-1 simulation only",
        allowed_actions=(
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
            OperatorActionKind.FINISH,
        ),
        allowed_command_types=("simulated_burn",),
        command_envelopes=(
            CommandEnvelope(
                command_type="simulated_burn",
                tool_version=SimulatedBurnTool.version,
                tool_qualification_sha256=SimulatedBurnTool.qualification_sha256,
                allowed_asset_ids=("sat-1",),
                parameter_limits=(
                    CommandParameterLimit(
                        parameter="delta_v_m_s", minimum=0.0, maximum=1.0, unit="m/s"
                    ),
                    CommandParameterLimit(
                        parameter="duration_s", minimum=1.0, maximum=10.0, unit="s"
                    ),
                ),
                max_commits=1,
            ),
        ),
        approval_required_for=(OperatorActionKind.EXECUTE_COMMAND,),
        approvals=(
            ActionApproval(
                approval_id="approve-burn-1",
                grant_version=1,
                action_id=execute.action_id,
                action_sha256=action_digest(execute),
            ),
        ),
        max_steps=3,
        max_candidate_evaluations=0,
    )


def test_operator_commits_one_supervised_simulated_command_end_to_end(
    tmp_path: Path,
) -> None:
    proposal, execute, finish = _actions()
    authority = _authority(execute)
    world = reduce_world_state(())

    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = CommandExecutionCoordinator(
            CommandToolRegistry((SimulatedBurnTool(),)),
            store,
            authority_resolver=lambda _grant_id: authority,
            world_state_resolver=lambda: world,
        )
        run = run_operator(
            objective=_objective(),
            authority=authority,
            reasoner=ScriptedReasoner((proposal, execute, finish)),
            evaluator=_UnusedEvaluator(),
            command_executor=coordinator,
        )

    assert run.schema_version == "1.2"
    assert run.status.value == "completed"
    assert run.steps[1].command_result is not None
    assert run.steps[1].command_result.status == "simulated"
    assert run.steps[1].command_execution_record is not None
    assert run.steps[1].command_execution_record.terminal.status.value == "committed"
    validate_operator_run_policy(run)


def test_command_record_tampering_fails_offline_verification(tmp_path: Path) -> None:
    proposal, execute, finish = _actions()
    authority = _authority(execute)
    world = reduce_world_state(())
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        run = run_operator(
            objective=_objective(),
            authority=authority,
            reasoner=ScriptedReasoner((proposal, execute, finish)),
            evaluator=_UnusedEvaluator(),
            command_executor=CommandExecutionCoordinator(
                CommandToolRegistry((SimulatedBurnTool(),)),
                store,
                authority_resolver=lambda _grant_id: authority,
                world_state_resolver=lambda: world,
            ),
        )

    record = run.steps[1].command_execution_record
    assert record is not None
    tampered_record = record.model_copy(
        update={
            "prepared": record.prepared.model_copy(
                update={"world_state_sha256": "f" * 64}
            )
        }
    )
    steps = list(run.steps)
    steps[1] = steps[1].model_copy(update={"command_execution_record": tampered_record})
    tampered = run.model_copy(update={"steps": tuple(steps)})

    with pytest.raises(OperatorPolicyError, match="world-state receipt mismatch"):
        validate_operator_run_policy(tampered)

    legacy_payload = run.model_dump(mode="python")
    legacy_payload["schema_version"] = "1.1"
    with pytest.raises(ValidationError, match="legacy operator schemas"):
        type(run).model_validate(legacy_payload)
