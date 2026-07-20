from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from astro_operator.command_execution import (
    CommandExecutionCoordinator,
    CommandToolRegistry,
    SimulatedBurnTool,
    SQLiteCommandExecutionStore,
)
from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    ActionApproval,
    AuthorityGrant,
    AuthorityLevel,
    CommandEnvelope,
    CommandExecutionRequest,
    CommandParameterLimit,
    CommandRequest,
    CommandResult,
    OperatorAction,
    OperatorActionKind,
    WorldState,
)
from astro_operator.policy import action_digest
from astro_operator.world_state import reduce_world_state


def _world() -> WorldState:
    return reduce_world_state(())


def _proposal(*, delta_v: float = 1.25) -> OperatorAction:
    return OperatorAction(
        action_id="proposal-1",
        kind=OperatorActionKind.PROPOSE_COMMAND,
        rationale="bounded simulated maneuver",
        command=CommandRequest(
            command_id="burn-1",
            command_type="simulated_burn",
            asset_id="sat-1",
            parameters={
                "delta_v_m_s": delta_v,
                "duration_s": 8.0,
                "frame": "TNW",
            },
        ),
    )


def _execution(
    world: WorldState,
    *,
    action_id: str = "execute-1",
    idempotency_key: str = "burn-attempt-1",
    approval_id: str | None = "approval-1",
) -> OperatorAction:
    return OperatorAction(
        action_id=action_id,
        kind=OperatorActionKind.EXECUTE_COMMAND,
        rationale="execute exact approved proposal",
        command_execution=CommandExecutionRequest(
            proposal_action_id="proposal-1",
            command_id="burn-1",
            idempotency_key=idempotency_key,
            expected_world_state_sha256=world.state_sha256,
            approval_id=approval_id,
        ),
    )


def _grant(action: OperatorAction, *, revoked: bool = False) -> AuthorityGrant:
    approval = ActionApproval(
        approval_id="approval-1",
        grant_version=3,
        action_id=action.action_id,
        action_sha256=action_digest(action),
    )
    return AuthorityGrant(
        grant_id="grant-1",
        grant_version=3,
        level=AuthorityLevel.SUPERVISED_AUTONOMY,
        mission_scope="sat-1 maneuver simulation",
        allowed_actions=(
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
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
                        parameter="delta_v_m_s", minimum=0.0, maximum=2.0, unit="m/s"
                    ),
                    CommandParameterLimit(
                        parameter="duration_s", minimum=1.0, maximum=20.0, unit="s"
                    ),
                ),
                max_commits=2,
            ),
        ),
        approval_required_for=(OperatorActionKind.EXECUTE_COMMAND,),
        approvals=(approval,),
        max_steps=10,
        max_candidate_evaluations=0,
        revoked=revoked,
    )


@dataclass
class _CountingBurnTool(SimulatedBurnTool):
    calls: int = 0

    def execute(self, request: CommandRequest, world_state: WorldState) -> CommandResult:
        self.calls += 1
        return super().execute(request, world_state)


class _CrashingTool(SimulatedBurnTool):
    calls = 0

    def execute(self, request: CommandRequest, world_state: WorldState) -> CommandResult:
        self.calls += 1
        raise RuntimeError("lost transport after dispatch")


def _coordinator(
    store: SQLiteCommandExecutionStore,
    tool: SimulatedBurnTool,
    grant_resolver: object,
    state_resolver: object,
) -> CommandExecutionCoordinator:
    return CommandExecutionCoordinator(
        CommandToolRegistry((tool,)),
        store,
        authority_resolver=grant_resolver,  # type: ignore[arg-type]
        world_state_resolver=state_resolver,  # type: ignore[arg-type]
    )


def test_committed_result_is_replayed_after_store_restart(tmp_path: Path) -> None:
    world = _world()
    action = _execution(world)
    grant = _grant(action)
    tool = _CountingBurnTool()
    path = tmp_path / "commands.sqlite3"

    with SQLiteCommandExecutionStore(path) as store:
        result, record = _coordinator(
            store, tool, lambda _grant_id: grant, lambda: world
        ).execute(
            action=action,
            proposal_action=_proposal(),
            authority=grant,
            world_state=world,
        )
        assert path.stat().st_mode & 0o777 == 0o600
    with SQLiteCommandExecutionStore(path) as reopened:
        replay, replay_record = _coordinator(
            reopened, tool, lambda _grant_id: grant, lambda: world
        ).execute(action, _proposal(), grant, world)

    assert replay == result
    assert replay_record == record
    assert tool.calls == 1


def test_same_idempotency_key_rejects_a_different_action(tmp_path: Path) -> None:
    world = _world()
    first = _execution(world)
    grant = _grant(first)
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        coordinator.execute(first, _proposal(), grant, world)
        changed = first.model_copy(update={"rationale": "different content"})
        with pytest.raises(OperatorPolicyError, match="different execution action"):
            coordinator.execute(changed, _proposal(), grant, world)
    assert tool.calls == 1


def test_approval_must_match_the_exact_execution_action(tmp_path: Path) -> None:
    world = _world()
    approved = _execution(world)
    changed = approved.model_copy(update={"rationale": "content changed after approval"})
    grant = _grant(approved)
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        with pytest.raises(OperatorPolicyError, match="exact execution action"):
            coordinator.execute(changed, _proposal(), grant, world)
        assert store.lookup("burn-attempt-1") is None
    assert tool.calls == 0


def test_tool_exception_is_indeterminate_and_never_reexecutes(tmp_path: Path) -> None:
    world = _world()
    action = _execution(world)
    grant = _grant(action)
    tool = _CrashingTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        with pytest.raises(OperatorPolicyError, match="indeterminate"):
            coordinator.execute(action, _proposal(), grant, world)
        with pytest.raises(OperatorPolicyError, match="cannot be retried"):
            coordinator.execute(action, _proposal(), grant, world)
    assert tool.calls == 1


def test_world_state_change_after_prepare_blocks_dispatch(tmp_path: Path) -> None:
    world = _world()
    changed = world.model_copy(update={"state_sha256": "f" * 64})
    action = _execution(world)
    grant = _grant(action)
    states = iter((world, changed))
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(
            store, tool, lambda _grant_id: grant, lambda: next(states)
        )
        with pytest.raises(OperatorPolicyError, match="digest does not match"):
            coordinator.execute(action, _proposal(), grant, world)
    assert tool.calls == 0


def test_revocation_after_prepare_blocks_dispatch(tmp_path: Path) -> None:
    world = _world()
    action = _execution(world)
    grant = _grant(action)
    grants = iter((grant, grant.model_copy(update={"revoked": True})))
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(
            store, tool, lambda _grant_id: next(grants), lambda: world
        )
        with pytest.raises(OperatorPolicyError, match="revoked"):
            coordinator.execute(action, _proposal(), grant, world)
    assert tool.calls == 0


def test_envelope_rejects_out_of_bounds_burn_before_prepare(tmp_path: Path) -> None:
    world = _world()
    action = _execution(world)
    grant = _grant(action)
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        with pytest.raises(OperatorPolicyError, match="outside"):
            coordinator.execute(action, _proposal(delta_v=2.5), grant, world)
        assert store.lookup("burn-attempt-1") is None
    assert tool.calls == 0


def test_expired_authority_blocks_prepare(tmp_path: Path) -> None:
    world = _world()
    action = _execution(world)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    grant = _grant(action).model_copy(update={"expires_at": now - timedelta(seconds=1)})
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = CommandExecutionCoordinator(
            CommandToolRegistry((tool,)),
            store,
            authority_resolver=lambda _grant_id: grant,
            world_state_resolver=lambda: world,
            clock=lambda: now,
        )
        with pytest.raises(OperatorPolicyError, match="expired"):
            coordinator.execute(action, _proposal(), grant, world)
        assert store.lookup("burn-attempt-1") is None
    assert tool.calls == 0


@pytest.mark.parametrize(
    "level", (AuthorityLevel.DELEGATED_AUTONOMY, AuthorityLevel.MISSION_AUTONOMY)
)
def test_higher_authority_still_uses_idempotency_and_envelope_without_per_action_approval(
    tmp_path: Path, level: AuthorityLevel
) -> None:
    world = _world()
    action = _execution(world, approval_id=None)
    grant = AuthorityGrant.model_validate(
        {
            **_grant(_execution(world)).model_dump(mode="python"),
            "level": level,
            "approval_required_for": (),
            "approvals": (),
        }
    )
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / f"{level.value}.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        first, _ = coordinator.execute(action, _proposal(), grant, world)
        replay, _ = coordinator.execute(action, _proposal(), grant, world)
    assert first == replay
    assert tool.calls == 1


def test_envelope_reserves_max_commit_capacity_at_prepare(tmp_path: Path) -> None:
    world = _world()
    first = _execution(world, approval_id=None)
    second = _execution(
        world,
        action_id="execute-2",
        idempotency_key="burn-attempt-2",
        approval_id=None,
    )
    supervised = _grant(_execution(world))
    data = supervised.model_dump(mode="python")
    data.update(
        level=AuthorityLevel.DELEGATED_AUTONOMY,
        approval_required_for=(),
        approvals=(),
    )
    data["command_envelopes"][0]["max_commits"] = 1
    grant = AuthorityGrant.model_validate(data)
    tool = _CountingBurnTool()
    with SQLiteCommandExecutionStore(tmp_path / "commands.sqlite3") as store:
        coordinator = _coordinator(store, tool, lambda _grant_id: grant, lambda: world)
        coordinator.execute(first, _proposal(), grant, world)
        with pytest.raises(OperatorPolicyError, match="budget exhausted"):
            coordinator.execute(second, _proposal(), grant, world)
    assert tool.calls == 1


def test_simulated_burn_is_exact_and_deterministic() -> None:
    tool = SimulatedBurnTool()
    request = _proposal().command
    assert request is not None
    first = tool.execute(request, _world())
    second = tool.execute(request, _world())
    assert first == second
    assert first.status == "simulated"
    assert first.evidence == ()

    extra = request.model_copy(
        update={"parameters": {**request.parameters, "unbounded_numeric": 1.0}}
    )
    with pytest.raises(ValueError, match="requires exactly"):
        tool.execute(extra, _world())
