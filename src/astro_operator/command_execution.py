"""Durable, fail-closed execution for explicitly registered command tools."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Protocol, cast
from uuid import uuid4

from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    AuthorityLevel,
    CommandEnvelope,
    CommandExecutionRecord,
    CommandExecutionRequest,
    CommandPreparedRecord,
    CommandRequest,
    CommandResult,
    CommandTerminalRecord,
    CommandTerminalStatus,
    OperatorAction,
    OperatorActionKind,
    WorldState,
)
from astro_operator.policy import action_digest
from astro_operator.reasoner import model_digest
from astro_operator.world_state import world_state_digest


class CommandTool(Protocol):
    """A versioned command implementation with qualification provenance."""

    command_type: str
    version: str
    simulation_only: bool
    qualification_sha256: str

    def execute(self, request: CommandRequest, world_state: WorldState) -> CommandResult: ...


class CommandToolRegistry:
    """Dispatch each command type to exactly one explicitly registered tool."""

    def __init__(self, tools: Iterable[CommandTool] = ()) -> None:
        self._tools: dict[str, CommandTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: CommandTool) -> None:
        if not tool.command_type or not tool.version:
            raise ValueError("command tool type and version must be non-empty")
        if len(tool.qualification_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in tool.qualification_sha256
        ):
            raise ValueError("command tool qualification digest must be lowercase SHA-256")
        if tool.command_type in self._tools:
            registered = self._tools[tool.command_type]
            raise ValueError(
                f"command tool {tool.command_type!r} is already registered at "
                f"version {registered.version!r}"
            )
        self._tools[tool.command_type] = tool

    def resolve(
        self,
        command_type: str,
        *,
        version: str | None = None,
        qualification_sha256: str | None = None,
    ) -> CommandTool:
        tool = self._tools.get(command_type)
        if tool is None:
            raise ValueError(f"command tool {command_type!r} is not registered")
        if version is not None and tool.version != version:
            raise ValueError(
                f"command tool {command_type!r} version mismatch: requested {version!r}, "
                f"registered {tool.version!r}"
            )
        if (
            qualification_sha256 is not None
            and tool.qualification_sha256 != qualification_sha256
        ):
            raise ValueError(f"command tool {command_type!r} qualification digest mismatch")
        return tool

    def execute(self, request: CommandRequest, world_state: WorldState) -> CommandResult:
        """Execute only through the exact tool registered for the request command type."""

        tool = self.resolve(request.command_type)
        dispatched = request.model_copy(deep=True)
        result = tool.execute(dispatched.model_copy(deep=True), world_state.model_copy(deep=True))
        if result.command != dispatched:
            raise ValueError("command tool result is not bound to the dispatched request")
        return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _record_digest(record: CommandPreparedRecord | CommandTerminalRecord) -> str:
    return sha256(
        _canonical_json(
            record.model_dump(mode="json", exclude={"record_sha256"})
        ).encode("utf-8")
    ).hexdigest()


class SQLiteCommandExecutionStore:
    """SQLite write-ahead journal for command preparation and terminal outcomes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.path.chmod(0o600)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS command_executions (
                idempotency_key TEXT PRIMARY KEY,
                execution_id TEXT NOT NULL UNIQUE,
                execution_action_sha256 TEXT NOT NULL,
                grant_id TEXT NOT NULL,
                grant_version INTEGER NOT NULL,
                command_type TEXT NOT NULL,
                prepared_json TEXT NOT NULL,
                terminal_json TEXT,
                result_json TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_approvals (
                approval_id TEXT PRIMARY KEY,
                execution_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY (execution_id) REFERENCES command_executions(execution_id)
            )
            """
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteCommandExecutionStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def lookup(
        self, idempotency_key: str
    ) -> tuple[CommandPreparedRecord, CommandTerminalRecord | None, CommandResult | None] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT prepared_json, terminal_json, result_json "
                "FROM command_executions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        prepared = CommandPreparedRecord.model_validate_json(row[0])
        terminal = CommandTerminalRecord.model_validate_json(row[1]) if row[1] else None
        result = CommandResult.model_validate_json(row[2]) if row[2] else None
        if prepared.record_sha256 != _record_digest(prepared):
            raise RuntimeError("stored command preparation digest does not match its content")
        if terminal is not None and terminal.record_sha256 != _record_digest(terminal):
            raise RuntimeError("stored command terminal digest does not match its content")
        if (
            terminal is not None
            and terminal.result_sha256 is not None
            and (result is None or model_digest(result) != terminal.result_sha256)
        ):
            raise RuntimeError(
                "stored command result digest does not match its terminal record"
            )
        return prepared, terminal, result

    def prepare(
        self,
        prepared: CommandPreparedRecord,
        *,
        execution_action_sha256: str,
        command_type: str,
        approval_id: str | None,
        max_commits: int,
    ) -> bool:
        """Atomically reserve idempotency, approval, and envelope capacity.

        Returns ``False`` only when the same idempotency key and exact execution action already
        exist. Any different reuse fails closed.
        """

        if prepared.record_sha256 != _record_digest(prepared):
            raise ValueError("command preparation record digest does not match its content")
        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT execution_action_sha256 FROM command_executions "
                    "WHERE idempotency_key = ?",
                    (prepared.idempotency_key,),
                ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    if existing[0] != execution_action_sha256:
                        raise OperatorPolicyError(
                            "idempotency key is already bound to a different execution action"
                        )
                    return False
                reserved = cast(
                    tuple[int],
                    connection.execute(
                        "SELECT COUNT(*) FROM command_executions WHERE grant_id = ? "
                        "AND grant_version = ? AND command_type = ?",
                        (prepared.grant_id, prepared.grant_version, command_type),
                    ).fetchone(),
                )[0]
                if reserved >= max_commits:
                    raise OperatorPolicyError(
                        f"command envelope commit budget exhausted for {command_type!r}"
                    )
                connection.execute(
                    "INSERT INTO command_executions VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                    (
                        prepared.idempotency_key,
                        prepared.execution_id,
                        execution_action_sha256,
                        prepared.grant_id,
                        prepared.grant_version,
                        command_type,
                        _canonical_json(prepared.model_dump(mode="json")),
                    ),
                )
                if approval_id is not None:
                    try:
                        connection.execute(
                            "INSERT INTO consumed_approvals VALUES (?, ?)",
                            (approval_id, prepared.execution_id),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise OperatorPolicyError(
                            f"approval {approval_id!r} has already been consumed"
                        ) from exc
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return True

    def finish(
        self, terminal: CommandTerminalRecord, result: CommandResult | None = None
    ) -> None:
        if terminal.record_sha256 != _record_digest(terminal):
            raise ValueError("command terminal record digest does not match its content")
        if terminal.status == CommandTerminalStatus.COMMITTED:
            if result is None or terminal.result_sha256 != model_digest(result):
                raise ValueError("committed terminal must bind the exact command result")
        elif result is not None:
            raise ValueError("non-committed terminal cannot store a command result")
        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT terminal_json FROM command_executions WHERE execution_id = ?",
                    (terminal.execution_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("cannot finish an execution that was not prepared")
                if row[0] is not None:
                    existing = CommandTerminalRecord.model_validate_json(row[0])
                    if existing != terminal:
                        raise OperatorPolicyError("command execution already has a terminal record")
                    connection.execute("COMMIT")
                    return
                connection.execute(
                    "UPDATE command_executions SET terminal_json = ?, result_json = ? "
                    "WHERE execution_id = ?",
                    (
                        _canonical_json(terminal.model_dump(mode="json")),
                        (
                            _canonical_json(result.model_dump(mode="json"))
                            if result is not None
                            else None
                        ),
                        terminal.execution_id,
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise


AuthorityResolver = Callable[[str], AuthorityGrant]
WorldStateResolver = Callable[[], WorldState]
Clock = Callable[[], datetime]


class CommandExecutionCoordinator:
    """Prepare, revalidate, execute, and durably commit one exact command action."""

    def __init__(
        self,
        registry: CommandToolRegistry,
        store: SQLiteCommandExecutionStore,
        *,
        authority_resolver: AuthorityResolver,
        world_state_resolver: WorldStateResolver,
        clock: Clock | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._authority_resolver = authority_resolver
        self._world_state_resolver = world_state_resolver
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(
        self,
        action: OperatorAction,
        proposal_action: OperatorAction,
        authority: AuthorityGrant,
        world_state: WorldState,
    ) -> tuple[CommandResult, CommandExecutionRecord]:
        request, command = self._validate_action_pair(action, proposal_action)
        execution_sha256 = action_digest(action)
        existing = self._store.lookup(request.idempotency_key)
        if existing is not None:
            return self._resolve_existing(existing, execution_sha256)

        tool = self._registry.resolve(command.command_type)
        self._validate_tool(tool)
        current_authority = self._current_authority(authority)
        current_state = self._current_world_state(request, world_state)
        envelope = self._validate_authority_and_envelope(
            action, command, current_authority, execution_sha256
        )
        self._validate_tool_envelope(tool, envelope)
        approval_id = self._approval_id(action, current_authority, execution_sha256)
        now = self._aware_now()
        prepared = CommandPreparedRecord(
            execution_id=str(uuid4()),
            idempotency_key=request.idempotency_key,
            proposal_action_id=proposal_action.action_id,
            proposal_sha256=action_digest(proposal_action),
            execution_action_sha256=execution_sha256,
            command_sha256=model_digest(command),
            tool_version=tool.version,
            tool_qualification_sha256=tool.qualification_sha256,
            simulation_only=tool.simulation_only,
            grant_id=current_authority.grant_id,
            grant_version=current_authority.grant_version,
            authority_sha256=model_digest(current_authority),
            world_state_sha256=current_state.state_sha256,
            approval_id=approval_id,
            prepared_at=now,
            record_sha256="0" * 64,
        )
        prepared = prepared.model_copy(update={"record_sha256": _record_digest(prepared)})
        created = self._store.prepare(
            prepared,
            execution_action_sha256=execution_sha256,
            command_type=command.command_type,
            approval_id=approval_id,
            max_commits=envelope.max_commits,
        )
        if not created:
            existing = self._store.lookup(request.idempotency_key)
            assert existing is not None
            return self._resolve_existing(existing, execution_sha256)

        try:
            current_authority = self._current_authority(authority)
            current_state = self._current_world_state(request, world_state)
            self._validate_authority_and_envelope(
                action, command, current_authority, execution_sha256
            )
        except Exception as exc:
            terminal = self._terminal(
                prepared.execution_id, CommandTerminalStatus.FAILED, str(exc)
            )
            self._store.finish(terminal)
            raise

        try:
            result = tool.execute(
                command.model_copy(deep=True), current_state.model_copy(deep=True)
            )
        except BaseException as exc:
            terminal = self._terminal(
                prepared.execution_id,
                CommandTerminalStatus.INDETERMINATE,
                f"command tool raised {type(exc).__name__}: {exc}",
            )
            self._store.finish(terminal)
            raise OperatorPolicyError(
                "command outcome is indeterminate; the idempotency key cannot be retried"
            ) from exc
        if result.command != command:
            terminal = self._terminal(
                prepared.execution_id,
                CommandTerminalStatus.INDETERMINATE,
                "command tool returned a result for a different command",
            )
            self._store.finish(terminal)
            raise OperatorPolicyError("command tool result is not bound to the dispatched command")

        # A third freshness check narrows the state/revocation race before the durable commit.
        try:
            current_authority = self._current_authority(authority)
            self._current_world_state(request, world_state)
            self._validate_authority_and_envelope(
                action, command, current_authority, execution_sha256
            )
        except Exception as exc:
            terminal = self._terminal(
                prepared.execution_id,
                CommandTerminalStatus.INDETERMINATE,
                f"post-dispatch freshness check failed: {exc}",
            )
            self._store.finish(terminal)
            raise OperatorPolicyError(
                "command ran but authority or world state changed before commit; "
                "outcome is indeterminate"
            ) from exc

        result_sha256 = model_digest(result)
        terminal = self._terminal(
            prepared.execution_id,
            CommandTerminalStatus.COMMITTED,
            result.message,
            result_sha256=result_sha256,
        )
        self._store.finish(terminal, result)
        return result, CommandExecutionRecord(prepared=prepared, terminal=terminal)

    def _validate_action_pair(
        self, action: OperatorAction, proposal: OperatorAction
    ) -> tuple[CommandExecutionRequest, CommandRequest]:
        if action.kind != OperatorActionKind.EXECUTE_COMMAND or action.command_execution is None:
            raise OperatorPolicyError("command execution requires a typed execution action")
        if proposal.kind != OperatorActionKind.PROPOSE_COMMAND or proposal.command is None:
            raise OperatorPolicyError("command execution requires the exact proposal action")
        request = action.command_execution
        command = proposal.command
        if request.proposal_action_id != proposal.action_id:
            raise OperatorPolicyError(
                "execution request does not name the supplied proposal action"
            )
        if request.command_id != command.command_id:
            raise OperatorPolicyError("execution request does not name the proposed command")
        return request, command

    def _current_authority(self, supplied: AuthorityGrant) -> AuthorityGrant:
        current = self._authority_resolver(supplied.grant_id)
        if current.grant_id != supplied.grant_id:
            raise OperatorPolicyError("authority resolver returned a different grant")
        if current.grant_version != supplied.grant_version:
            raise OperatorPolicyError("authority grant version changed before command execution")
        if current.revoked:
            raise OperatorPolicyError(f"authority grant {current.grant_id} is revoked")
        if current != supplied:
            raise OperatorPolicyError(
                "authority grant content changed without a grant-version increment"
            )
        now = self._aware_now()
        if current.valid_from is not None and now < current.valid_from:
            raise OperatorPolicyError("authority grant is not yet valid")
        if current.expires_at is not None and now >= current.expires_at:
            raise OperatorPolicyError("authority grant has expired")
        return current

    def _current_world_state(
        self, request: CommandExecutionRequest, supplied: WorldState
    ) -> WorldState:
        expected = request.expected_world_state_sha256
        if supplied.state_sha256 != world_state_digest(supplied):
            raise OperatorPolicyError("supplied world state digest does not match its content")
        if supplied.state_sha256 != expected:
            raise OperatorPolicyError(
                "execution expected world state does not match supplied state"
            )
        current = self._world_state_resolver()
        if current.state_sha256 != world_state_digest(current):
            raise OperatorPolicyError("current world state digest does not match its content")
        if current.state_sha256 != expected or current != supplied:
            raise OperatorPolicyError("world state changed before command execution")
        return current

    def _validate_authority_and_envelope(
        self,
        action: OperatorAction,
        command: CommandRequest,
        authority: AuthorityGrant,
        execution_sha256: str,
    ) -> CommandEnvelope:
        if OperatorActionKind.EXECUTE_COMMAND not in authority.allowed_actions:
            raise OperatorPolicyError("command execution is outside the current authority grant")
        if authority.level.value not in {
            AuthorityLevel.SUPERVISED_AUTONOMY.value,
            AuthorityLevel.DELEGATED_AUTONOMY.value,
            AuthorityLevel.MISSION_AUTONOMY.value,
        }:
            raise OperatorPolicyError("authority level cannot execute commands")
        if command.command_type not in authority.allowed_command_types:
            raise OperatorPolicyError("command type is outside the current authority grant")
        envelope = next(
            (
                item
                for item in authority.command_envelopes
                if item.command_type == command.command_type
            ),
            None,
        )
        if envelope is None:
            raise OperatorPolicyError("command type has no current execution envelope")
        if command.asset_id is None or command.asset_id not in envelope.allowed_asset_ids:
            raise OperatorPolicyError("command asset is outside the current execution envelope")
        limits = {item.parameter: item for item in envelope.parameter_limits}
        for parameter, limit in limits.items():
            if parameter not in command.parameters:
                raise OperatorPolicyError(f"command is missing bounded parameter {parameter!r}")
            value = command.parameters[parameter]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise OperatorPolicyError(f"command parameter {parameter!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or not limit.minimum <= numeric <= limit.maximum:
                raise OperatorPolicyError(
                    f"command parameter {parameter!r} is outside "
                    f"[{limit.minimum}, {limit.maximum}] {limit.unit}"
                )
        for parameter, value in command.parameters.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float) and parameter not in limits:
                raise OperatorPolicyError(
                    f"numeric command parameter {parameter!r} has no envelope limit"
                )
        self._approval_id(action, authority, execution_sha256)
        return envelope

    @staticmethod
    def _approval_id(
        action: OperatorAction, authority: AuthorityGrant, execution_sha256: str
    ) -> str | None:
        request = action.command_execution
        assert request is not None
        required = OperatorActionKind.EXECUTE_COMMAND in authority.approval_required_for
        if not required:
            if request.approval_id is not None:
                raise OperatorPolicyError(
                    "execution supplied an approval that the grant does not require"
                )
            return None
        if request.approval_id is None:
            raise OperatorPolicyError("command execution requires an exact approval ID")
        approval = next(
            (item for item in authority.approvals if item.approval_id == request.approval_id),
            None,
        )
        if approval is None:
            raise OperatorPolicyError("command approval is not present in the current grant")
        if (
            approval.grant_version != authority.grant_version
            or approval.action_id != action.action_id
            or approval.action_sha256 != execution_sha256
        ):
            raise OperatorPolicyError(
                "command approval is not bound to this exact execution action"
            )
        return approval.approval_id

    @staticmethod
    def _validate_tool(tool: CommandTool) -> None:
        if not tool.simulation_only:
            raise OperatorPolicyError(
                "real-effect command tools are not enabled by this execution coordinator"
            )
        if len(tool.qualification_sha256) != 64:
            raise OperatorPolicyError(
                "registered command tool has invalid qualification provenance"
            )

    @staticmethod
    def _validate_tool_envelope(tool: CommandTool, envelope: CommandEnvelope) -> None:
        if (
            tool.version != envelope.tool_version
            or tool.qualification_sha256 != envelope.tool_qualification_sha256
            or tool.simulation_only != envelope.simulation_only
        ):
            raise OperatorPolicyError(
                "registered command tool does not match the authority qualification envelope"
            )

    def _terminal(
        self,
        execution_id: str,
        status: CommandTerminalStatus,
        message: str,
        *,
        result_sha256: str | None = None,
    ) -> CommandTerminalRecord:
        terminal = CommandTerminalRecord(
            execution_id=execution_id,
            status=status,
            result_sha256=result_sha256,
            completed_at=self._aware_now(),
            message=message,
            record_sha256="0" * 64,
        )
        return terminal.model_copy(update={"record_sha256": _record_digest(terminal)})

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("command execution clock must return timezone-aware timestamps")
        return value

    @staticmethod
    def _resolve_existing(
        stored: tuple[CommandPreparedRecord, CommandTerminalRecord | None, CommandResult | None],
        execution_sha256: str,
    ) -> tuple[CommandResult, CommandExecutionRecord]:
        prepared, terminal, result = stored
        if prepared.execution_action_sha256 != execution_sha256:
            raise OperatorPolicyError(
                "idempotency key is already bound to a different execution action"
            )
        if terminal is None:
            raise OperatorPolicyError(
                "command was durably prepared without a terminal outcome and cannot be re-executed"
            )
        record = CommandExecutionRecord(prepared=prepared, terminal=terminal)
        if terminal.status == CommandTerminalStatus.COMMITTED and result is not None:
            return result, record
        raise OperatorPolicyError(
            f"command execution is terminal with status {terminal.status.value!r} "
            "and cannot be retried"
        )


class SimulatedBurnTool:
    """Deterministic simulation-only burn command with no external side effects."""

    command_type = "simulated_burn"
    version = "1.0"
    simulation_only = True
    qualification_sha256 = sha256(b"astro.simulated_burn.v1:no-real-effects").hexdigest()

    def execute(self, request: CommandRequest, world_state: WorldState) -> CommandResult:
        if request.command_type != self.command_type:
            raise ValueError("simulated burn tool received a different command type")
        if request.asset_id is None:
            raise ValueError("simulated burn requires asset_id")
        if set(request.parameters) != {"delta_v_m_s", "duration_s", "frame"}:
            raise ValueError(
                "simulated burn requires exactly delta_v_m_s, duration_s, and frame"
            )
        for name in ("delta_v_m_s", "duration_s"):
            value = request.parameters[name]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"simulated burn {name} must be an exact finite numeric scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"simulated burn {name} must be finite")
        frame = request.parameters["frame"]
        if not isinstance(frame, str) or not frame:
            raise ValueError("simulated burn frame must be a non-empty string")
        return CommandResult(
            command=request.model_copy(deep=True),
            status="simulated",
            message=(
                f"simulated burn for {request.asset_id} at world state "
                f"{world_state.state_sha256}"
            ),
        )
