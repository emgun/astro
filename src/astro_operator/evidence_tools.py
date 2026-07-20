"""Deterministic dispatch for provider-neutral evidence acquisition tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from astro_operator.models import (
    EvidenceAcquisitionResult,
    EvidenceRequest,
    EvidenceToolSpec,
    WorldState,
)


class EvidenceTool(Protocol):
    """A versioned evidence producer with a declared request/output contract."""

    @property
    def spec(self) -> EvidenceToolSpec: ...

    def acquire(
        self, request: EvidenceRequest, world_state: WorldState
    ) -> EvidenceAcquisitionResult: ...


class EvidenceToolRegistry:
    """Dispatch requests only to an exactly matching, explicitly registered tool."""

    def __init__(self, tools: Iterable[EvidenceTool] = ()) -> None:
        self._tools: dict[tuple[str, str], EvidenceTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: EvidenceTool) -> None:
        spec = tool.spec
        key = (spec.tool_id, spec.version)
        if key in self._tools:
            raise ValueError(
                f"evidence tool {spec.tool_id!r}@{spec.version!r} is already registered"
            )
        self._tools[key] = tool

    def acquire(
        self, request: EvidenceRequest, world_state: WorldState
    ) -> EvidenceAcquisitionResult:
        tool = self._tools.get((request.tool_id, request.tool_version))
        if tool is None:
            available_versions = sorted(
                version for tool_id, version in self._tools if tool_id == request.tool_id
            )
            if available_versions:
                raise ValueError(
                    f"evidence tool {request.tool_id!r} version mismatch: requested "
                    f"{request.tool_version!r}, registered {available_versions!r}"
                )
            raise ValueError(f"evidence tool {request.tool_id!r} is not registered")
        spec = tool.spec.model_copy(deep=True)
        if request.request_kind != spec.request_kind:
            raise ValueError(
                f"evidence tool {request.tool_id!r} does not support request kind "
                f"{request.request_kind!r}"
            )

        dispatched_request = request.model_copy(deep=True)
        result = tool.acquire(
            dispatched_request.model_copy(deep=True),
            world_state.model_copy(deep=True),
        )
        if result.request != dispatched_request:
            raise ValueError("evidence tool result is not bound to the dispatched request")
        if result.tool != spec:
            raise ValueError(
                "evidence tool result does not match the registered tool specification"
            )
        return result
