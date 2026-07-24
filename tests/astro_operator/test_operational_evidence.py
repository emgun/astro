from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from astro_operator.io import verify_operator_run
from astro_operator.models import (
    AcquisitionStatus,
    EpistemicKind,
    EvidenceAcquisitionResult,
    EvidenceAssertion,
    EvidenceRequest,
    OperatorAction,
    OperatorActionKind,
)
from astro_operator.operational_evidence import (
    OperationalEvidenceKind,
    OperationalEvidenceSource,
    SimulatedTelemetryTool,
    verify_operational_acquisition,
)
from astro_operator.policy import action_digest
from astro_operator.world_state import reduce_world_state


def test_checked_operational_assertions_are_rederived_from_captured_bytes(
    tmp_path: Path,
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = tmp_path / "review"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            "examples/operator/post_launch_recovery_review.yaml",
            "--reasoner-replay",
            "examples/operator/post_launch_recovery_review_replay.yaml",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    run = verify_operator_run(output)
    acquisition = run.steps[1].acquisition_result
    assert acquisition is not None
    evidence = acquisition.evidence[0]

    verify_operational_acquisition(acquisition, output / evidence.path)
    changed = tuple(
        assertion.model_copy(update={"value": 9.9, "assertion_sha256": None})
        if assertion.predicate == "position_sigma_km"
        else assertion
        for assertion in acquisition.assertions
    )
    forged = acquisition.model_copy(
        update={"assertions": reduce_world_state(changed).assertions}
    )
    with pytest.raises(ValueError, match="do not match captured content"):
        verify_operational_acquisition(forged, output / evidence.path)


def test_operational_source_catalog_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    outside = tmp_path / "telemetry.yaml"
    outside.write_text(
        "\n".join(
            (
                "snapshot_id: escaped",
                "asset_id: spacecraft-a",
                "configuration_id: baseline",
                'observed_at: "2026-01-01T00:00:00Z"',
                'decision_time: "2026-01-01T00:00:01Z"',
                "operating_mode: nominal",
                "source_simulation: test",
            )
        ),
        encoding="utf-8",
    )
    tool = SimulatedTelemetryTool(
        (
            OperationalEvidenceSource(
                source_id="escaped",
                kind=OperationalEvidenceKind.SIMULATED_TELEMETRY,
                path="../telemetry.yaml",
            ),
        ),
        source_root=root,
        output_root=tmp_path / "output",
    )
    request = EvidenceRequest(
        request_id="request",
        tool_id=tool.spec.tool_id,
        tool_version=tool.spec.version,
        request_kind=tool.spec.request_kind,
        parameters={"source_id": "escaped"},
    )

    result = tool.acquire(request, reduce_world_state(()))

    assert result.status == AcquisitionStatus.FAILED
    assert "escapes" in result.message


def test_operational_source_catalog_rejects_symlinked_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    tool = SimulatedTelemetryTool(
        (
            OperationalEvidenceSource(
                source_id="snapshot",
                kind=OperationalEvidenceKind.SIMULATED_TELEMETRY,
                path="snapshot.yaml",
            ),
        ),
        source_root=linked_root,
        output_root=tmp_path / "output",
    )
    request = EvidenceRequest(
        request_id="request",
        tool_id=tool.spec.tool_id,
        tool_version=tool.spec.version,
        request_kind=tool.spec.request_kind,
        parameters={"source_id": "snapshot"},
    )

    result = tool.acquire(request, reduce_world_state(()))

    assert result.status == AcquisitionStatus.FAILED
    assert "symbolic link" in result.message


def test_recognized_operational_tool_requires_one_captured_artifact(
    tmp_path: Path,
) -> None:
    tool = SimulatedTelemetryTool(
        (), source_root=tmp_path, output_root=tmp_path
    )
    request = EvidenceRequest(
        request_id="request",
        tool_id=tool.spec.tool_id,
        tool_version=tool.spec.version,
        request_kind=tool.spec.request_kind,
        parameters={"source_id": "missing"},
    )
    assertion = EvidenceAssertion(
        assertion_id="forged",
        subject="spacecraft-a",
        predicate="asset_configuration_id",
        value="baseline",
        epistemic_kind=EpistemicKind.SIMULATED,
        scope="forged",
        source_evidence_ids=("base-scenario",),
        producer_tool_id=tool.spec.tool_id,
        producer_tool_version=tool.spec.version,
    )
    forged = EvidenceAcquisitionResult(
        request=request,
        tool=tool.spec,
        status=AcquisitionStatus.SUCCEEDED,
        assertions=(assertion,),
    )

    with pytest.raises(ValueError, match="one captured source"):
        verify_operational_acquisition(forged, tmp_path)

    failed = EvidenceAcquisitionResult(
        request=request,
        tool=tool.spec,
        status=AcquisitionStatus.FAILED,
        message="catalog source unavailable",
    )
    verify_operational_acquisition(failed, tmp_path)


def test_legacy_action_digest_preserves_user_parameters_named_predicates() -> None:
    action = OperatorAction(
        action_id="request",
        kind=OperatorActionKind.REQUEST_EVIDENCE,
        rationale="Preserve arbitrary typed request parameters.",
        evidence_request=EvidenceRequest(
            request_id="request",
            tool_id="custom",
            tool_version="1.0",
            request_kind="custom",
            parameters={"predicates": []},
        ),
    )
    payload = json.dumps(
        action.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert action_digest(action) == sha256(payload).hexdigest()
