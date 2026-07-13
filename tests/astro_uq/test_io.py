from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from astro_uq.io import (
    CASES_FILE,
    TRANSACTION_FILE,
    CampaignArtifactStore,
    CampaignIOError,
    CampaignLockedError,
    canonical_hash,
    canonical_json,
    load_campaign_definition,
    read_jsonl,
    relative_artifact_path,
)
from astro_uq.models import (
    CampaignDefinition,
    CampaignState,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)


def definition(*, samples: int = 3) -> CampaignDefinition:
    return CampaignDefinition(
        campaign_id="io-fixture",
        workflow=WorkflowSpec(kind="propagate", scenario="scenario.yaml"),
        uncertainty=UncertaintyModel(
            parameters=(
                UncertainParameter(
                    parameter_id="mass",
                    target="vehicle.mass",
                    unit="kg",
                    uncertainty_kind=UncertaintyKind.ALEATORY,
                    distribution=DistributionSpec(
                        kind=DistributionKind.UNIFORM, low=99.0, high=101.0
                    ),
                ),
            )
        ),
        sampler=SamplerSpec(kind=SamplerKind.PSEUDORANDOM, samples=samples, seed=7),
        evaluator=EvaluatorSpec(
            evaluator_id="authoritative",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow="propagate",
            implementation_version="1",
            claim_boundary="fixture only",
        ),
    )


def test_canonical_json_and_hash_ignore_mapping_order() -> None:
    left = {"z": 1, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "z": 1}

    assert canonical_json(left) == b'{"nested":{"a":1,"b":2},"z":1}'
    assert canonical_hash(left) == canonical_hash(right)


def test_relative_artifact_references_reject_escape() -> None:
    assert relative_artifact_path(Path("artifacts") / "case.json") == "artifacts/case.json"
    with pytest.raises(CampaignIOError, match="relative"):
        relative_artifact_path("../case.json")
    with pytest.raises(CampaignIOError, match="relative"):
        relative_artifact_path("/tmp/case.json")


def test_load_campaign_definition_round_trips_yaml(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        json.dumps(definition().model_dump(mode="json")),
        encoding="utf-8",
    )

    loaded = load_campaign_definition(path)

    assert loaded == definition()


def test_initialize_resume_and_atomic_artifacts(tmp_path: Path) -> None:
    store = CampaignArtifactStore(tmp_path)
    digest = store.initialize(definition(), software_compatibility={"astro": "1"})
    store.set_state(CampaignState.INTERRUPTED)
    store.append_sample({"sample_id": "sample-0", "value": 1.0})
    store.append_case({"sample_id": "sample-0", "status": "success"})
    store.write_statistics({"completed_samples": 1})
    store.write_summary("interrupted after one sample\n")

    resumed = store.resume(definition(), software_compatibility={"astro": "1"})

    assert resumed.definition_digest == digest
    assert resumed.state is CampaignState.INTERRUPTED
    assert resumed.completed_sample_ids == {"sample-0"}
    manifest = json.loads((tmp_path / "campaign.json").read_text())
    assert manifest["case_index_path"] == "cases.jsonl"
    assert not list(tmp_path.glob(".*.json.*"))


def test_initialize_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    store = CampaignArtifactStore(tmp_path)
    store.initialize(definition(), software_compatibility={"astro": "1"})

    with pytest.raises(CampaignIOError, match="already contains evidence"):
        store.initialize(definition(), software_compatibility={"astro": "1"})


def test_resume_rejects_modified_case_evidence(tmp_path: Path) -> None:
    store = CampaignArtifactStore(tmp_path)
    store.initialize(definition(), software_compatibility={"astro": "1"})
    store.append_case({"sample_id": "sample-0", "metric": 1.0})
    case_path = tmp_path / CASES_FILE
    case_path.write_text('{"metric":999.0,"sample_id":"sample-0"}\n', encoding="utf-8")

    with pytest.raises(CampaignIOError, match="integrity digest"):
        store.resume(definition(), software_compatibility={"astro": "1"})


def test_resume_rejects_definition_or_software_change(tmp_path: Path) -> None:
    store = CampaignArtifactStore(tmp_path)
    store.initialize(definition(), software_compatibility={"astro": "1"})

    with pytest.raises(CampaignIOError, match="definition digest"):
        store.resume(definition(samples=4), software_compatibility={"astro": "1"})
    with pytest.raises(CampaignIOError, match="software compatibility"):
        store.resume(definition(), software_compatibility={"astro": "2"})


def test_jsonl_recovers_truncated_final_record(tmp_path: Path) -> None:
    path = tmp_path / CASES_FILE
    path.write_bytes(b'{"sample_id":"sample-0"}\n{"sample_id":"sam')

    assert read_jsonl(path) == [{"sample_id": "sample-0"}]
    assert path.read_bytes() == b'{"sample_id":"sample-0"}\n'


def test_jsonl_rejects_interior_corruption(tmp_path: Path) -> None:
    path = tmp_path / CASES_FILE
    path.write_bytes(b'{"sample_id":"sample-0"}\nnot-json\n{"sample_id":"sample-2"}\n')

    with pytest.raises(CampaignIOError, match="line 2"):
        read_jsonl(path)


def test_lock_is_exclusive_and_owner_can_release(tmp_path: Path) -> None:
    first = CampaignArtifactStore(tmp_path)
    second = CampaignArtifactStore(tmp_path)
    owner = first.acquire(owner="test-runner")

    assert owner["owner"] == "test-runner"
    with pytest.raises(CampaignLockedError):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


def test_os_lock_is_released_when_owner_process_descriptor_closes(tmp_path: Path) -> None:
    first = CampaignArtifactStore(tmp_path)
    second = CampaignArtifactStore(tmp_path)
    first.acquire(owner="crashed-owner")
    assert first._lock_fd is not None
    os.close(first._lock_fd)
    first._lock_fd = None
    first._lock_token = None

    owner = second.acquire(owner="replacement-owner")

    assert owner["owner"] == "replacement-owner"
    second.release()


def test_resume_recovers_committed_index_when_manifest_update_was_interrupted(
    tmp_path: Path,
) -> None:
    store = CampaignArtifactStore(tmp_path)
    store.initialize(definition(), software_compatibility={"astro": "1"})
    old_digest = canonical_hash([])
    cases = [{"sample_id": "sample-0", "status": "success"}]
    new_digest = canonical_hash(cases)
    (tmp_path / CASES_FILE).write_text(
        "".join(json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n" for case in cases),
        encoding="utf-8",
    )
    (tmp_path / TRANSACTION_FILE).write_text(
        json.dumps(
            {
                "filename": CASES_FILE,
                "manifest_key": "cases_digest",
                "old_digest": old_digest,
                "new_digest": new_digest,
            }
        ),
        encoding="utf-8",
    )

    resumed = store.resume(definition(), software_compatibility={"astro": "1"})

    assert resumed.completed_sample_ids == {"sample-0"}
    assert not (tmp_path / TRANSACTION_FILE).exists()
    manifest = json.loads((tmp_path / "campaign.json").read_text())
    assert manifest["cases_digest"] == new_digest


def test_interrupted_resume_matches_uninterrupted_case_index(tmp_path: Path) -> None:
    uninterrupted = CampaignArtifactStore(tmp_path / "uninterrupted")
    interrupted = CampaignArtifactStore(tmp_path / "interrupted")
    policy = {"astro": "1"}
    uninterrupted.initialize(definition(), software_compatibility=policy)
    interrupted.initialize(definition(), software_compatibility=policy)

    for index in range(3):
        case = {"sample_id": f"sample-{index}", "sample_index": index}
        uninterrupted.append_case(case)
        if index == 0:
            interrupted.append_case(case)
    interrupted.set_state(CampaignState.INTERRUPTED)
    resume = interrupted.resume(definition(), software_compatibility=policy)
    for index in range(3):
        if f"sample-{index}" not in resume.completed_sample_ids:
            interrupted.append_case({"sample_id": f"sample-{index}", "sample_index": index})
    interrupted.set_state(CampaignState.COMPLETED)

    assert (tmp_path / "interrupted" / CASES_FILE).read_bytes() == (
        tmp_path / "uninterrupted" / CASES_FILE
    ).read_bytes()
