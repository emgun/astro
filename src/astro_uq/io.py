from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel

from astro_uq.models import CampaignDefinition, CampaignState

CAMPAIGN_FILE = "campaign.json"
SAMPLES_FILE = "samples.jsonl"
CASES_FILE = "cases.jsonl"
STATISTICS_FILE = "statistics.json"
SUMMARY_FILE = "summary.txt"
LOCK_FILE = ".campaign.lock"
TRANSACTION_FILE = ".campaign-transaction.json"


class CampaignIOError(ValueError):
    """Campaign artifacts are corrupt or incompatible with the requested resume."""


class CampaignLockedError(CampaignIOError):
    """Another owner holds the campaign directory lock."""


def load_campaign_definition(path: str | Path) -> CampaignDefinition:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignIOError(f"could not load campaign definition {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignIOError("campaign definition must contain a mapping")
    try:
        return CampaignDefinition.model_validate(payload)
    except Exception as exc:
        raise CampaignIOError(f"campaign definition is invalid: {exc}") from exc


@dataclass(frozen=True)
class ResumeState:
    state: CampaignState
    definition_digest: str
    completed_sample_ids: frozenset[str]
    samples: tuple[dict[str, Any], ...]
    cases: tuple[dict[str, Any], ...]
    statistics: dict[str, Any] | None


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for persisted artifacts and digests."""
    try:
        text = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CampaignIOError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def relative_artifact_path(path: str | Path) -> str:
    """Validate and normalize a campaign-local artifact reference."""
    candidate = str(path).replace("\\", "/")
    normalized = PurePosixPath(candidate)
    if not candidate or normalized.is_absolute() or ".." in normalized.parts:
        raise CampaignIOError("artifact references must be relative campaign paths")
    return normalized.as_posix()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write(Path(path), canonical_json(value) + b"\n")


def atomic_write_text(path: str | Path, value: str) -> None:
    _atomic_write(Path(path), value.encode("utf-8"))


def read_jsonl(path: str | Path, *, recover_truncated_final: bool = True) -> list[dict[str, Any]]:
    """Read JSONL, optionally removing only a malformed truncated final record."""
    artifact = Path(path)
    if not artifact.exists():
        return []
    payload = artifact.read_bytes()
    lines = payload.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        terminated = line.endswith((b"\n", b"\r"))
        raw = line.rstrip(b"\r\n")
        if not raw:
            raise CampaignIOError(f"blank JSONL record at line {index + 1} in {artifact}")
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise TypeError("record is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            is_truncated_tail = index == len(lines) - 1 and not terminated
            if recover_truncated_final and is_truncated_tail:
                _atomic_write(artifact, payload[:valid_bytes])
                return records
            raise CampaignIOError(
                f"invalid JSONL record at line {index + 1} in {artifact}"
            ) from exc
        records.append(record)
        valid_bytes += len(line)
    return records


def atomic_write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    _atomic_write(Path(path), payload)


def append_jsonl(path: str | Path, record: Any) -> None:
    artifact = Path(path)
    records = read_jsonl(artifact)
    records.append(_json_value(record))
    atomic_write_jsonl(artifact, records)


class CampaignArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock_token: str | None = None
        self._lock_fd: int | None = None

    def _path(self, name: str) -> Path:
        return self.root / name

    def acquire(self, *, owner: str | None = None) -> Mapping[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        record = {
            "hostname": socket.gethostname(),
            "owner": owner or os.environ.get("USER", "unknown"),
            "pid": os.getpid(),
            "token": token,
        }
        fd = os.open(self._path(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignLockedError(f"campaign is locked: {self._path(LOCK_FILE)}") from exc
            payload = canonical_json(record) + b"\n"
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            os.fsync(fd)
        except BaseException:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise
        self._lock_token = token
        self._lock_fd = fd
        return record

    def release(self) -> None:
        if self._lock_token is None or self._lock_fd is None:
            return
        try:
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            record = json.loads(os.read(self._lock_fd, 65536))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignLockedError("campaign ownership record is missing or corrupt") from exc
        if record.get("token") != self._lock_token:
            raise CampaignLockedError("campaign ownership changed before release")
        os.ftruncate(self._lock_fd, 0)
        os.fsync(self._lock_fd)
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_token = None
        self._lock_fd = None

    def __enter__(self) -> CampaignArtifactStore:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()

    def initialize(
        self,
        definition: CampaignDefinition,
        *,
        software_compatibility: Mapping[str, str],
    ) -> str:
        existing = [
            name
            for name in (CAMPAIGN_FILE, SAMPLES_FILE, CASES_FILE, STATISTICS_FILE, SUMMARY_FILE)
            if self._path(name).exists()
        ]
        if existing:
            raise CampaignIOError(
                "campaign directory already contains evidence; use resume or a new output directory"
            )
        digest = canonical_hash(definition)
        manifest = {
            "schema_version": "1.0",
            "campaign_id": definition.campaign_id,
            "definition": _json_value(definition),
            "definition_digest": digest,
            "software_compatibility": dict(software_compatibility),
            "state": CampaignState.PENDING.value,
            "sample_index_path": relative_artifact_path(SAMPLES_FILE),
            "case_index_path": relative_artifact_path(CASES_FILE),
            "statistics_path": relative_artifact_path(STATISTICS_FILE),
            "summary_path": relative_artifact_path(SUMMARY_FILE),
            "samples_digest": canonical_hash([]),
            "cases_digest": canonical_hash([]),
            "statistics_digest": None,
        }
        atomic_write_json(self._path(CAMPAIGN_FILE), manifest)
        atomic_write_jsonl(self._path(SAMPLES_FILE), ())
        atomic_write_jsonl(self._path(CASES_FILE), ())
        return digest

    def set_state(self, state: CampaignState) -> None:
        manifest = self._read_manifest()
        manifest["state"] = state.value
        atomic_write_json(self._path(CAMPAIGN_FILE), manifest)

    def append_sample(self, sample: Any) -> None:
        self._append_index(SAMPLES_FILE, "samples_digest", sample)

    def append_case(self, case: Any) -> None:
        self._append_index(CASES_FILE, "cases_digest", case)

    def write_statistics(self, statistics: Any) -> None:
        atomic_write_json(self._path(STATISTICS_FILE), statistics)
        manifest = self._read_manifest()
        manifest["statistics_digest"] = canonical_hash(statistics)
        atomic_write_json(self._path(CAMPAIGN_FILE), manifest)

    def write_summary(self, summary: str) -> None:
        atomic_write_text(self._path(SUMMARY_FILE), summary)

    def resume(
        self,
        definition: CampaignDefinition,
        *,
        software_compatibility: Mapping[str, str],
        require_completed_statistics: bool = True,
    ) -> ResumeState:
        self._recover_transaction()
        manifest = self._read_manifest()
        expected_digest = canonical_hash(definition)
        if manifest.get("definition_digest") != expected_digest:
            raise CampaignIOError("campaign definition digest is incompatible with resume")
        if manifest.get("software_compatibility") != dict(software_compatibility):
            raise CampaignIOError("campaign software compatibility policy does not match")
        stored_definition = manifest.get("definition")
        if canonical_hash(stored_definition) != expected_digest:
            raise CampaignIOError("stored campaign definition does not match its digest")
        try:
            state = CampaignState(manifest["state"])
        except (KeyError, ValueError) as exc:
            raise CampaignIOError("campaign manifest has an invalid state") from exc
        samples = tuple(read_jsonl(self._path(SAMPLES_FILE)))
        cases = tuple(read_jsonl(self._path(CASES_FILE)))
        if manifest.get("samples_digest") != canonical_hash(list(samples)):
            raise CampaignIOError("sample index integrity digest does not match")
        if manifest.get("cases_digest") != canonical_hash(list(cases)):
            raise CampaignIOError("case index integrity digest does not match")
        completed: set[str] = set()
        for case in cases:
            sample_id = case.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise CampaignIOError("case index contains a missing or invalid sample_id")
            if sample_id in completed:
                raise CampaignIOError(f"case index contains duplicate sample_id {sample_id!r}")
            completed.add(sample_id)
        statistics: dict[str, Any] | None = None
        statistics_path = self._path(STATISTICS_FILE)
        if statistics_path.exists():
            try:
                loaded = json.loads(statistics_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignIOError("campaign statistics are corrupt") from exc
            if not isinstance(loaded, dict):
                raise CampaignIOError("campaign statistics must be a JSON object")
            if manifest.get("statistics_digest") != canonical_hash(loaded):
                raise CampaignIOError("statistics integrity digest does not match")
            statistics = loaded
        if require_completed_statistics and state is CampaignState.COMPLETED and statistics is None:
            raise CampaignIOError("completed campaign is missing statistics evidence")
        return ResumeState(
            state,
            expected_digest,
            frozenset(completed),
            samples,
            cases,
            statistics,
        )

    def _append_index(self, filename: str, manifest_key: str, record: Any) -> None:
        if self._path(TRANSACTION_FILE).exists():
            self._recover_transaction()
        records = read_jsonl(self._path(filename))
        old_digest = canonical_hash(records)
        manifest = self._read_manifest()
        if manifest.get(manifest_key) != old_digest:
            raise CampaignIOError(f"{filename} and manifest digest diverged before append")
        records.append(_json_value(record))
        new_digest = canonical_hash(records)
        atomic_write_json(
            self._path(TRANSACTION_FILE),
            {
                "filename": filename,
                "manifest_key": manifest_key,
                "old_digest": old_digest,
                "new_digest": new_digest,
            },
        )
        atomic_write_jsonl(self._path(filename), records)
        manifest[manifest_key] = new_digest
        atomic_write_json(self._path(CAMPAIGN_FILE), manifest)
        self._remove_transaction()

    def _recover_transaction(self) -> None:
        transaction_path = self._path(TRANSACTION_FILE)
        if not transaction_path.exists():
            return
        try:
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            filename = str(transaction["filename"])
            manifest_key = str(transaction["manifest_key"])
            old_digest = str(transaction["old_digest"])
            new_digest = str(transaction["new_digest"])
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignIOError("campaign append transaction is corrupt") from exc
        if filename not in {SAMPLES_FILE, CASES_FILE} or manifest_key not in {
            "samples_digest",
            "cases_digest",
        }:
            raise CampaignIOError("campaign append transaction targets an invalid index")
        current_digest = canonical_hash(read_jsonl(self._path(filename)))
        manifest = self._read_manifest()
        manifest_digest = manifest.get(manifest_key)
        if current_digest == old_digest and manifest_digest == old_digest:
            self._remove_transaction()
            return
        if current_digest == new_digest and manifest_digest in {old_digest, new_digest}:
            if manifest_digest != new_digest:
                manifest[manifest_key] = new_digest
                atomic_write_json(self._path(CAMPAIGN_FILE), manifest)
            self._remove_transaction()
            return
        raise CampaignIOError("campaign append transaction cannot be recovered safely")

    def _remove_transaction(self) -> None:
        transaction_path = self._path(TRANSACTION_FILE)
        with suppress(FileNotFoundError):
            transaction_path.unlink()
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self._path(CAMPAIGN_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignIOError("campaign manifest is missing or corrupt") from exc
        if not isinstance(manifest, dict):
            raise CampaignIOError("campaign manifest must be a JSON object")
        return manifest
