from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class SurrogateArtifactError(ValueError):
    """A surrogate artifact is unsafe, corrupt, or does not match its manifest."""


def canonical_json(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SurrogateArtifactError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SurrogateArtifactError(f"could not hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _relative_path(path: str | Path) -> str:
    candidate = str(path).replace("\\", "/")
    normalized = PurePosixPath(candidate)
    if not candidate or normalized.is_absolute() or ".." in normalized.parts:
        raise SurrogateArtifactError("artifact paths must be relative and contained")
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
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _safe_array(name: str, value: Any) -> NDArray[Any]:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise SurrogateArtifactError(f"unsafe array name: {name!r}")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise SurrogateArtifactError(f"object arrays are prohibited: {name}")
    return array


def write_npz_artifact(
    arrays_path: str | Path,
    manifest_path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    """Atomically write data-only NPZ and canonical JSON bound to its digest.

    Returns the SHA-256 digest of the exact manifest bytes.
    """
    if not arrays:
        raise SurrogateArtifactError("an NPZ artifact must contain at least one array")
    reserved = {"arrays_path", "arrays_sha256", "array_names"}
    if reserved & metadata.keys():
        raise SurrogateArtifactError("metadata may not override integrity manifest fields")
    safe_arrays = {name: _safe_array(name, value) for name, value in arrays.items()}
    destination = Path(arrays_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        with Path(temporary).open("wb") as stream:
            np.savez(stream, **safe_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        npz_digest = sha256_file(temporary)
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise

    manifest = dict(metadata)
    manifest.update(
        arrays_path=_relative_path(destination.name),
        arrays_sha256=npz_digest,
        array_names=sorted(safe_arrays),
    )
    payload = canonical_json(manifest) + b"\n"
    _atomic_write(Path(manifest_path), payload)
    return sha256_bytes(payload)


def load_npz_artifact(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Verify and load a data-only NPZ artifact without pickle or executable loaders."""
    source = Path(manifest_path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise SurrogateArtifactError(f"could not read manifest {source}: {exc}") from exc
    if expected_manifest_sha256 is not None and sha256_bytes(payload) != expected_manifest_sha256:
        raise SurrogateArtifactError("manifest SHA-256 mismatch")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SurrogateArtifactError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or canonical_json(manifest) + b"\n" != payload:
        raise SurrogateArtifactError("manifest must be a canonical JSON object")
    try:
        relative = _relative_path(manifest["arrays_path"])
        expected_npz = manifest["arrays_sha256"]
        expected_names = manifest["array_names"]
    except (KeyError, TypeError) as exc:
        raise SurrogateArtifactError("manifest is missing integrity fields") from exc
    if not isinstance(expected_npz, str) or len(expected_npz) != 64:
        raise SurrogateArtifactError("manifest contains an invalid array digest")
    if not isinstance(expected_names, list) or not all(
        isinstance(name, str) for name in expected_names
    ):
        raise SurrogateArtifactError("manifest contains invalid array names")
    arrays_path = source.parent / relative
    if sha256_file(arrays_path) != expected_npz:
        raise SurrogateArtifactError("NPZ SHA-256 mismatch")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(expected_names):
                raise SurrogateArtifactError("NPZ members do not match the manifest")
            arrays = {name: _safe_array(name, archive[name]).copy() for name in archive.files}
    except SurrogateArtifactError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise SurrogateArtifactError(f"NPZ is unsafe or invalid: {exc}") from exc
    return arrays, manifest
