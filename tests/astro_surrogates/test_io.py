from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astro_surrogates.io import SurrogateArtifactError, load_npz_artifact, write_npz_artifact


def test_npz_json_round_trip_and_integrity(tmp_path: Path) -> None:
    digest = write_npz_artifact(
        tmp_path / "weights.npz",
        tmp_path / "model.json",
        {"weights": np.arange(6, dtype=np.float64).reshape(2, 3)},
        {"schema_version": "1.0", "model_id": "ridge"},
    )

    arrays, manifest = load_npz_artifact(
        tmp_path / "model.json", expected_manifest_sha256=digest
    )

    np.testing.assert_array_equal(arrays["weights"], np.arange(6).reshape(2, 3))
    assert manifest["array_names"] == ["weights"]
    assert manifest["arrays_path"] == "weights.npz"


def test_tampered_npz_is_rejected(tmp_path: Path) -> None:
    write_npz_artifact(
        tmp_path / "data.npz", tmp_path / "data.json", {"x": [1.0]}, {"kind": "dataset"}
    )
    with (tmp_path / "data.npz").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(SurrogateArtifactError, match="NPZ SHA-256"):
        load_npz_artifact(tmp_path / "data.json")


def test_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    digest = write_npz_artifact(
        tmp_path / "data.npz", tmp_path / "data.json", {"x": [1.0]}, {"kind": "dataset"}
    )
    manifest = json.loads((tmp_path / "data.json").read_text())
    manifest["kind"] = "model"
    (tmp_path / "data.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(SurrogateArtifactError, match="manifest SHA-256"):
        load_npz_artifact(tmp_path / "data.json", expected_manifest_sha256=digest)


def test_object_arrays_are_never_written_or_loaded(tmp_path: Path) -> None:
    payload = np.array([{"call": "anything"}], dtype=object)
    with pytest.raises(SurrogateArtifactError, match="object arrays"):
        write_npz_artifact(
            tmp_path / "unsafe.npz", tmp_path / "unsafe.json", {"payload": payload}, {}
        )
