from __future__ import annotations

import tomllib
from pathlib import Path


def test_dependency_pins_keep_optional_launch_stack_numpy_1_compatible() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "numpy>=1.26,<2" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["optional-dependencies"]["launch"] == [
        "rocketpy>=1.11,<1.12",
    ]
    assert pyproject["project"]["optional-dependencies"]["optimization"] == [
        "dymos>=1.13.1,<1.14",
        "openmdao>=3.41,<3.42",
    ]


def test_public_package_metadata_declares_license_and_classifiers() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["license"] == {"text": "Apache-2.0"}
    assert Path("LICENSE").exists()
    assert "License :: OSI Approved :: Apache Software License" in pyproject["project"][
        "classifiers"
    ]
    assert "Topic :: Scientific/Engineering :: Astronomy" in pyproject["project"]["classifiers"]
