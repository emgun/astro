from __future__ import annotations

import tomllib
from pathlib import Path

from astro_uq.cli import SOFTWARE_COMPATIBILITY


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

    assert pyproject["project"]["version"] == "0.2.0"
    assert pyproject["project"]["license"] == {"text": "Apache-2.0"}
    assert Path("LICENSE").exists()
    assert "License :: OSI Approved :: Apache Software License" in pyproject["project"][
        "classifiers"
    ]
    assert "Topic :: Scientific/Engineering :: Astronomy" in pyproject["project"]["classifiers"]


def test_campaign_compatibility_tracks_package_version() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert SOFTWARE_COMPATIBILITY["astro-suite"] == pyproject["project"]["version"]
    assert SOFTWARE_COMPATIBILITY["campaign-runtime"] == "1.2"


def test_dev_dependencies_keep_ci_type_checking_stable() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert "mypy>=1.11,<1.12" in dev_dependencies
    assert "types-PyYAML>=6.0" in dev_dependencies


def test_reentry_package_is_in_wheel_and_strict_type_surfaces() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "src/astro_reentry" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "packages"
    ]
    assert "astro_reentry" in pyproject["tool"]["mypy"]["packages"]


def test_mission_package_is_in_wheel_and_strict_type_surfaces() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        "src/astro_mission" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    )
    assert "astro_mission" in pyproject["tool"]["mypy"]["packages"]


def test_assurance_package_is_in_wheel_and_strict_type_surfaces() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/astro_assurance" in packages
    assert "astro_assurance" in pyproject["tool"]["mypy"]["packages"]
