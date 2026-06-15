from importlib.metadata import PackageNotFoundError
from types import ModuleType

import pytest

from astro_backends.orekit import smoke
from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke

OREKIT_VERSION = "13.1.0"


def test_run_orekit_smoke_reports_forced_unavailable_without_importing_wrapper() -> None:
    result = run_orekit_smoke(strict=False, force_unavailable=True)

    assert isinstance(result, OrekitSmokeResult)
    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version is None
    assert "not installed" in result.message
    assert result.to_dict() == {
        "available": False,
        "wrapper": "orekit_jpype",
        "version": None,
        "message": result.message,
    }


def test_run_orekit_smoke_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(distribution_name: str) -> str:
        raise PackageNotFoundError(distribution_name)

    def fail_import(_module_name: str) -> ModuleType:
        raise AssertionError("wrapper import should not run when distribution is missing")

    monkeypatch.setattr(smoke, "version", missing_version)
    monkeypatch.setattr(smoke, "import_module", fail_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version is None
    assert "not installed" in result.message


def test_run_orekit_smoke_reports_wrapper_import_failure_with_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fail_wrapper_import(module_name: str) -> ModuleType:
        assert module_name == "orekit_jpype"
        raise ImportError("broken jpype import")

    monkeypatch.setattr(smoke, "version", fake_version)
    monkeypatch.setattr(smoke, "import_module", fail_wrapper_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version == OREKIT_VERSION
    assert "import failed" in result.message
    assert "broken jpype import" in result.message
    assert "not installed" not in result.message


def test_run_orekit_smoke_reports_vm_frame_time_failure_with_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrekit:
        def initVM(self) -> None:
            return None

    class FailingFramesFactory:
        @staticmethod
        def getEME2000() -> object:
            raise RuntimeError("frame unavailable")

    class FakeTimeScalesFactory:
        @staticmethod
        def getUTC() -> object:
            return object()

    frames_module = ModuleType("org.orekit.frames")
    frames_module.FramesFactory = FailingFramesFactory
    time_module = ModuleType("org.orekit.time")
    time_module.TimeScalesFactory = FakeTimeScalesFactory

    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        modules: dict[str, object] = {
            "orekit_jpype": FakeOrekit(),
            "org.orekit.frames": frames_module,
            "org.orekit.time": time_module,
        }
        return modules[module_name]

    monkeypatch.setattr(smoke, "version", fake_version)
    monkeypatch.setattr(smoke, "import_module", fake_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version == OREKIT_VERSION
    assert "VM/frame/time smoke failure" in result.message
    assert "frame unavailable" in result.message


def test_run_orekit_smoke_strict_re_raises_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(distribution_name: str) -> str:
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr(smoke, "version", missing_version)

    with pytest.raises(PackageNotFoundError):
        run_orekit_smoke(strict=True)


def test_run_orekit_smoke_strict_re_raises_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fail_wrapper_import(_module_name: str) -> ModuleType:
        raise ImportError("broken jpype import")

    monkeypatch.setattr(smoke, "version", fake_version)
    monkeypatch.setattr(smoke, "import_module", fail_wrapper_import)

    with pytest.raises(ImportError, match="broken jpype import"):
        run_orekit_smoke(strict=True)


def test_run_orekit_smoke_strict_re_raises_vm_frame_time_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOrekit:
        def initVM(self) -> object:
            raise RuntimeError("vm unavailable")

    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        assert module_name == "orekit_jpype"
        return FailingOrekit()

    monkeypatch.setattr(smoke, "version", fake_version)
    monkeypatch.setattr(smoke, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="vm unavailable"):
        run_orekit_smoke(strict=True)
