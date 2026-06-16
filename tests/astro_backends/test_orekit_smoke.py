from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType

import pytest

from astro_backends.orekit import runtime
from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke

OREKIT_VERSION = "13.1.0"


class FakeOrekit:
    def initVM(self) -> None:
        return None


class FakePyHelpers:
    loaded_filenames: list[str] = []
    loaded_from_pip_library: list[bool] = []

    @classmethod
    def setup_orekit_data(cls, filenames: str, from_pip_library: bool) -> None:
        cls.loaded_filenames.append(filenames)
        cls.loaded_from_pip_library.append(from_pip_library)


def _fake_runtime_modules(
    *,
    frames_factory: object,
    time_scales_factory: object = object(),
) -> dict[str, object]:
    frames_module = ModuleType("org.orekit.frames")
    frames_module.FramesFactory = frames_factory
    frames_module.TopocentricFrame = object()
    time_module = ModuleType("org.orekit.time")
    time_module.TimeScalesFactory = time_scales_factory
    time_module.AbsoluteDate = object()
    geometry_module = ModuleType("org.hipparchus.geometry.euclidean.threed")
    geometry_module.Vector3D = object()
    utils_module = ModuleType("org.orekit.utils")
    utils_module.PVCoordinates = object()
    utils_module.IERSConventions = object()
    utils_module.Constants = object()
    orbits_module = ModuleType("org.orekit.orbits")
    orbits_module.CartesianOrbit = object()
    orbits_module.OrbitType = object()
    orbits_module.PositionAngleType = object()
    propagation_module = ModuleType("org.orekit.propagation")
    propagation_module.SpacecraftState = object()
    analytical_module = ModuleType("org.orekit.propagation.analytical")
    analytical_module.KeplerianPropagator = object()
    numerical_module = ModuleType("org.orekit.propagation.numerical")
    numerical_module.NumericalPropagator = object()
    conversion_module = ModuleType("org.orekit.propagation.conversion")
    conversion_module.NumericalPropagatorBuilder = object()
    conversion_module.DormandPrince853IntegratorBuilder = object()
    gravity_module = ModuleType("org.orekit.forces.gravity")
    gravity_module.J2OnlyPerturbation = object()
    gravity_module.ThirdBodyAttraction = object()
    bodies_module = ModuleType("org.orekit.bodies")
    bodies_module.OneAxisEllipsoid = object()
    bodies_module.CelestialBodyFactory = object()
    bodies_module.GeodeticPoint = object()
    atmosphere_module = ModuleType("org.orekit.models.earth.atmosphere")
    atmosphere_module.SimpleExponentialAtmosphere = object()
    drag_module = ModuleType("org.orekit.forces.drag")
    drag_module.DragForce = object()
    drag_module.IsotropicDrag = object()
    radiation_module = ModuleType("org.orekit.forces.radiation")
    radiation_module.SolarRadiationPressure = object()
    radiation_module.IsotropicRadiationSingleCoefficient = object()
    measurements_module = ModuleType("org.orekit.estimation.measurements")
    measurements_module.GroundStation = object()
    measurements_module.ObservableSatellite = object()
    measurements_module.Range = object()
    measurements_module.RangeRate = object()
    least_squares_module = ModuleType("org.orekit.estimation.leastsquares")
    least_squares_module.BatchLSEstimator = object()
    ode_module = ModuleType("org.hipparchus.ode.nonstiff")
    ode_module.DormandPrince853Integrator = object()
    hipparchus_least_squares_module = ModuleType(
        "org.hipparchus.optim.nonlinear.vector.leastsquares"
    )
    hipparchus_least_squares_module.LevenbergMarquardtOptimizer = object()
    return {
        "orekit_jpype": FakeOrekit(),
        "orekit_jpype.pyhelpers": FakePyHelpers,
        "org.orekit.frames": frames_module,
        "org.orekit.time": time_module,
        "org.hipparchus.geometry.euclidean.threed": geometry_module,
        "org.orekit.utils": utils_module,
        "org.orekit.orbits": orbits_module,
        "org.orekit.propagation": propagation_module,
        "org.orekit.propagation.analytical": analytical_module,
        "org.orekit.propagation.numerical": numerical_module,
        "org.orekit.propagation.conversion": conversion_module,
        "org.orekit.forces.gravity": gravity_module,
        "org.orekit.bodies": bodies_module,
        "org.orekit.models.earth.atmosphere": atmosphere_module,
        "org.orekit.forces.drag": drag_module,
        "org.orekit.forces.radiation": radiation_module,
        "org.orekit.estimation.measurements": measurements_module,
        "org.orekit.estimation.leastsquares": least_squares_module,
        "org.hipparchus.ode.nonstiff": ode_module,
        "org.hipparchus.optim.nonlinear.vector.leastsquares": hipparchus_least_squares_module,
    }


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

    monkeypatch.setattr(runtime, "version", missing_version)
    monkeypatch.setattr(runtime, "import_module", fail_import)

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

    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fail_wrapper_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version == OREKIT_VERSION
    assert "import failed" in result.message
    assert "broken jpype import" in result.message
    assert "not installed" not in result.message


def test_run_orekit_smoke_reports_vm_frame_time_failure_with_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "orekit-data.zip"
    data_path.write_bytes(b"placeholder")

    class FailingFramesFactory:
        @staticmethod
        def getEME2000() -> object:
            raise RuntimeError("frame unavailable")

    class FakeTimeScalesFactory:
        @staticmethod
        def getUTC() -> object:
            return object()

    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        modules = _fake_runtime_modules(
            frames_factory=FailingFramesFactory,
            time_scales_factory=FakeTimeScalesFactory,
        )
        return modules[module_name]

    monkeypatch.setenv("ASTRO_OREKIT_DATA_PATH", str(data_path))
    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fake_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version == OREKIT_VERSION
    assert "VM/frame/time smoke failure" in result.message
    assert "frame unavailable" in result.message


def test_load_orekit_runtime_sets_up_data_from_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "orekit-data.zip"
    data_path.write_bytes(b"placeholder")
    FakePyHelpers.loaded_filenames = []
    FakePyHelpers.loaded_from_pip_library = []
    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        modules = _fake_runtime_modules(frames_factory=object())
        return modules[module_name]

    monkeypatch.setenv("ASTRO_OREKIT_DATA_PATH", str(data_path))
    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fake_import)

    orekit_runtime = runtime.load_orekit_runtime()

    assert orekit_runtime.data_path == str(data_path)
    assert FakePyHelpers.loaded_filenames == [str(data_path)]
    assert FakePyHelpers.loaded_from_pip_library == [False]


def test_run_orekit_smoke_reports_missing_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_data_path = tmp_path / "missing-orekit-data.zip"

    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        assert module_name == "orekit_jpype"
        return FakeOrekit()

    monkeypatch.setenv("ASTRO_OREKIT_DATA_PATH", str(missing_data_path))
    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fake_import)

    result = run_orekit_smoke(strict=False)

    assert result.available is False
    assert result.version == OREKIT_VERSION
    assert "Orekit data path" in result.message
    assert str(missing_data_path) in result.message


def test_run_orekit_smoke_strict_re_raises_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(distribution_name: str) -> str:
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr(runtime, "version", missing_version)

    with pytest.raises(PackageNotFoundError):
        run_orekit_smoke(strict=True)


def test_run_orekit_smoke_strict_re_raises_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fail_wrapper_import(_module_name: str) -> ModuleType:
        raise ImportError("broken jpype import")

    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fail_wrapper_import)

    with pytest.raises(ImportError, match="broken jpype import"):
        run_orekit_smoke(strict=True)


def test_run_orekit_smoke_strict_re_raises_vm_frame_time_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "orekit-data.zip"
    data_path.write_bytes(b"placeholder")

    class FailingOrekit:
        def initVM(self) -> object:
            raise RuntimeError("vm unavailable")

    def fake_version(_distribution_name: str) -> str:
        return OREKIT_VERSION

    def fake_import(module_name: str) -> object:
        assert module_name == "orekit_jpype"
        return FailingOrekit()

    monkeypatch.setenv("ASTRO_OREKIT_DATA_PATH", str(data_path))
    monkeypatch.setattr(runtime, "version", fake_version)
    monkeypatch.setattr(runtime, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="vm unavailable"):
        run_orekit_smoke(strict=True)
