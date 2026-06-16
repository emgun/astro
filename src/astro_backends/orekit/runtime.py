from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from astro_core.errors import UnsupportedBackendError

WRAPPER = "orekit_jpype"
DISTRIBUTION = "orekit-jpype"
OREKIT_DATA_ENV_VARS = ("ASTRO_OREKIT_DATA_PATH", "OREKIT_DATA_PATH")
DEFAULT_OREKIT_DATA_PATH = Path.home() / ".orekit" / "orekit-data.zip"


class OrekitRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional Orekit runtime cannot be initialized."""

    def __init__(self, message: str, *, wrapper_version: str | None = None) -> None:
        super().__init__(message)
        self.wrapper_version = wrapper_version


@dataclass(frozen=True)
class OrekitRuntime:
    wrapper: str
    wrapper_version: str
    data_path: str
    frames_factory: Any
    time_scales_factory: Any
    absolute_date: Any
    vector3d: Any
    pv_coordinates: Any
    cartesian_orbit: Any
    keplerian_propagator: Any
    spacecraft_state: Any
    numerical_propagator: Any
    numerical_propagator_builder: Any
    dormand_prince_853_integrator: Any
    dormand_prince_853_integrator_builder: Any
    j2_only_perturbation: Any
    third_body_attraction: Any
    one_axis_ellipsoid: Any
    geodetic_point: Any
    topocentric_frame: Any
    simple_exponential_atmosphere: Any
    drag_force: Any
    isotropic_drag: Any
    celestial_body_factory: Any
    solar_radiation_pressure: Any
    isotropic_radiation_single_coefficient: Any
    orekit_ground_station: Any
    observable_satellite: Any
    range_measurement: Any
    range_rate_measurement: Any
    batch_ls_estimator: Any
    levenberg_marquardt_optimizer: Any
    orbit_type: Any
    position_angle_type: Any
    iers_conventions: Any
    constants: Any


def _runtime_unavailable(
    message: str,
    *,
    wrapper_version: str | None = None,
) -> OrekitRuntimeUnavailable:
    return OrekitRuntimeUnavailable(
        f"Orekit backend unavailable: {message}",
        wrapper_version=wrapper_version,
    )


def _configured_data_path() -> Path:
    for env_var in OREKIT_DATA_ENV_VARS:
        configured_path = os.environ.get(env_var)
        if configured_path:
            return Path(configured_path).expanduser()
    return DEFAULT_OREKIT_DATA_PATH


def _missing_data_error(data_path: Path, *, wrapper_version: str) -> OrekitRuntimeUnavailable:
    return _runtime_unavailable(
        f"Orekit data path {data_path} does not exist; download orekit-data.zip "
        "or set ASTRO_OREKIT_DATA_PATH to a valid Orekit data zip/folder.",
        wrapper_version=wrapper_version,
    )


def load_orekit_runtime(*, strict: bool = False) -> OrekitRuntime:
    try:
        wrapper_version = version(DISTRIBUTION)
    except PackageNotFoundError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            "Orekit JPype wrapper is not installed; install astro-suite[orekit] "
            "to enable orekit-jpype.",
        ) from exc

    try:
        orekit = import_module(WRAPPER)
    except ImportError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"Orekit JPype wrapper import failed: {exc}",
            wrapper_version=wrapper_version,
        ) from exc

    data_path = _configured_data_path()
    if not data_path.exists():
        if strict:
            raise FileNotFoundError(data_path)
        raise _missing_data_error(data_path, wrapper_version=wrapper_version)

    try:
        orekit.initVM()
        pyhelpers_module = import_module("orekit_jpype.pyhelpers")
        pyhelpers_module.setup_orekit_data(
            filenames=str(data_path),
            from_pip_library=False,
        )
        frames_module = import_module("org.orekit.frames")
        time_module = import_module("org.orekit.time")
        geometry_module = import_module("org.hipparchus.geometry.euclidean.threed")
        utils_module = import_module("org.orekit.utils")
        orbits_module = import_module("org.orekit.orbits")
        propagation_module = import_module("org.orekit.propagation")
        analytical_module = import_module("org.orekit.propagation.analytical")
        numerical_module = import_module("org.orekit.propagation.numerical")
        conversion_module = import_module("org.orekit.propagation.conversion")
        gravity_module = import_module("org.orekit.forces.gravity")
        bodies_module = import_module("org.orekit.bodies")
        atmosphere_module = import_module("org.orekit.models.earth.atmosphere")
        drag_module = import_module("org.orekit.forces.drag")
        radiation_module = import_module("org.orekit.forces.radiation")
        measurements_module = import_module("org.orekit.estimation.measurements")
        least_squares_module = import_module("org.orekit.estimation.leastsquares")
        ode_module = import_module("org.hipparchus.ode.nonstiff")
        hipparchus_least_squares_module = import_module(
            "org.hipparchus.optim.nonlinear.vector.leastsquares"
        )
    except Exception as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"JVM, Orekit imports, or data context failed: {exc}",
            wrapper_version=wrapper_version,
        ) from exc

    return OrekitRuntime(
        wrapper=WRAPPER,
        wrapper_version=wrapper_version,
        data_path=str(data_path),
        frames_factory=frames_module.FramesFactory,
        time_scales_factory=time_module.TimeScalesFactory,
        absolute_date=time_module.AbsoluteDate,
        vector3d=geometry_module.Vector3D,
        pv_coordinates=utils_module.PVCoordinates,
        cartesian_orbit=orbits_module.CartesianOrbit,
        keplerian_propagator=analytical_module.KeplerianPropagator,
        spacecraft_state=propagation_module.SpacecraftState,
        numerical_propagator=numerical_module.NumericalPropagator,
        numerical_propagator_builder=conversion_module.NumericalPropagatorBuilder,
        dormand_prince_853_integrator=ode_module.DormandPrince853Integrator,
        dormand_prince_853_integrator_builder=(
            conversion_module.DormandPrince853IntegratorBuilder
        ),
        j2_only_perturbation=gravity_module.J2OnlyPerturbation,
        third_body_attraction=gravity_module.ThirdBodyAttraction,
        one_axis_ellipsoid=bodies_module.OneAxisEllipsoid,
        geodetic_point=bodies_module.GeodeticPoint,
        topocentric_frame=frames_module.TopocentricFrame,
        simple_exponential_atmosphere=atmosphere_module.SimpleExponentialAtmosphere,
        drag_force=drag_module.DragForce,
        isotropic_drag=drag_module.IsotropicDrag,
        celestial_body_factory=bodies_module.CelestialBodyFactory,
        solar_radiation_pressure=radiation_module.SolarRadiationPressure,
        isotropic_radiation_single_coefficient=(
            radiation_module.IsotropicRadiationSingleCoefficient
        ),
        orekit_ground_station=measurements_module.GroundStation,
        observable_satellite=measurements_module.ObservableSatellite,
        range_measurement=measurements_module.Range,
        range_rate_measurement=measurements_module.RangeRate,
        batch_ls_estimator=least_squares_module.BatchLSEstimator,
        levenberg_marquardt_optimizer=(
            hipparchus_least_squares_module.LevenbergMarquardtOptimizer
        ),
        orbit_type=orbits_module.OrbitType,
        position_angle_type=orbits_module.PositionAngleType,
        iers_conventions=utils_module.IERSConventions,
        constants=utils_module.Constants,
    )
