from importlib.metadata import entry_points

from typer import Typer
from typer.testing import CliRunner


def test_packages_import() -> None:
    import astro_backends
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_od

    expected_core_exports = {
        "AstroError",
        "InvalidScenarioError",
        "J2_EARTH",
        "MU_EARTH_KM3_S2",
        "NumericalConvergenceError",
        "R_EARTH_KM",
        "SECONDS_PER_DAY",
        "Scenario",
        "UnsupportedBackendError",
    }

    assert expected_core_exports <= set(astro_core.__all__)
    assert astro_core.MU_EARTH_KM3_S2 == 398600.4418
    assert astro_core.R_EARTH_KM == 6378.1363
    assert astro_core.J2_EARTH == 1.08262668e-3
    assert astro_core.SECONDS_PER_DAY == 86400.0
    assert issubclass(astro_core.InvalidScenarioError, astro_core.AstroError)
    assert issubclass(astro_core.UnsupportedBackendError, astro_core.AstroError)
    assert issubclass(astro_core.NumericalConvergenceError, astro_core.AstroError)
    assert astro_dynamics.__all__ == []
    assert astro_od.__all__ == []
    assert astro_backends.__all__ == []
    assert astro_cli.__all__ == []


def test_console_script_entry_point_loads() -> None:
    entry_point = next(
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == "astro" and entry_point.value == "astro_cli.main:app"
    )

    app = entry_point.load()

    assert isinstance(app, Typer)

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
