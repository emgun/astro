from importlib.metadata import entry_points

from typer import Typer
from typer.testing import CliRunner


def test_packages_import() -> None:
    import astro_backends
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_od

    assert astro_core.__all__ == []
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
