from typer import Typer


def _callback() -> None:
    """Astro Suite command line interface."""


app: Typer = Typer(callback=_callback)
