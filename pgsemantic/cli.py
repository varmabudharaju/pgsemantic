"""CLI root — registers all subcommands for pgsemantic."""
from __future__ import annotations

import typer

from pgsemantic import __version__

app = typer.Typer(
    name="pgsemantic",
    help="Zero-config semantic search bootstrap for any PostgreSQL database.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        print(f"pgsemantic {__version__}")  # noqa: T201
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """pgsemantic: Point it at your Postgres database. Get semantic search in 60 seconds."""


# Register subcommands
from pgsemantic.commands.apply import apply_command  # noqa: E402
from pgsemantic.commands.index import index_command  # noqa: E402
from pgsemantic.commands.inspect import inspect_command  # noqa: E402
from pgsemantic.commands.integrate import integrate_command  # noqa: E402
from pgsemantic.commands.retry import retry_command  # noqa: E402
from pgsemantic.commands.search import search_command  # noqa: E402
from pgsemantic.commands.serve import serve_command  # noqa: E402
from pgsemantic.commands.status import status_command  # noqa: E402
from pgsemantic.commands.ui import ui_command  # noqa: E402
from pgsemantic.commands.worker import worker_command  # noqa: E402

app.command(name="inspect")(inspect_command)
app.command(name="apply")(apply_command)
app.command(name="index")(index_command)
app.command(name="search")(search_command)
app.command(name="worker")(worker_command)
app.command(name="serve")(serve_command)
app.command(name="status")(status_command)
app.command(name="integrate")(integrate_command)
app.command(name="ui")(ui_command)
app.command(name="retry")(retry_command)
