from pathlib import Path

import typer
from rich.console import Console

from playtest.definition.loader import load_definition
from playtest.definition.validator import load_schema, validate_definition
from playtest.definition.capabilities import analyze_capabilities

app = typer.Typer()
console = Console()


@app.command("hello")
def hello():
    console.print("Playtest AI is running")


@app.command("validate-game")
def validate_game(
    definition_path: Path,
):
    """
    Validate a Playtest Game Definition file.
    """

    definition = load_definition(definition_path)

    schema = load_schema(
        "schemas/playtest_definition.schema.json"
    )

    validate_definition(
        definition,
        schema,
    )

    console.print("[green]Definition: VALID[/green]")


@app.command("capabilities")
def capabilities(
    definition_path: Path,
):
    """
    Analyze game capabilities from a PGD file.
    """

    definition = load_definition(definition_path)

    schema = load_schema(
        "schemas/playtest_definition.schema.json"
    )

    validate_definition(
        definition,
        schema,
    )

    report = analyze_capabilities(definition)

    console.print("\n[bold]Capabilities[/bold]")
    console.print(
        f"State observable: {report.state_observable}"
    )
    console.print(
        f"Finite actions: {report.finite_actions}"
    )
    console.print(
        f"Transition model: {report.transition_model}"
    )
    console.print(
        f"Deterministic: {report.deterministic}"
    )
    console.print(
        f"Goal semantics: {report.goal_semantics}"
    )
    console.print(
        f"Event instrumentation: {report.event_instrumentation}"
    )
    console.print(
        f"Headless: {report.headless}"
    )
    console.print(
        f"Parallel safe: {report.parallel_safe}"
    )

    console.print()
    console.print(
        f"Solver: [bold]{report.solver_grade.value}[/bold]"
    )
    console.print(
        f"Replay: [bold]{report.replay_mode.value}[/bold]"
    )


if __name__ == "__main__":
    app()