from pathlib import Path

import pytest
from typer.testing import CliRunner

from playtest.cli import app


@pytest.mark.parametrize("command, expected", [
    ("validate-game", "Definition: VALID"),
    ("capabilities", "FULL_SEARCH"),
])
def test_cli_works_outside_repository_root(monkeypatch, command, expected):
    definition = Path("examples/arrow_grid/playtest.yaml").resolve()
    monkeypatch.chdir(definition.parent)
    result = CliRunner().invoke(app, [command, str(definition)])
    assert result.exit_code == 0, result.output
    assert expected in result.output
