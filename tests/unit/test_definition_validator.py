import pytest

from playtest.core.errors import DefinitionError
from playtest.definition.loader import load_definition
from playtest.definition.validator import (
    load_schema,
    validate_definition,
)


def test_arrow_definition_is_valid():

    definition = load_definition(
        "examples/arrow_grid/playtest.yaml"
    )

    schema = load_schema()

    validate_definition(
        definition,
        schema,
    )


def test_missing_game_id_is_invalid():

    definition = {
        "schema_version": "1.0",

        "game": {
            "display_name": "Broken Game",
            "deterministic": True,
        },

        "environment": {
            "time_model": "discrete",
        },

        "observation": {},

        "actions": {
            "finite": True,
        },

        "goals": {
            "success": "value >= 5",
        },
    }

    schema = load_schema()

    with pytest.raises(DefinitionError):
        validate_definition(
            definition,
            schema,
        )


@pytest.mark.parametrize("section, field, value", [
    (None, "schema_version", "unknown"),
    ("goals", "success", None),
    ("goals", "success", ""),
    ("goals", "success", "   "),
    ("goals", "success", True),
    ("goals", "success", {}),
    ("goals", "failure", ""),
])
def test_invalid_version_and_goals_are_rejected(section, field, value):
    definition = load_definition("examples/arrow_grid/playtest.yaml")
    target = definition if section is None else definition[section]
    target[field] = value
    with pytest.raises(DefinitionError):
        validate_definition(definition, load_schema())
