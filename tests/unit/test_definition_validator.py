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

    schema = load_schema(
        "schemas/playtest_definition.schema.json"
    )

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
            "success": True,
        },
    }

    schema = load_schema(
        "schemas/playtest_definition.schema.json"
    )

    with pytest.raises(DefinitionError):
        validate_definition(
            definition,
            schema,
        )