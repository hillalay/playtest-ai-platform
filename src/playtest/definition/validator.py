from pathlib import Path
from typing import Any

import json
from jsonschema import Draft202012Validator

from playtest.core.errors import DefinitionError


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    path = Path(schema_path)

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DefinitionError(
            f"Could not load schema: {path}"
        ) from exc


def validate_definition(
    definition: dict[str, Any],
    schema: dict[str, Any],
) -> None:

    validator = Draft202012Validator(schema)

    errors = sorted(
        validator.iter_errors(definition),
        key=lambda error: list(error.absolute_path),
    )

    if not errors:
        return

    messages: list[str] = []

    for error in errors:
        location = ".".join(
            str(part)
            for part in error.absolute_path
        )

        if not location:
            location = "<root>"

        messages.append(
            f"{location}: {error.message}"
        )

    raise DefinitionError(
        "Definition validation failed:\n"
        + "\n".join(messages)
    )