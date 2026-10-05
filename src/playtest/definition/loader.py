from pathlib import Path
from typing import Any

import json
import yaml

from playtest.core.errors import DefinitionError


def load_definition(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        raise DefinitionError(
            f"Definition file does not exist: {file_path}"
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DefinitionError(
            f"Could not read definition file: {file_path}"
        ) from exc

    suffix = file_path.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw_text)

        elif suffix == ".json":
            data = json.loads(raw_text)

        else:
            raise DefinitionError(
                f"Unsupported definition format: {suffix}"
            )

    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise DefinitionError(
            f"Definition file could not be parsed: {file_path}"
        ) from exc

    if not isinstance(data, dict):
        raise DefinitionError(
            "Definition root must be an object."
        )

    return data