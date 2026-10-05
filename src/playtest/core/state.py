from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateMetadata:
    level_ref: str | None = None
    step_index: int = 0


@dataclass
class State:
    payload: Any
    metadata: StateMetadata = field(default_factory=StateMetadata)