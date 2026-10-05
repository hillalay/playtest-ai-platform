from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    type: str
    params: dict[str, Any] = field(default_factory=dict)