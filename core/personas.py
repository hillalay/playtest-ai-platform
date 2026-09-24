"""
core/personas.py

Game-agnostic human player persona definitions.
"""

from dataclasses import dataclass
from enum import Enum


class GoalPriority(Enum):
    COMPLETION = "completion"
    EXPLORATION = "exploration"
    SAFETY = "safety"


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    error_rate: float
    lookahead_depth: int
    goal_priority: GoalPriority
    patience_multiplier: float


DEFAULT_PERSONAS = {
    "rusher": PersonaProfile(
        name="Rusher",
        error_rate=0.35,
        lookahead_depth=0,
        goal_priority=GoalPriority.COMPLETION,
        patience_multiplier=0.6,
    ),

    "cautious": PersonaProfile(
        name="Cautious",
        error_rate=0.05,
        lookahead_depth=2,
        goal_priority=GoalPriority.SAFETY,
        patience_multiplier=1.5,
    ),

    "explorer": PersonaProfile(
        name="Explorer",
        error_rate=0.50,
        lookahead_depth=0,
        goal_priority=GoalPriority.EXPLORATION,
        patience_multiplier=1.0,
    ),

    "novice": PersonaProfile(
        name="Novice",
        error_rate=0.60,
        lookahead_depth=0,
        goal_priority=GoalPriority.COMPLETION,
        patience_multiplier=0.5,
    ),
}