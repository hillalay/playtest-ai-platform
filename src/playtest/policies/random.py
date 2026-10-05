import random
from typing import Any

from playtest.core.actions import Action


class RandomPolicy:

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_action(
        self,
        observation: Any,
        valid_actions: list[Action],
    ) -> Action:

        if not valid_actions:
            raise ValueError("No valid actions available.")

        return self._rng.choice(valid_actions)