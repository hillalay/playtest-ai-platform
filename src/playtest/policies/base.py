from typing import Protocol, Any

from playtest.core.actions import Action


class Policy(Protocol):

    def select_action(
        self,
        observation: Any,
        valid_actions: list[Action],
    ) -> Action:
        ...