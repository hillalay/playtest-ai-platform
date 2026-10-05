from typing import Protocol, Any

from playtest.core.actions import Action
from playtest.core.results import StepResult


class GameAdapter(Protocol):

    adapter_id: str
    adapter_version: str

    def capabilities(self) -> dict[str, Any]:
        ...

    def load_level(
        self,
        level_ref: str,
        seed: int | None = None,
    ) -> None:
        ...

    def reset(
        self,
        seed: int | None = None,
    ) -> Any:
        ...

    def observation_spec(self) -> dict[str, Any]:
        ...

    def action_spec(self) -> dict[str, Any]:
        ...

    def valid_actions(
        self,
        state: Any | None = None,
    ) -> list[Action] | None:
        ...

    def step(
        self,
        action: Action,
    ) -> StepResult:
        ...

    def canonical_state(self) -> bytes | None:
        ...

    def clone_state(self) -> Any | None:
        ...

    def restore_state(
        self,
        handle: Any,
    ) -> None:
        ...

    def goal_test(
        self,
        state: Any | None = None,
    ) -> bool | None:
        ...

    def events(self) -> list[Any]:
        ...

    def close(self) -> None:
        ...