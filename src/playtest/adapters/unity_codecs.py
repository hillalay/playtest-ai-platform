"""Small, game-owned observation and action contracts for the Unity backend."""

from copy import deepcopy
from typing import Any, Literal, Protocol

import numpy as np

from playtest.core.actions import Action

GameOutcome = Literal["SUCCESS", "FAILURE"]


class ObservationCodec(Protocol):
    # A version asserts that the observation is sufficient for semantic state identity.
    canonicalization_version: str | None
    goal_semantics: bool

    def observation_spec(self) -> dict[str, Any]: ...
    def decode(self, observations: list[np.ndarray]) -> Any: ...
    def canonical_state(self, observation: Any) -> bytes | None: ...
    def game_outcome(self, observation: Any) -> GameOutcome | None: ...


class RawObservationCodec:
    """Expose all sensor tensors without claiming full state or goal knowledge.

    Game codecs may subclass this or implement ObservationCodec directly. Only
    opt into canonicalization after checking hidden state and defining stable,
    versioned bytes. Outcomes must come from an explicit game protocol signal.
    """

    canonicalization_version = None
    goal_semantics = False

    def observation_spec(self) -> dict[str, Any]:
        return {"observations": "sensor tensors as nested lists; sensor order preserved"}

    def decode(self, observations: list[np.ndarray]) -> Any:
        return {"observations": [tensor.tolist() for tensor in observations]}

    def canonical_state(self, observation: Any) -> bytes | None:
        return None

    def game_outcome(self, observation: Any) -> GameOutcome | None:
        return None


class ActionMapper(Protocol):
    def action_spec(self) -> dict[str, Any]: ...
    def validate(self, action_count: int) -> None: ...
    def from_index(self, index: int, observation: Any) -> Action: ...
    def to_index(self, action: Action, observation: Any) -> int: ...


class DiscreteActionMapper:
    """Explicit index-to-semantic-action table, or a clearly named engine fallback.

    A supplied table must cover the entire branch with unique actions. Dynamic
    entity mappings can implement ActionMapper using the current observation.
    Neither form assumes persistent entity IDs equal engine indexes.
    """

    def __init__(self, actions: dict[int, Action] | None = None):
        self._actions = deepcopy(actions)

    def action_spec(self) -> dict[str, Any]:
        if self._actions is None:
            return {"unity_discrete": {"index": "int (engine branch index, not entity ID)"}}
        return {"index_to_action": deepcopy(self._actions)}

    def validate(self, action_count: int) -> None:
        if self._actions is None:
            return
        if (any(type(index) is not int for index in self._actions)
                or set(self._actions) != set(range(action_count))):
            raise ValueError("Action mapping must cover every discrete branch index.")
        actions = list(self._actions.values())
        if any(not isinstance(action, Action) for action in actions):
            raise ValueError("Action mapping values must be Action objects.")
        if any(action in actions[:i] for i, action in enumerate(actions)):
            raise ValueError("Action mapping must contain unique semantic actions.")

    def from_index(self, index: int, observation: Any) -> Action:
        if self._actions is None:
            return Action("unity_discrete", {"index": index})
        return deepcopy(self._actions[index])

    def to_index(self, action: Action, observation: Any) -> int:
        if not isinstance(action, Action):
            raise ValueError("Expected an Action.")
        if self._actions is None:
            if (action.type != "unity_discrete" or set(action.params) != {"index"}
                    or type(action.params["index"]) is not int):
                raise ValueError("Expected unity_discrete with an integer index.")
            return action.params["index"]
        for index, semantic_action in self._actions.items():
            if action == semantic_action:
                return index
        raise ValueError("Unknown semantic Unity action.")
