"""Single-agent Arrow Puzzle bridge; ML-Agents imports stay inside this module."""

import json
from typing import Any

import numpy as np

from playtest.core.actions import Action
from playtest.core.errors import EnvironmentError
from playtest.core.results import StepResult


class UnityMLAgentsAdapter:
    """Connect on reset, then operate on the cached Unity steps.

    Only the already-open scene (level_ref='static') is supported. Actions use
    Action('select_arrow', {'action_id': N}); N is the discrete action index.
    Runtime reseeding, level injection and arbitrary-state queries are unsupported.
    """

    adapter_id = "unity-mlagents"
    adapter_version = "0.1.0"

    def __init__(self, file_name: str | None = None, timeout_wait: int = 120):
        self.file_name = file_name
        self.timeout_wait = timeout_wait
        self._env = None
        self._closed = False
        self._behavior_name = None
        self._action_count = 0
        self.decision_steps = None
        self.terminal_steps = None
        self._observation = None

    def capabilities(self) -> dict[str, Any]:
        return {"canonical_state": True, "valid_actions": True,
                "clone_restore": False, "goal_semantics": False,
                "level_loading": False, "seed": False}

    def load_level(self, level_ref: str, seed: int | None = None) -> None:
        self._check_seed(seed)
        if level_ref != "static":
            raise EnvironmentError("Unity level injection is unsupported; use level_ref='static'.")
        # Explicit no-op: the active Unity scene owns its level.

    @staticmethod
    def _check_seed(seed: int | None) -> None:
        if seed is not None:
            raise EnvironmentError("Unity runtime reseeding is not supported.")

    def reset(self, seed: int | None = None) -> Any:
        self._check_seed(seed)
        if self._closed:
            raise EnvironmentError("Unity adapter is closed.")
        if self._env is None:
            try:
                from mlagents_envs.environment import UnityEnvironment
            except ImportError as exc:
                raise EnvironmentError("Install mlagents-envs in the Unity spike environment.") from exc
            self._env = UnityEnvironment(file_name=self.file_name, timeout_wait=self.timeout_wait)
        self._observation = None
        try:
            self._env.reset()
            names = list(self._env.behavior_specs)
            if len(names) != 1:
                raise EnvironmentError(f"Expected exactly one Unity behavior; found {names!r}.")
            self._behavior_name = names[0]
            spec = self._env.behavior_specs[self._behavior_name].action_spec
            if spec.continuous_size or len(spec.discrete_branches) != 1:
                raise EnvironmentError("Arrow Puzzle requires one discrete branch and no continuous actions.")
            self._action_count = int(spec.discrete_branches[0])
            if self._action_count <= 0:
                raise EnvironmentError("Unity discrete branch must contain actions.")
            self._refresh_steps()
        except Exception:
            self.close()
            raise
        return self._copy_observation()

    def _require_state(self) -> None:
        if self._closed or self._observation is None:
            raise EnvironmentError("Reset an open Unity adapter before querying or stepping it.")

    def _refresh_steps(self) -> None:
        self.decision_steps, self.terminal_steps = self._env.get_steps(self._behavior_name)
        if len(self.decision_steps) + len(self.terminal_steps) != 1:
            raise EnvironmentError("Expected exactly one agent in decision or terminal steps.")
        steps = self.terminal_steps if len(self.terminal_steps) else self.decision_steps
        if len(steps.obs) != 1:
            raise EnvironmentError("Expected one vector observation containing arrow records.")
        vector = np.asarray(steps.obs[0])
        if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] % 5:
            raise EnvironmentError("Expected observation shape (1, 5 * arrow_count).")
        if not np.all(np.isfinite(vector)) or not np.all(vector == np.floor(vector)):
            raise EnvironmentError("Arrow observation must contain finite integer values.")
        arrows = sorted(vector.astype(int).reshape(-1, 5).tolist())
        if len({arrow[0] for arrow in arrows}) != len(arrows):
            raise EnvironmentError("Arrow observation contains duplicate arrow IDs.")
        if any(arrow[4] not in (0, 1) for arrow in arrows):
            raise EnvironmentError("Arrow isActive must be 0 or 1.")
        self._observation = {"arrows": arrows}

    def _copy_observation(self) -> dict[str, Any]:
        return {"arrows": [arrow.copy() for arrow in self._observation["arrows"]]}

    def observation_spec(self) -> dict[str, Any]:
        return {"arrows": ["arrowId", "row", "column", "direction", "isActive"]}

    def action_spec(self) -> dict[str, Any]:
        return {"select_arrow": {"action_id": "int (discrete branch index)"}}

    def valid_actions(self, state: Any | None = None) -> list[Action] | None:
        self._require_state()
        if state is not None:
            raise EnvironmentError("Unity action enumeration supports only the current state.")
        if len(self.terminal_steps):
            return []
        mask = self.decision_steps.action_mask
        if mask is None:
            ids = range(self._action_count)
        else:
            if len(mask) != 1 or np.asarray(mask[0]).shape != (1, self._action_count):
                raise EnvironmentError("Unity action mask does not match the discrete branch.")
            if np.asarray(mask[0]).dtype.kind != "b":
                raise EnvironmentError("Unity action mask must contain booleans.")
            ids = [i for i, disabled in enumerate(mask[0][0]) if not disabled]
        return [Action("select_arrow", {"action_id": i}) for i in ids]

    def step(self, action: Action) -> StepResult:
        self._require_state()
        if (not isinstance(action, Action)
                or type(action.params.get("action_id")) is not int
                or action not in self.valid_actions()):
            return StepResult(next_observation=self._copy_observation(),
                              invalid_reason="Invalid or disabled Unity action",
                              game_terminal=bool(len(self.terminal_steps)))
        from mlagents_envs.base_env import ActionTuple

        actions = ActionTuple(discrete=np.array([[action.params["action_id"]]], dtype=np.int32))
        self._env.set_actions(self._behavior_name, actions)
        self._env.step()
        self._observation = None
        self._refresh_steps()
        terminal = bool(len(self.terminal_steps))
        steps = self.terminal_steps if terminal else self.decision_steps
        interrupted = terminal and bool(steps.interrupted[0])
        return StepResult(next_observation=self._copy_observation(),
                          reward_signals={"unity": float(steps.reward[0])},
                          game_terminal=terminal and not interrupted,
                          test_boundary_reached=interrupted,
                          state_signature=self.canonical_state().decode("utf-8"))

    def canonical_state(self) -> bytes | None:
        self._require_state()
        return json.dumps(self._observation, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def goal_test(self, state: Any | None = None) -> bool | None:
        self._require_state()
        # TODO: Unity must report explicit completion versus failure. EndEpisode,
        # interruption and rewards alone cannot establish a successful level.
        return None

    def clone_state(self) -> Any | None:
        return None

    def restore_state(self, handle: Any) -> None:
        raise EnvironmentError("Unity state restoration is unsupported.")

    def events(self) -> list[Any]:
        return []

    def close(self) -> None:
        env, self._env = self._env, None
        self._closed = True
        self._observation = None
        self.decision_steps = self.terminal_steps = None
        if env is not None:
            env.close()
