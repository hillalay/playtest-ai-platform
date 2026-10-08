"""Reusable single-agent, single-discrete-branch Unity ML-Agents backend."""

from copy import deepcopy
from hashlib import sha256
from typing import Any

import numpy as np

from playtest.adapters.unity_codecs import (
    ActionMapper,
    DiscreteActionMapper,
    ObservationCodec,
    RawObservationCodec,
)
from playtest.core.actions import Action
from playtest.core.errors import EnvironmentError
from playtest.core.results import StepResult


class UnityMLAgentsAdapter:
    """Connect lazily on reset; never reset or cross an episode inside step.

    Only level_ref='static' is supported. max_wait_steps bounds additional
    simulation advances while waiting for this agent; each network call also
    uses ML-Agents' timeout_wait. A decision must mean logical action completion
    in the game's Unity integration, not merely transport acknowledgement.
    """

    adapter_id = "unity-mlagents"
    adapter_version = "0.2.0"

    def __init__(
        self,
        file_name: str | None = None,
        timeout_wait: int = 120,
        *,
        behavior_name: str | None = None,
        observation_codec: ObservationCodec | None = None,
        action_mapper: ActionMapper | None = None,
        max_wait_steps: int = 100,
    ):
        if type(max_wait_steps) is not int or max_wait_steps < 0:
            raise ValueError("max_wait_steps must be a non-negative integer.")
        if type(timeout_wait) is not int or timeout_wait <= 0:
            raise ValueError("timeout_wait must be a positive integer.")
        self.file_name = file_name
        self.timeout_wait = timeout_wait
        self.behavior_name = behavior_name
        self.codec = observation_codec if observation_codec is not None else RawObservationCodec()
        self.mapper = action_mapper if action_mapper is not None else DiscreteActionMapper()
        version = self.codec.canonicalization_version
        if version is not None and (not isinstance(version, str) or not version.strip()):
            raise ValueError("Canonicalization requires a non-empty version string.")
        self.max_wait_steps = max_wait_steps
        self._env = None
        self._closed = False
        self._ready = False
        self._behavior_name = None
        self._action_count = 0
        self._agent_id = None
        self.decision_steps = self.terminal_steps = None
        self._observation = None
        self._canonical = None
        self._enabled: list[int] = []
        self._terminal = self._interrupted = False
        self._outcome = None
        self._reward = 0.0

    def capabilities(self) -> dict[str, Any]:
        return {
            "canonical_state": self.codec.canonicalization_version is not None,
            "canonicalization_version": self.codec.canonicalization_version,
            "valid_actions": True,
            "clone_restore": False,
            "goal_semantics": self.codec.goal_semantics,
            "level_loading": False,
            "seed": False,
            "deterministic": False,
            "transition_model": False,
            "event_instrumentation": False,
            "parallel_safe": False,
            "headless": False,
        }

    def load_level(self, level_ref: str, seed: int | None = None) -> None:
        self._check_seed(seed)
        if self._closed:
            raise EnvironmentError("Unity adapter is closed.")
        if level_ref != "static":
            raise EnvironmentError("Unity level injection is unsupported; use level_ref='static'.")

    @staticmethod
    def _check_seed(seed: int | None) -> None:
        if seed is not None:
            raise EnvironmentError("Unity runtime reseeding is not supported.")

    def reset(self, seed: int | None = None) -> Any:
        self._check_seed(seed)
        if self._closed:
            raise EnvironmentError("Unity adapter is closed.")
        self._ready = False
        self._agent_id = None
        try:
            if self._env is None:
                try:
                    from mlagents_envs.environment import UnityEnvironment
                except ImportError as exc:
                    raise EnvironmentError("Install mlagents-envs==1.1.0 in Python 3.10.") from exc
                self._env = UnityEnvironment(file_name=self.file_name, timeout_wait=self.timeout_wait)
            self._env.reset()
            names = list(self._env.behavior_specs)
            if self.behavior_name is not None:
                if self.behavior_name not in names:
                    raise EnvironmentError(
                        f"Unity behavior {self.behavior_name!r} is missing; available: {names!r}."
                    )
                self._behavior_name = self.behavior_name
            elif len(names) == 1:
                self._behavior_name = names[0]
            else:
                raise EnvironmentError(
                    f"Expected exactly one Unity behavior; found {names!r}. Set behavior_name explicitly."
                )
            spec = self._env.behavior_specs[self._behavior_name].action_spec
            if spec.continuous_size or len(spec.discrete_branches) != 1:
                raise EnvironmentError("Unity backend requires one discrete branch and no continuous actions.")
            self._action_count = int(spec.discrete_branches[0])
            if self._action_count <= 0:
                raise EnvironmentError("Unity discrete branch must contain actions.")
            self.mapper.validate(self._action_count)
            self._await_steps()
        except Exception as exc:
            raise self._abort("reset/connection", exc) from exc
        return deepcopy(self._observation)

    def _abort(self, operation: str, error: Exception) -> EnvironmentError:
        message = f"Unity {operation} failed: {error}"
        try:
            self.close()
        except Exception as cleanup_error:
            message += f"; cleanup also failed: {cleanup_error}"
        return EnvironmentError(message)

    def _require_state(self) -> None:
        if self._closed or not self._ready:
            raise EnvironmentError("Reset an open Unity adapter before querying or stepping it.")

    def _await_steps(self) -> None:
        for advance in range(self.max_wait_steps + 1):
            self.decision_steps, self.terminal_steps = self._env.get_steps(self._behavior_name)
            # Stepping would implicitly send zero actions to other active behaviors.
            for name in self._env.behavior_specs:
                if name != self._behavior_name:
                    decision, terminal = self._env.get_steps(name)
                    if len(decision) or len(terminal):
                        raise EnvironmentError("Multiple active behaviors/agents are unsupported.")
            decision, terminal = self.decision_steps, self.terminal_steps
            ids = set(map(int, decision.agent_id)) | set(map(int, terminal.agent_id))
            if len(decision) > 1 or len(terminal) > 1 or len(ids) > 1:
                raise EnvironmentError("Multiple Unity agents are unsupported.")
            if ids:
                agent_id = next(iter(ids))
                if self._agent_id is not None and agent_id != self._agent_id:
                    raise EnvironmentError("Unity agent changed without a terminal step; refusing a new episode.")
                self._agent_id = agent_id
                # A terminal always wins, even if a new decision is in the same batch.
                self._cache_steps(terminal if len(terminal) else decision, bool(len(terminal)))
                return
            if advance < self.max_wait_steps:
                self._env.step()
        raise EnvironmentError(f"No Unity decision or terminal after {self.max_wait_steps} waiting steps.")

    def _cache_steps(self, steps: Any, terminal: bool) -> None:
        sensor_specs = self._env.behavior_specs[self._behavior_name].observation_specs
        if len(steps.obs) != len(sensor_specs):
            raise EnvironmentError("Unity observations do not match the sensor spec.")
        observations = []
        for tensor, spec in zip(steps.obs, sensor_specs):
            tensor = np.asarray(tensor)
            if tensor.shape != (1, *spec.shape) or not np.all(np.isfinite(tensor)):
                raise EnvironmentError("Unity observation has an invalid shape or non-finite values.")
            observations.append(np.array(tensor[0], copy=True))
        enabled = [] if terminal else list(range(self._action_count))
        if not terminal and steps.action_mask is not None:
            mask = steps.action_mask
            if len(mask) != 1 or np.asarray(mask[0]).shape != (1, self._action_count):
                raise EnvironmentError("Unity action mask does not match the discrete branch.")
            if np.asarray(mask[0]).dtype.kind != "b":
                raise EnvironmentError("Unity action mask must contain booleans.")
            enabled = [i for i, disabled in enumerate(mask[0][0]) if not disabled]
        observation = self.codec.decode(observations)
        canonical = None
        if self.codec.canonicalization_version is not None:
            canonical = self.codec.canonical_state(deepcopy(observation))
            if not isinstance(canonical, bytes):
                raise EnvironmentError("Canonical codec must return bytes for its advertised capability.")
        outcome = self.codec.game_outcome(deepcopy(observation)) if self.codec.goal_semantics else None
        if outcome not in (None, "SUCCESS", "FAILURE"):
            raise EnvironmentError("Codec game outcome must be SUCCESS, FAILURE or None.")
        reward = float(steps.reward[0])
        if not np.isfinite(reward):
            raise EnvironmentError("Unity reward must be finite.")
        self._observation = deepcopy(observation)
        self._canonical = canonical
        self._enabled = enabled
        self._terminal = terminal
        self._interrupted = terminal and bool(steps.interrupted[0])
        self._outcome = outcome
        self._reward = reward
        self._ready = True

    def observation_spec(self) -> dict[str, Any]:
        return deepcopy(self.codec.observation_spec())

    def action_spec(self) -> dict[str, Any]:
        return deepcopy(self.mapper.action_spec())

    def valid_actions(self, state: Any | None = None) -> list[Action] | None:
        self._require_state()
        if state is not None:
            raise EnvironmentError("Unity action enumeration supports only the current state.")
        try:
            return [self.mapper.from_index(i, deepcopy(self._observation)) for i in self._enabled]
        except Exception as exc:
            raise EnvironmentError(f"Unity action enumeration failed: {exc}") from exc

    def step(self, action: Action) -> StepResult:
        self._require_state()
        try:
            index = self.mapper.to_index(action, deepcopy(self._observation))
        except (ValueError, KeyError, TypeError) as exc:
            return self._result(invalid_reason=str(exc))
        except Exception as exc:
            raise self._abort("action mapping", exc) from exc
        if type(index) is not int or not 0 <= index < self._action_count or index not in self._enabled:
            return self._result(invalid_reason="Invalid, disabled or terminal Unity action.")
        self._ready = False
        try:
            from mlagents_envs.base_env import ActionTuple

            actions = ActionTuple(discrete=np.array([[index]], dtype=np.int32))
            self._env.set_actions(self._behavior_name, actions)
            self._env.step()
            self._await_steps()
        except Exception as exc:
            raise self._abort("step", exc) from exc
        return self._result()

    def _result(self, invalid_reason: str | None = None) -> StepResult:
        return StepResult(
            next_observation=deepcopy(self._observation),
            reward_signals={"unity": self._reward} if invalid_reason is None else {},
            game_terminal=self._terminal and not self._interrupted,
            test_boundary_reached=self._interrupted,
            invalid_reason=invalid_reason,
            state_signature=(
                f"{self.codec.canonicalization_version}:sha256:{sha256(self._canonical).hexdigest()}"
                if self._canonical is not None else None
            ),
            game_outcome=self._outcome,
        )

    def canonical_state(self) -> bytes | None:
        self._require_state()
        return self._canonical

    def goal_test(self, state: Any | None = None) -> bool | None:
        self._require_state()
        if state is not None:
            raise EnvironmentError("Unity goal queries support only the current state.")
        return None if self._outcome is None else self._outcome == "SUCCESS"

    def clone_state(self) -> Any | None:
        return None

    def restore_state(self, handle: Any) -> None:
        raise EnvironmentError("Unity state restoration is unsupported.")

    def events(self) -> list[Any]:
        return []

    def close(self) -> None:
        env, self._env = self._env, None
        self._closed = True
        self._ready = False
        self._observation = self._canonical = None
        self._enabled = []
        self.decision_steps = self.terminal_steps = None
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                raise EnvironmentError(f"Unity close failed: {exc}") from exc
