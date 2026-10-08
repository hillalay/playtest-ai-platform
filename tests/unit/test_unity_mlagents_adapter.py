import json
import subprocess
import sys
from collections import deque
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from playtest.adapters.unity_codecs import DiscreteActionMapper, RawObservationCodec
from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.core.actions import Action
from playtest.core.errors import EnvironmentError


def action(index):
    return Action("unity_discrete", {"index": index})


class Steps:
    def __init__(self, ids=(17,), mask=None, values=(1.25, 2.5), interrupted=False, reward=0.5):
        self.agent_id = np.array(ids, dtype=np.int32)
        self.action_mask = mask
        self.obs = [np.tile(np.array(values, dtype=np.float32), (len(ids), 1))]
        self.reward = np.full(len(ids), reward, dtype=np.float32)
        self.interrupted = np.full(len(ids), interrupted, dtype=bool)

    def __len__(self):
        return len(self.agent_id)


def empty():
    return Steps(ids=())


def behavior(branches=(5,), continuous=0, shapes=((2,),)):
    return SimpleNamespace(
        action_spec=SimpleNamespace(continuous_size=continuous, discrete_branches=np.array(branches)),
        observation_specs=[SimpleNamespace(shape=shape) for shape in shapes],
    )


class FakeUnity:
    def __init__(self):
        self.kwargs = None
        self.behavior_specs = {"Game": behavior()}
        self.decision = Steps(mask=[np.array([[True, True, False, False, True]])])
        self.terminal = empty()
        self.other_steps = (empty(), empty())
        self.frames = deque()
        self.sent = []
        self.close_count = self.step_count = self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def get_steps(self, name):
        return (self.decision, self.terminal) if name == "Game" else self.other_steps

    def set_actions(self, name, actions):
        assert name == "Game"
        self.sent.append(actions.discrete.copy())

    def step(self):
        self.step_count += 1
        if self.frames:
            self.decision, self.terminal = self.frames.popleft()

    def close(self):
        self.close_count += 1


@pytest.fixture
def unity(monkeypatch):
    env = FakeUnity()
    package = ModuleType("mlagents_envs")
    environment = ModuleType("mlagents_envs.environment")
    base = ModuleType("mlagents_envs.base_env")

    def connect(**kwargs):
        env.kwargs = kwargs
        return env

    environment.UnityEnvironment = connect
    base.ActionTuple = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in [("mlagents_envs", package),
                         ("mlagents_envs.environment", environment),
                         ("mlagents_envs.base_env", base)]:
        monkeypatch.setitem(sys.modules, name, module)
    return env


@pytest.fixture
def bridge(unity):
    adapter = UnityMLAgentsAdapter()
    adapter.reset()
    yield adapter, unity
    adapter.close()


def test_lazy_connection_and_reset_observation(unity):
    adapter = UnityMLAgentsAdapter(file_name=None, timeout_wait=45)
    assert unity.kwargs is None
    adapter.load_level("static")
    observation = adapter.reset()
    assert unity.kwargs == {"file_name": None, "timeout_wait": 45}
    assert unity.reset_count == 1
    assert observation == {"observations": [[1.25, 2.5]]}
    observation["observations"][0][0] = 99
    assert adapter.step(action(0)).next_observation == {"observations": [[1.25, 2.5]]}
    assert "observations" in adapter.observation_spec()
    assert "unity_discrete" in adapter.action_spec()
    adapter.close()


@pytest.mark.parametrize("names", [[], ["Game", "Other"]])
def test_behavior_requires_unambiguous_selection(unity, names):
    unity.behavior_specs = dict.fromkeys(names, behavior())
    with pytest.raises(EnvironmentError, match="exactly one.*behavior_name"):
        UnityMLAgentsAdapter().reset()
    assert unity.close_count == 1


def test_explicit_behavior_selection(unity):
    unity.behavior_specs["Other"] = behavior()
    adapter = UnityMLAgentsAdapter(behavior_name="Game")
    adapter.reset()
    adapter.step(action(2))
    assert unity.sent
    adapter.close()


def test_missing_selected_behavior(unity):
    with pytest.raises(EnvironmentError, match="'Missing'.*missing.*Game"):
        UnityMLAgentsAdapter(behavior_name="Missing").reset()
    assert unity.close_count == 1


def test_other_active_behavior_is_rejected_before_implicit_zero_actions(unity):
    unity.behavior_specs["Other"] = behavior()
    unity.other_steps = (Steps(), empty())
    with pytest.raises(EnvironmentError, match="Multiple active behaviors"):
        UnityMLAgentsAdapter(behavior_name="Game").reset()
    assert not unity.sent and unity.step_count == 0
    assert unity.close_count == 1


@pytest.mark.parametrize("branches,continuous", [((2, 3), 0), ((), 1), ((5,), 1), ((), 0)])
def test_unsupported_action_spaces(unity, branches, continuous):
    unity.behavior_specs["Game"] = behavior(branches, continuous)
    with pytest.raises(EnvironmentError, match="one discrete branch"):
        UnityMLAgentsAdapter().reset()
    assert unity.close_count == 1


def test_empty_branch_is_rejected(unity):
    unity.behavior_specs["Game"] = behavior((0,))
    with pytest.raises(EnvironmentError, match="must contain actions"):
        UnityMLAgentsAdapter().reset()


@pytest.mark.parametrize("mask,expected", [
    (None, [0, 1, 2, 3, 4]),
    ([np.array([[True, True, False, False, True]])], [2, 3]),
    ([np.ones((1, 5), dtype=bool)], []),
])
def test_mask_true_means_disabled(unity, mask, expected):
    unity.decision.action_mask = mask
    adapter = UnityMLAgentsAdapter()
    adapter.reset()
    assert adapter.valid_actions() == [action(i) for i in expected]
    if not expected:
        assert adapter.step(action(0)).invalid_reason
        assert not unity.sent and unity.step_count == 0
    adapter.close()


@pytest.mark.parametrize("mask", [[], [np.ones((1, 4), dtype=bool)],
                                  [np.ones((1, 5), dtype=int)],
                                  [np.ones((1, 5), dtype=bool)] * 2])
def test_malformed_masks_fail_on_refresh(unity, mask):
    unity.decision.action_mask = mask
    with pytest.raises(EnvironmentError, match="mask"):
        UnityMLAgentsAdapter().reset()
    assert unity.close_count == 1


@pytest.mark.parametrize("invalid", [Action("unknown"), action(0), action(-1), action(5),
                                     action(True), action(2.0),
                                     Action("unity_discrete", {"index": 2, "extra": 0}), None])
def test_invalid_action_never_reaches_unity(bridge, invalid):
    adapter, env = bridge
    result = adapter.step(invalid)
    assert result.invalid_reason
    assert result.reward_signals == {}
    assert result.next_observation == {"observations": [[1.25, 2.5]]}
    assert not env.sent and env.step_count == 0 and env.reset_count == 1


def test_semantic_mapping_does_not_assume_entity_id_equals_index(unity):
    mapping = {i: Action("select_arrow", {"arrow_id": entity})
               for i, entity in enumerate([90, 12, 721, 48, 333])}
    mapper = DiscreteActionMapper(mapping)
    mapping[2].params["arrow_id"] = -1
    adapter = UnityMLAgentsAdapter(action_mapper=mapper)
    adapter.reset()
    assert adapter.valid_actions() == [Action("select_arrow", {"arrow_id": 721}),
                                       Action("select_arrow", {"arrow_id": 48})]
    adapter.valid_actions()[0].params["arrow_id"] = -2
    assert adapter.step(Action("select_arrow", {"arrow_id": 2})).invalid_reason
    assert adapter.step(Action("select_arrow", {"arrow_id": 90})).invalid_reason
    assert not unity.sent
    adapter.step(Action("select_arrow", {"arrow_id": 721}))
    np.testing.assert_array_equal(unity.sent[0], [[2]])
    assert unity.sent[0].dtype == np.int32
    assert adapter.action_spec()["index_to_action"][2].params == {"arrow_id": 721}
    adapter.close()


@pytest.mark.parametrize("mapping", [{0: Action("a")}, {i: Action("same") for i in range(5)},
                                      {i: "bad" for i in range(5)}])
def test_incomplete_or_ambiguous_mapping_fails_during_initialization(unity, mapping):
    with pytest.raises(EnvironmentError, match="mapping"):
        UnityMLAgentsAdapter(action_mapper=DiscreteActionMapper(mapping)).reset()
    assert unity.close_count == 1 and not unity.sent


def test_step_refreshes_observation_and_mask_with_defensive_copies(bridge):
    adapter, env = bridge
    env.frames.append((Steps(values=(9.5, 4.25), mask=[np.array([[False] * 5])]), empty()))
    result = adapter.step(action(2))
    assert env.step_count == 1
    assert result.next_observation == {"observations": [[9.5, 4.25]]}
    assert result.reward_signals == {"unity": 0.5}
    result.next_observation["observations"][0][0] = -1
    env.decision.obs[0][0][0] = -2
    env.decision.action_mask[0][:] = True
    assert adapter.valid_actions() == [action(i) for i in range(5)]
    assert adapter.step(action(-1)).next_observation == {"observations": [[9.5, 4.25]]}
    assert adapter.canonical_state() is None
    assert result.state_signature is None


def test_multiple_observation_tensors_need_no_game_specific_layout(unity):
    unity.behavior_specs["Game"] = behavior(shapes=((2,), (2, 2, 1)))
    unity.decision.obs.append(np.ones((1, 2, 2, 1), dtype=np.float32))
    adapter = UnityMLAgentsAdapter()
    assert adapter.reset()["observations"] == [[1.25, 2.5], [[[1.0], [1.0]], [[1.0], [1.0]]]]
    adapter.close()


def test_initial_reset_waits_without_sending_actions(unity):
    unity.decision = empty()
    unity.frames.extend([(empty(), empty()), (Steps(), empty())])
    adapter = UnityMLAgentsAdapter(max_wait_steps=2)
    assert adapter.reset() == {"observations": [[1.25, 2.5]]}
    assert unity.step_count == 2 and not unity.sent
    adapter.close()


def test_delayed_decision_after_action_does_not_resend_or_reset(bridge):
    adapter, env = bridge
    env.frames.extend([(empty(), empty()), (empty(), empty()), (Steps(values=(8, 3)), empty())])
    result = adapter.step(action(2))
    assert result.next_observation == {"observations": [[8.0, 3.0]]}
    assert len(env.sent) == 1 and env.step_count == 3 and env.reset_count == 1


@pytest.mark.parametrize("during_reset", [True, False])
def test_wait_limit_closes_instead_of_using_stale_state(unity, during_reset):
    adapter = UnityMLAgentsAdapter(max_wait_steps=2)
    if during_reset:
        unity.decision = empty()
        operation = adapter.reset
    else:
        adapter.reset()
        unity.frames.append((empty(), empty()))
        operation = lambda: adapter.step(action(2))
    with pytest.raises(EnvironmentError, match="after 2 waiting steps"):
        operation()
    assert unity.step_count == (2 if during_reset else 3)
    assert unity.close_count == 1
    with pytest.raises(EnvironmentError):
        adapter.valid_actions()


@pytest.mark.parametrize("interrupted", [False, True])
def test_delayed_terminal_preserves_last_observation_and_unknown_outcome(bridge, interrupted):
    adapter, env = bridge
    env.frames.extend([(empty(), empty()), (empty(), Steps(values=(7, 0), interrupted=interrupted))])
    result = adapter.step(action(2))
    assert result.next_observation == {"observations": [[7.0, 0.0]]}
    assert result.game_terminal is (not interrupted)
    assert result.test_boundary_reached is interrupted
    assert result.game_outcome is None and adapter.goal_test() is None
    assert result.reward_signals == {"unity": 0.5}
    assert adapter.valid_actions() == []
    calls = env.step_count
    assert adapter.step(action(2)).invalid_reason
    assert env.step_count == calls and env.reset_count == 1 and len(env.sent) == 1


def test_terminal_takes_priority_over_same_agent_decision(bridge):
    adapter, env = bridge
    env.frames.append((Steps(values=(99, 99)), Steps(values=(7, 0))))
    result = adapter.step(action(2))
    assert result.game_terminal
    assert result.next_observation == {"observations": [[7.0, 0.0]]}
    assert adapter.valid_actions() == []


def test_terminal_on_reset_does_not_advance_into_new_episode(unity):
    unity.decision, unity.terminal = empty(), Steps(values=(7, 0))
    adapter = UnityMLAgentsAdapter()
    assert adapter.reset() == {"observations": [[7.0, 0.0]]}
    assert adapter.valid_actions() == [] and unity.step_count == 0
    adapter.close()


@pytest.mark.parametrize("decision,terminal", [
    (Steps(ids=(17, 18)), empty()),
    (empty(), Steps(ids=(17, 18))),
    (Steps(ids=(18,)), Steps(ids=(17,))),
])
def test_multiple_agents_are_rejected(unity, decision, terminal):
    unity.decision, unity.terminal = decision, terminal
    with pytest.raises(EnvironmentError, match="Multiple Unity agents"):
        UnityMLAgentsAdapter().reset()
    assert unity.close_count == 1


def test_changed_agent_cannot_silently_become_next_episode(bridge):
    adapter, env = bridge
    env.frames.append((Steps(ids=(18,)), empty()))
    with pytest.raises(EnvironmentError, match="agent changed"):
        adapter.step(action(2))
    assert env.close_count == 1


class EntityCodec(RawObservationCodec):
    # This fake protocol's complete state is two unordered entity/value pairs.
    canonicalization_version = "test-entities-v1"

    def decode(self, observations):
        return {"entities": sorted(observations[0].reshape(-1, 2).tolist())}

    def canonical_state(self, observation):
        return json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_canonical_identity_is_stable_versioned_and_opt_in(unity):
    unity.behavior_specs["Game"] = behavior(shapes=((4,),))
    unity.decision = Steps(values=(90, 1, 12, 0))
    adapter = UnityMLAgentsAdapter(observation_codec=EntityCodec())
    observation = adapter.reset()
    canonical = adapter.canonical_state()
    assert canonical == b'{"entities":[[12.0,0.0],[90.0,1.0]]}'
    observation["entities"][0][0] = -1
    unity.frames.append((Steps(values=(12, 0, 90, 1)), empty()))
    result = adapter.step(action(2))
    assert adapter.canonical_state() == canonical
    signature = result.state_signature
    assert signature.startswith("test-entities-v1:sha256:")
    unity.decision = Steps(values=(12, 0, 90, 1))
    adapter.reset()
    assert adapter.canonical_state() == canonical
    unity.frames.append((Steps(values=(12, 0, 90, 0)), empty()))
    assert adapter.step(action(2)).state_signature != signature
    assert adapter.capabilities()["canonical_state"] is True
    adapter.close()


class OutcomeCodec(RawObservationCodec):
    # A fake game protocol explicitly reports 0=unknown, 1=success, -1=failure.
    goal_semantics = True

    def game_outcome(self, observation):
        return {0: None, 1: "SUCCESS", -1: "FAILURE"}[observation["observations"][0][1]]


@pytest.mark.parametrize("signal,outcome,goal", [(0, None, None), (1, "SUCCESS", True), (-1, "FAILURE", False)])
def test_outcomes_require_explicit_codec_protocol(unity, signal, outcome, goal):
    unity.decision = Steps(values=(7, 0))
    adapter = UnityMLAgentsAdapter(observation_codec=OutcomeCodec())
    adapter.reset()
    unity.frames.append((empty(), Steps(values=(7, signal))))
    result = adapter.step(action(2))
    assert result.game_terminal and result.game_outcome == outcome
    assert adapter.goal_test() is goal
    assert adapter.capabilities()["goal_semantics"] is True
    adapter.close()


def test_unsupported_capabilities_are_explicit(bridge):
    adapter, env = bridge
    caps = adapter.capabilities()
    assert caps["valid_actions"] is True
    assert all(caps[key] is False for key in ["canonical_state", "clone_restore", "goal_semantics",
                                            "level_loading", "seed", "deterministic", "transition_model",
                                            "event_instrumentation", "parallel_safe", "headless"])
    assert caps["canonicalization_version"] is None
    for operation in [lambda: adapter.load_level("other"), lambda: adapter.load_level("static", seed=1),
                      lambda: adapter.reset(seed=42), lambda: adapter.valid_actions({}),
                      lambda: adapter.goal_test({}), lambda: adapter.restore_state({})]:
        with pytest.raises(EnvironmentError):
            operation()
    assert env.reset_count == 1
    assert adapter.canonical_state() is None and adapter.clone_state() is None
    assert adapter.events() == []


@pytest.mark.parametrize("operation", ["reset", "get_steps", "set_actions", "step"])
def test_transport_failures_are_technical_errors_and_close(bridge, monkeypatch, operation):
    adapter, env = bridge

    def disconnected(*args):
        raise RuntimeError("disconnected")

    monkeypatch.setattr(env, operation, disconnected)
    with pytest.raises(EnvironmentError, match="disconnected") as error:
        adapter.reset() if operation == "reset" else adapter.step(action(2))
    assert error.value.__cause__ is not None
    assert env.close_count == 1
    with pytest.raises(EnvironmentError):
        adapter.canonical_state()


def test_constructor_failure_is_wrapped_and_close_remains_safe(unity, monkeypatch):
    def fail(**kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(sys.modules["mlagents_envs.environment"], "UnityEnvironment", fail)
    adapter = UnityMLAgentsAdapter()
    with pytest.raises(EnvironmentError, match="connection refused"):
        adapter.reset()
    adapter.close()
    assert unity.close_count == 0


def test_missing_optional_dependency_has_actionable_error(unity, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlagents_envs.environment", None)
    with pytest.raises(EnvironmentError, match="Install mlagents-envs==1.1.0"):
        UnityMLAgentsAdapter().reset()


def test_close_is_idempotent_and_final(bridge):
    adapter, env = bridge
    adapter.close()
    adapter.close()
    assert env.close_count == 1
    for operation in [adapter.reset, adapter.valid_actions, lambda: adapter.step(action(2)),
                      lambda: adapter.load_level("static")]:
        with pytest.raises(EnvironmentError):
            operation()


def test_close_before_connection_never_creates_environment(unity):
    adapter = UnityMLAgentsAdapter()
    adapter.close()
    adapter.close()
    assert unity.kwargs is None


def test_cleanup_error_does_not_hide_original_failure(bridge, monkeypatch):
    adapter, env = bridge

    def bad_close():
        env.close_count += 1
        raise OSError("close failed")

    monkeypatch.setattr(env, "close", bad_close)
    monkeypatch.setattr(env, "step", lambda: (_ for _ in ()).throw(RuntimeError("disconnect")))
    with pytest.raises(EnvironmentError, match="disconnect.*cleanup also failed.*close failed"):
        adapter.step(action(2))
    adapter.close()
    assert env.close_count == 1


@pytest.mark.parametrize("values", [(float("nan"), 1), (float("inf"), 1), (1, 2, 3)])
def test_malformed_observations_fail_and_close(unity, values):
    unity.decision = Steps(values=values)
    with pytest.raises(EnvironmentError, match="observation"):
        UnityMLAgentsAdapter().reset()
    assert unity.close_count == 1


def test_false_canonical_capability_is_rejected(unity):
    class BadCodec(RawObservationCodec):
        canonicalization_version = "v1"

    with pytest.raises(EnvironmentError, match="must return bytes"):
        UnityMLAgentsAdapter(observation_codec=BadCodec()).reset()
    assert unity.close_count == 1


def test_unexpected_mapper_failure_is_technical_and_closes(bridge, monkeypatch):
    adapter, env = bridge

    def fail(*args):
        raise RuntimeError("mapping protocol failed")

    monkeypatch.setattr(adapter.mapper, "to_index", fail)
    with pytest.raises(EnvironmentError, match="action mapping.*protocol failed"):
        adapter.step(action(2))
    assert env.close_count == 1 and not env.sent


def test_close_failure_is_reported_once(bridge, monkeypatch):
    adapter, env = bridge

    def fail():
        env.close_count += 1
        raise OSError("disconnected while closing")

    monkeypatch.setattr(env, "close", fail)
    with pytest.raises(EnvironmentError, match="Unity close failed.*disconnected"):
        adapter.close()
    adapter.close()
    assert env.close_count == 1


@pytest.mark.parametrize("kwargs", [{"max_wait_steps": -1}, {"max_wait_steps": True},
                                    {"timeout_wait": 0}, {"timeout_wait": 1.5}])
def test_invalid_wait_configuration_fails_before_connecting(unity, kwargs):
    with pytest.raises(ValueError):
        UnityMLAgentsAdapter(**kwargs)
    assert unity.kwargs is None


def test_adapter_import_does_not_require_mlagents():
    code = "import sys; sys.modules['mlagents_envs'] = None; from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter; assert UnityMLAgentsAdapter().capabilities()['canonical_state'] is False"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_existing_runner_and_random_policy_preserve_unknown_terminal(unity):
    from playtest.orchestration.runner import EpisodeRunner
    from playtest.policies.random import RandomPolicy

    unity.frames.append((empty(), Steps(values=(0, 0))))
    adapter = UnityMLAgentsAdapter()
    result = EpisodeRunner().run(adapter, RandomPolicy(seed=42), max_steps=3)
    assert result.outcome == "GAME_TERMINAL" and result.steps == 1
    assert result.trace.steps[0].state_signature is None
    assert unity.reset_count == 1
    adapter.close()


def test_manual_smoke_stops_at_terminal_then_explicitly_resets_and_closes(unity, capsys):
    from scripts.unity_adapter_smoke import main

    unity.frames.append((empty(), Steps(values=(0, 0))))
    main(["--steps", "3"])
    assert unity.reset_count == 2 and unity.step_count == 1 and unity.close_count == 1
    assert len(unity.sent) == 1
    assert "PASS: adapter" in capsys.readouterr().out


def test_manual_smoke_reports_no_actions_without_sending_one(unity, capsys):
    from scripts.unity_adapter_smoke import main

    unity.decision.action_mask = [np.ones((1, 5), dtype=bool)]
    main([])
    assert unity.reset_count == 2 and unity.close_count == 1
    assert not unity.sent and unity.step_count == 0
    assert "No legal actions" in capsys.readouterr().out
