import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.core.actions import Action
from playtest.core.errors import EnvironmentError


class Steps:
    def __init__(self, count=1, mask=None, arrows=None, interrupted=False):
        self.count = count
        self.action_mask = mask
        self.obs = [np.array([arrows or [0, 1, 2, 3, 1]], dtype=np.float32)]
        self.reward = np.array([0.5])
        self.interrupted = np.array([interrupted])

    def __len__(self):
        return self.count


class FakeUnity:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.behavior_specs = {"Arrow": SimpleNamespace(action_spec=SimpleNamespace(
            continuous_size=0, discrete_branches=np.array([5])))}
        self.decision = Steps(mask=[np.array([[True, True, False, False, True]])])
        self.terminal = Steps(count=0)
        self.sent = []
        self.close_count = 0
        self.step_count = 0

    def reset(self):
        pass

    def get_steps(self, name):
        assert name == "Arrow"
        return self.decision, self.terminal

    def set_actions(self, name, actions):
        self.sent.append(actions.discrete)

    def step(self):
        self.step_count += 1

    def close(self):
        self.close_count += 1


@pytest.fixture
def bridge(monkeypatch):
    env = FakeUnity()
    package = ModuleType("mlagents_envs")
    environment = ModuleType("mlagents_envs.environment")
    base = ModuleType("mlagents_envs.base_env")
    environment.UnityEnvironment = lambda **kwargs: env
    base.ActionTuple = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in [("mlagents_envs", package),
                         ("mlagents_envs.environment", environment),
                         ("mlagents_envs.base_env", base)]:
        monkeypatch.setitem(sys.modules, name, module)
    adapter = UnityMLAgentsAdapter()
    adapter.reset()
    yield adapter, env
    adapter.close()


def test_mask_and_unmasked_actions(bridge):
    adapter, env = bridge
    assert adapter.valid_actions() == [Action("select_arrow", {"action_id": i}) for i in [2, 3]]
    env.decision.action_mask = None
    assert len(adapter.valid_actions()) == 5
    env.decision.action_mask = [np.ones((1, 5), dtype=bool)]
    assert adapter.valid_actions() == []
    env.decision.action_mask = [np.ones((1, 4), dtype=bool)]
    with pytest.raises(EnvironmentError, match="mask"):
        adapter.valid_actions()


@pytest.mark.parametrize("action", [Action("unknown"), Action("select_arrow", {"action_id": 0}),
                                    Action("select_arrow", {"action_id": True}),
                                    Action("select_arrow", {"action_id": 2.0})])
def test_invalid_action_does_not_reach_unity(bridge, action):
    adapter, env = bridge
    before = adapter.canonical_state()
    assert adapter.step(action).invalid_reason
    assert not env.sent and env.step_count == 0
    assert adapter.canonical_state() == before


def test_step_refreshes_and_canonicalizes_without_numpy_leak(bridge):
    adapter, env = bridge
    env.decision = Steps(arrows=[2, 0, 1, 3, 1, 1, 2, 0, 0, 0])
    result = adapter.step(Action("select_arrow", {"action_id": 2}))
    np.testing.assert_array_equal(env.sent[0], np.array([[2]], dtype=np.int32))
    assert env.sent[0].dtype == np.int32
    assert env.step_count == 1
    assert result.next_observation == {"arrows": [[1, 2, 0, 0, 0], [2, 0, 1, 3, 1]]}
    signature = adapter.canonical_state()
    result.next_observation["arrows"][0][0] = 99
    assert adapter.canonical_state() == signature
    env.decision = Steps(arrows=[1, 2, 0, 0, 0, 2, 0, 1, 3, 1])
    adapter.reset()
    assert adapter.canonical_state() == signature


@pytest.mark.parametrize("interrupted", [False, True])
def test_terminal_never_implies_success(bridge, interrupted):
    adapter, env = bridge
    env.decision = Steps(count=0)
    env.terminal = Steps(interrupted=interrupted)
    result = adapter.step(Action("select_arrow", {"action_id": 2}))
    assert result.game_terminal is (not interrupted)
    assert result.test_boundary_reached is interrupted
    assert result.game_outcome is None
    assert adapter.goal_test() is None
    assert adapter.valid_actions() == []
    assert result.reward_signals == {"unity": 0.5}


def test_close_is_idempotent(bridge):
    adapter, env = bridge
    adapter.close()
    adapter.close()
    assert env.close_count == 1
    with pytest.raises(EnvironmentError, match="closed"):
        adapter.reset()


@pytest.mark.parametrize("names", [[], ["Arrow", "Other"]])
def test_behavior_selection_failure_closes_environment(bridge, names):
    adapter, env = bridge
    spec = env.behavior_specs["Arrow"]
    env.behavior_specs = dict.fromkeys(names, spec)
    with pytest.raises(EnvironmentError, match="exactly one"):
        adapter.reset()
    assert env.close_count == 1


def test_unsupported_operations_are_explicit(bridge):
    adapter, _ = bridge
    adapter.load_level("static")
    for operation in [lambda: adapter.load_level("other"), lambda: adapter.reset(seed=42),
                      lambda: adapter.valid_actions({}), lambda: adapter.restore_state({})]:
        with pytest.raises(EnvironmentError):
            operation()
    assert adapter.clone_state() is None
    assert adapter.events() == []


@pytest.mark.parametrize("branches,continuous", [([2, 3], 0), ([], 1), ([5], 1)])
def test_unsupported_action_spaces(bridge, branches, continuous):
    adapter, env = bridge
    env.behavior_specs["Arrow"].action_spec = SimpleNamespace(
        continuous_size=continuous, discrete_branches=np.array(branches))
    with pytest.raises(EnvironmentError, match="one discrete branch"):
        adapter.reset()


@pytest.mark.parametrize("arrows", [[0, 1, 2], [0, 1, 2, 3, float("nan")],
                                    [0, 1, 2, 3, 0.5], [0, 1, 2, 3, 2],
                                    [0, 1, 2, 3, 1, 0, 2, 3, 1, 0]])
def test_malformed_observations_fail_explicitly(bridge, arrows):
    adapter, env = bridge
    env.decision = Steps(arrows=arrows)
    with pytest.raises(EnvironmentError):
        adapter.reset()
    assert env.close_count == 1
