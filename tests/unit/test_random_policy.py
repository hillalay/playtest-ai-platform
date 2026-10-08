from copy import deepcopy

import pytest

from playtest.core.actions import Action
from playtest.policies.random import RandomPolicy


def test_random_policy_returns_valid_action():

    policy = RandomPolicy(seed=42)

    valid_actions = [
        Action(type="increment"),
        Action(type="decrement"),
    ]

    action = policy.select_action(
        observation={"value": 0},
        valid_actions=valid_actions,
    )

    assert action in valid_actions


def test_random_policy_is_reproducible_with_same_seed():

    actions = [
        Action(type="increment"),
        Action(type="decrement"),
    ]

    policy_a = RandomPolicy(seed=42)
    policy_b = RandomPolicy(seed=42)

    result_a = [
        policy_a.select_action({}, actions)
        for _ in range(10)
    ]

    result_b = [
        policy_b.select_action({}, actions)
        for _ in range(10)
    ]

    assert result_a == result_b


def test_random_policy_rejects_empty_actions():
    with pytest.raises(ValueError, match="No valid actions"):
        RandomPolicy(seed=42).select_action({}, [])


def test_random_policy_does_not_mutate_observations_or_actions():
    observation = {"nested": {"value": [1, 2]}}
    actions = [Action("select", {"entity": {"id": 90}}), Action("select", {"entity": {"id": 12}})]
    before = deepcopy((observation, actions))
    policy = RandomPolicy(seed=42)
    for _ in range(20):
        assert policy.select_action(observation, actions) in actions
    assert (observation, actions) == before
