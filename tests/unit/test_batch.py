import os
import subprocess
import sys
from copy import deepcopy

import pytest

from playtest.core.actions import Action
from playtest.core.results import EpisodeResult, StepResult
from playtest.orchestration.batch import BatchRunner, action_sequence_signature
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.random import RandomPolicy
from playtest.replay.trace import EpisodeTrace


class FakeAdapter:
    """A counter fixture with legal semantic actions, unrelated to any game engine."""

    def __init__(self, *, after=3, outcome="GAME_TERMINAL", actions=None, fail_on_step=None):
        self.after = after
        self.outcome = outcome
        self.actions = actions if actions is not None else [Action("pick", {"slot": i}) for i in range(3)]
        self.fail_on_step = fail_on_step
        self.reset_seeds = []
        self.sent = []
        self.close_count = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return {"counter": 0}

    def valid_actions(self):
        if self.outcome == "NO_VALID_ACTIONS":
            return []
        return self.actions

    def step(self, action):
        assert action in self.actions
        self.sent.append(deepcopy(action))
        if len(self.sent) == self.fail_on_step:
            raise OSError("disconnected")
        self.steps += 1
        end = self.after is not None and self.steps >= self.after
        return StepResult(
            {"counter": self.steps},
            game_terminal=end and self.outcome in {"GAME_TERMINAL", "SUCCESS", "FAILURE"},
            test_boundary_reached=end and self.outcome == "TEST_BOUNDARY_REACHED",
            game_outcome=self.outcome if end and self.outcome in {"SUCCESS", "FAILURE"} else None,
        )

    def close(self):
        self.close_count += 1


def random_factory(index, policy_seed):
    return RandomPolicy(seed=policy_seed)


@pytest.mark.parametrize("episodes", [1, 100])
def test_batch_reuses_runner_and_one_caller_owned_adapter(monkeypatch, episodes):
    adapter = FakeAdapter()
    runs = []
    original = EpisodeRunner.run

    def tracked(self, environment, policy, max_steps=100, seed=None):
        runs.append((environment, seed))
        return original(self, environment, policy, max_steps=max_steps, seed=seed)

    monkeypatch.setattr(EpisodeRunner, "run", tracked)
    batch = BatchRunner().run(adapter, random_factory, episodes=episodes, max_steps=5, base_seed=42)
    assert batch.requested_episodes == batch.attempted_episodes == batch.completed_episodes == episodes
    assert batch.total_steps == len(adapter.sent) == 3 * episodes
    assert batch.outcomes == {"GAME_TERMINAL": episodes} and batch.technical_errors == 0
    assert batch.stop_reason == "EPISODES_COMPLETED" and batch.stopped_episode is None
    assert runs == [(adapter, None)] * episodes
    assert adapter.reset_seeds == [None] * episodes and adapter.close_count == 0
    assert all(record.result.trace.seed is None for record in batch.episode_results)
    assert all(record.trace_validated and record.action_sequence_signature for record in batch.episode_results)
    assert batch.diversity_episodes == episodes and batch.diversity_excluded_episodes == 0
    assert batch.average_episode_seconds >= 0 and batch.duration_seconds >= 0
    adapter.close()
    assert adapter.close_count == 1


def test_same_base_seed_repeats_plan_and_action_sequences():
    batches = [BatchRunner().run(FakeAdapter(), random_factory, episodes=20, base_seed=42) for _ in range(2)]
    first, second = batches
    assert [r.policy_seed for r in first.episode_results] == list(range(42, 62))
    assert [r.policy_seed for r in first.episode_results] == [r.policy_seed for r in second.episode_results]
    assert [r.action_sequence_signature for r in first.episode_results] == [r.action_sequence_signature for r in second.episode_results]
    assert first.unique_action_sequences == second.unique_action_sequences
    assert first.unique_action_sequences > 1  # Evidence for this fixture, not a guarantee for other games.


def test_generated_base_seed_is_recorded_and_can_be_reused(monkeypatch):
    monkeypatch.setattr("playtest.orchestration.batch.secrets.randbits", lambda bits: 987654)
    generated = BatchRunner().run(FakeAdapter(), random_factory, episodes=3)
    repeated = BatchRunner().run(FakeAdapter(), random_factory, episodes=3, base_seed=generated.base_seed)
    assert generated.base_seed == 987654
    assert [r.policy_seed for r in generated.episode_results] == [987654, 987655, 987656]
    assert [r.action_sequence_signature for r in generated.episode_results] == [r.action_sequence_signature for r in repeated.episode_results]


def test_factory_supports_other_policies_and_duplicates_are_counted_once():
    sequences = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
    calls = []

    class ScriptedPolicy:
        def __init__(self, sequence):
            self.sequence = iter(sequence)

        def select_action(self, observation, valid_actions):
            return valid_actions[next(self.sequence)]

    def factory(index, policy_seed):
        calls.append((index, policy_seed))
        return ScriptedPolicy(sequences[index])

    batch = BatchRunner().run(FakeAdapter(), factory, episodes=3, base_seed=42)
    assert calls == [(0, 42), (1, 43), (2, 44)]
    signatures = [r.action_sequence_signature for r in batch.episode_results]
    assert signatures[0] == signatures[2] and signatures[0] != signatures[1]
    assert batch.unique_action_sequences == 2 and batch.diversity_supported


def test_distinct_seeds_are_not_a_guarantee_of_distinct_paths():
    batch = BatchRunner().run(
        FakeAdapter(actions=[Action("only_choice")]), random_factory, episodes=10, base_seed=42,
    )
    assert len({r.policy_seed for r in batch.episode_results}) == 10
    assert batch.unique_action_sequences == 1


def test_fingerprint_ignores_dictionary_order_and_handles_nested_params():
    a = Action("select", {"entity": {"id": 90, "coords": [1, 2]}, "flags": [True, None, {"b": 2, "a": "ok"}]})
    b = Action("select", {"flags": [True, None, {"a": "ok", "b": 2}], "entity": {"coords": [1, 2], "id": 90}})
    signature = action_sequence_signature([a])
    assert signature == action_sequence_signature([b])
    assert signature.startswith("action-sequence-v1:sha256:")
    assert len(signature.rsplit(":", 1)[1]) == 64


def test_fingerprint_distinguishes_type_values_nesting_and_action_order():
    actions = [
        Action("pick", {"id": 90}), Action("other", {"id": 90}),
        Action("pick", {"id": 12}), Action("pick", {"entity": {"id": 90}}),
        Action("pick", {"id": "90"}), Action("pick", {"id": [90]}),
    ]
    assert len({action_sequence_signature([a]) for a in actions}) == len(actions)
    assert action_sequence_signature(actions) != action_sequence_signature(list(reversed(actions)))
    assert action_sequence_signature([]) != action_sequence_signature([Action("noop")])


@pytest.mark.parametrize("params", [
    {"value": (1, 2)}, {"value": {1, 2}}, {"value": b"data"},
    {1: "non-string key"}, {"value": float("nan")}, {"value": float("inf")},
    {"value": float("-inf")}, {"nested": [{"key": object()}]},
])
def test_unsupported_params_do_not_produce_ambiguous_fingerprints(params):
    with pytest.raises(ValueError, match="Unsupported action sequence"):
        action_sequence_signature([Action("pick", params)])


def test_fingerprint_rejects_cycles_and_does_not_use_string_fallbacks():
    cycle = []
    cycle.append(cycle)

    class NoFallback:
        def __str__(self):
            raise AssertionError("str fallback forbidden")

        def __repr__(self):
            raise AssertionError("repr fallback forbidden")

    for value in (cycle, NoFallback()):
        with pytest.raises(ValueError, match="Unsupported action sequence"):
            action_sequence_signature([Action("pick", {"value": value})])


def test_fingerprint_is_stable_across_process_hash_seeds():
    code = (
        "from playtest.core.actions import Action; "
        "from playtest.orchestration.batch import action_sequence_signature; "
        "print(action_sequence_signature([Action('pick', {'nested': {'z': 2, 'a': [1, None]}})]))"
    )
    signatures = []
    for seed in ("1", "123"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, timeout=20)
        signatures.append(result.stdout.strip())
    assert signatures[0] == signatures[1]


def test_unsupported_diversity_is_explicit_and_not_a_fake_count():
    batch = BatchRunner().run(
        FakeAdapter(actions=[Action("pick", {"coords": (1, 2)})]), random_factory,
        episodes=2, base_seed=42,
    )
    assert batch.completed_episodes == 2 and batch.technical_errors == 0
    assert not batch.diversity_supported and batch.unique_action_sequences is None
    assert batch.diversity_episodes == 0 and batch.diversity_excluded_episodes == 2
    assert all(r.action_sequence_signature is None and r.diversity_error for r in batch.episode_results)


@pytest.mark.parametrize("outcome", [
    "GAME_TERMINAL", "SUCCESS", "FAILURE", "TEST_BOUNDARY_REACHED", "NO_VALID_ACTIONS", "STEP_LIMIT_REACHED",
])
def test_normal_stop_outcomes_remain_distinct(outcome):
    adapter = FakeAdapter(outcome=outcome)
    batch = BatchRunner().run(adapter, random_factory, episodes=2, max_steps=4, base_seed=42)
    assert batch.outcomes == {outcome: 2} and batch.completed_episodes == 2
    assert batch.technical_errors == 0 and batch.diversity_supported
    if outcome == "NO_VALID_ACTIONS":
        assert batch.total_steps == 0 and adapter.sent == []
        assert batch.unique_action_sequences == 1  # A reliable empty sequence.
        assert all(r.result.trace.steps == [] for r in batch.episode_results)
    if outcome == "GAME_TERMINAL":
        assert all(s.game_outcome is None for r in batch.episode_results for s in r.result.trace.steps)
    if outcome == "STEP_LIMIT_REACHED":
        assert batch.total_steps == 8 and all(r.result.steps == 4 for r in batch.episode_results)


def test_technical_error_stops_preserves_partial_trace_and_excludes_partial_path():
    adapter = FakeAdapter(fail_on_step=5)
    batch = BatchRunner().run(adapter, random_factory, episodes=10, base_seed=42)
    assert batch.attempted_episodes == 2 and batch.completed_episodes == 1
    assert batch.total_steps == 4 and len(adapter.sent) == 5
    assert batch.technical_errors == 1 and batch.outcomes == {"GAME_TERMINAL": 1, "TECHNICAL_ERROR": 1}
    assert batch.stop_reason == "TECHNICAL_ERROR" and batch.stopped_episode == 2
    partial = batch.episode_results[1]
    assert partial.result.steps == len(partial.result.trace.steps) == 1
    assert partial.result.trace.steps[0].next_observation == {"counter": 1}
    assert "disconnected" in partial.result.issues[0]
    assert partial.trace_validated and partial.action_sequence_signature is None
    assert batch.unique_action_sequences == batch.diversity_episodes == batch.diversity_excluded_episodes == 1
    assert adapter.reset_seeds == [None, None] and adapter.close_count == 0


def test_policy_factory_failure_is_recorded_without_reusing_adapter():
    adapter = FakeAdapter()

    def factory(index, policy_seed):
        if index == 1:
            raise RuntimeError("factory failed")
        return random_factory(index, policy_seed)

    batch = BatchRunner().run(adapter, factory, episodes=10, base_seed=42)
    assert batch.attempted_episodes == 2 and batch.completed_episodes == 1
    assert batch.technical_errors == 1 and batch.stopped_episode == 2
    failed = batch.episode_results[-1]
    assert failed.result.trace is None and not failed.trace_validated
    assert "factory failed" in failed.result.issues[0]
    assert batch.total_steps is None and batch.unique_action_sequences == 1
    assert adapter.reset_seeds == [None] and adapter.close_count == 0


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_inconsistent_trace_is_not_counted_as_a_complete_path(monkeypatch, missing):
    trace = None if missing else EpisodeTrace(seed=None)
    monkeypatch.setattr(EpisodeRunner, "run", lambda *args, **kwargs: EpisodeResult(
        "GAME_TERMINAL", 1, 0.1, trace=trace,
    ))
    batch = BatchRunner().run(FakeAdapter(), random_factory, episodes=5, base_seed=42)
    assert batch.attempted_episodes == 1 and batch.completed_episodes == 0
    assert batch.technical_errors == 1 and batch.total_steps is None
    assert batch.unique_action_sequences == 0 and batch.diversity_episodes == 0
    assert batch.diversity_excluded_episodes == 1
    assert batch.episode_results[0].result.trace is trace


def test_total_step_budget_caps_last_episode_without_exceeding_limit():
    adapter = FakeAdapter(after=None)
    batch = BatchRunner().run(adapter, random_factory, episodes=5, max_steps=4, max_total_steps=6, base_seed=42)
    assert batch.total_steps == len(adapter.sent) == 6
    assert [r.step_budget for r in batch.episode_results] == [4, 2]
    assert [r.result.steps for r in batch.episode_results] == [4, 2]
    assert batch.attempted_episodes == batch.completed_episodes == 2
    assert batch.stop_reason == "TOTAL_STEP_LIMIT_REACHED" and batch.stopped_episode == 2
    assert batch.technical_errors == 0 and adapter.reset_seeds == [None, None]


def test_total_budget_exhausted_exactly_on_requested_last_episode_is_completed():
    batch = BatchRunner().run(FakeAdapter(after=None), random_factory, episodes=2, max_steps=4, max_total_steps=6, base_seed=42)
    assert batch.total_steps == 6 and batch.completed_episodes == 2
    assert batch.stop_reason == "EPISODES_COMPLETED" and batch.stopped_episode is None


def test_total_budget_is_shared_across_short_terminal_episodes():
    adapter = FakeAdapter(after=1)
    batch = BatchRunner().run(adapter, random_factory, episodes=10, max_steps=50, max_total_steps=3, base_seed=42)
    assert batch.total_steps == batch.completed_episodes == len(adapter.sent) == 3
    assert batch.stop_reason == "TOTAL_STEP_LIMIT_REACHED" and batch.technical_errors == 0


def test_zero_total_budget_never_resets_or_creates_policy():
    adapter = FakeAdapter()

    def factory(*args):
        raise AssertionError("No policy may be created after budget exhaustion")

    batch = BatchRunner().run(adapter, factory, episodes=10, max_total_steps=0, base_seed=42)
    assert batch.attempted_episodes == batch.completed_episodes == batch.total_steps == 0
    assert batch.stop_reason == "TOTAL_STEP_LIMIT_REACHED" and batch.stopped_episode is None
    assert batch.average_episode_seconds is None and batch.unique_action_sequences == 0
    assert adapter.reset_seeds == [] and adapter.sent == [] and adapter.close_count == 0


def test_keyboard_interrupt_is_preserved_and_stops_batch(monkeypatch):
    adapter = FakeAdapter()

    def interrupt(seed=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(adapter, "reset", interrupt)
    batch = BatchRunner().run(adapter, random_factory, episodes=10, base_seed=42)
    assert batch.stop_reason == "INTERRUPTED" and batch.stopped_episode == 1
    assert batch.attempted_episodes == 1 and batch.completed_episodes == batch.technical_errors == 0
    assert batch.diversity_episodes == 0 and batch.episode_results[0].result.issues
    assert adapter.close_count == 0


def test_interrupt_during_diversity_preserves_finished_episode(monkeypatch):
    def interrupt(actions):
        raise KeyboardInterrupt

    monkeypatch.setattr("playtest.orchestration.batch.action_sequence_signature", interrupt)
    adapter = FakeAdapter()
    batch = BatchRunner().run(adapter, random_factory, episodes=10, base_seed=42)
    assert batch.stop_reason == "INTERRUPTED" and batch.stopped_episode == 1
    assert batch.attempted_episodes == batch.completed_episodes == 1
    assert batch.episode_results[0].result.outcome == "GAME_TERMINAL"
    assert len(batch.episode_results[0].result.trace.steps) == batch.total_steps == 3
    assert not batch.diversity_supported and batch.unique_action_sequences is None
    assert "KeyboardInterrupt" in batch.episode_results[0].diversity_error
    assert adapter.reset_seeds == [None] and adapter.close_count == 0


@pytest.mark.parametrize("kwargs", [
    {"episodes": 0}, {"episodes": -1}, {"episodes": True}, {"episodes": 1.5},
    {"max_steps": 0}, {"max_steps": False}, {"max_steps": 1.5},
    {"max_total_steps": -1}, {"max_total_steps": True}, {"max_total_steps": 1.5},
    {"base_seed": True}, {"base_seed": "42"}, {"base_seed": 1.5},
])
def test_invalid_configuration_fails_before_adapter_use(kwargs):
    adapter = FakeAdapter()
    with pytest.raises(ValueError):
        BatchRunner().run(adapter, random_factory, **kwargs)
    assert adapter.reset_seeds == [] and adapter.sent == []


def test_invalid_factory_is_rejected_before_adapter_use():
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="callable"):
        BatchRunner().run(adapter, None)
    assert adapter.reset_seeds == []


def test_invalid_policy_action_is_excluded_and_stops_batch():
    class BadPolicy:
        def select_action(self, observation, valid_actions):
            return Action("invalid")

    adapter = FakeAdapter()
    batch = BatchRunner().run(adapter, lambda index, seed: BadPolicy(), episodes=3, base_seed=42)
    assert batch.stop_reason == "INVALID_ACTION" and batch.attempted_episodes == 1
    assert batch.completed_episodes == batch.technical_errors == batch.unique_action_sequences == 0
    assert batch.diversity_excluded_episodes == 1 and not adapter.sent


def test_batch_import_does_not_load_a_unity_backend():
    code = (
        "import sys; sys.modules['mlagents_envs'] = None; "
        "from playtest.orchestration.batch import BatchRunner; "
        "assert 'playtest.adapters.unity_mlagents' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=20)
