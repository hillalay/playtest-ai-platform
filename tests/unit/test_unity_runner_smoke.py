import json
import subprocess
import sys

import numpy as np
import pytest

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.core.actions import Action
from playtest.core.results import EpisodeResult
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.random import RandomPolicy
from playtest.replay.trace import EpisodeTrace, TraceStep
from scripts import unity_runner_smoke as smoke
from test_unity_adapter_stability import scene
from test_unity_mlagents_adapter import Steps, empty, unity


def messages(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_batch_uses_runner_single_connection_and_separate_policy_seed(scene, monkeypatch, capsys):
    module = sys.modules["mlagents_envs.environment"]
    connect = module.UnityEnvironment
    connections = []
    runs = []
    run = EpisodeRunner.run

    def tracked_connect(**kwargs):
        connections.append(kwargs)
        return connect(**kwargs)

    def tracked_run(self, environment, policy, max_steps=100, seed=None):
        runs.append((environment, policy, seed))
        return run(self, environment, policy, max_steps=max_steps, seed=seed)

    monkeypatch.setattr(module, "UnityEnvironment", tracked_connect)
    monkeypatch.setattr(EpisodeRunner, "run", tracked_run)
    for _ in range(3):
        scene.frames.extend([
            (Steps(mask=[np.array([[False, True, True, True, True]])], values=(7, 0)), empty()),
            (Steps(values=(99, 99)), Steps(values=(0, 0))),
        ])
    assert smoke.main(["--episodes", "3", "--max-steps", "50", "--seed", "42", "--include-traces"]) == 0
    output = messages(capsys)
    records = [m for m in output if m["kind"] == "episode"]
    assert len(runs) == len(records) == 3 and len(connections) == 1
    assert len({id(env) for env, _, _ in runs}) == 1
    assert len({id(policy) for _, policy, _ in runs}) == 3
    assert all(seed is None for _, _, seed in runs)
    assert scene.reset_count == 3 and scene.close_count == 1
    assert scene.step_count == len(scene.sent) == 6
    assert [sent[0][0] for sent in scene.sent] == [2, 0, 2, 0, 3, 0]
    assert all(r["outcome"] == "GAME_TERMINAL" and r["steps"] == 2 for r in records)
    assert all(r["trace_validated"] for r in records)
    assert [r["policy_seed"] for r in records] == [42, 43, 44]
    for record in records:
        trace = record["trace"]
        assert trace["seed"] is None
        assert len(trace["steps"]) == record["steps"]
        assert trace["steps"][0]["observation"] == {"observations": [[1.25, 2.5]]}
        last = trace["steps"][-1]
        assert last["next_observation"] == {"observations": [[0.0, 0.0]]}
        assert last["reward_signals"] == {"unity": 0.5}
        assert last["game_terminal"] and not last["test_boundary_reached"]
        assert last["game_outcome"] is None and last["state_signature"] is None
        assert last["invalid_reason"] is None
    summary = output[-1]
    assert summary["requested_episodes"] == summary["completed_episodes"] == 3
    assert summary["outcomes"] == {"GAME_TERMINAL": 3}
    assert summary["total_steps"] == 6 and summary["technical_errors"] == 0
    assert summary["average_episode_seconds"] >= 0
    assert summary["base_seed"] == 42 and summary["unique_action_sequences"] == 2


@pytest.mark.parametrize("ending,outcome", [
    ("terminal", "GAME_TERMINAL"), ("interrupted", "TEST_BOUNDARY_REACHED"),
    ("no_actions", "NO_VALID_ACTIONS"), ("budget", "STEP_LIMIT_REACHED"),
])
def test_distinct_episode_stop_reasons(scene, capsys, ending, outcome):
    if ending in ("terminal", "interrupted"):
        scene.frames.append((empty(), Steps(values=(0, 0), interrupted=ending == "interrupted")))
    elif ending == "no_actions":
        scene.frames.append((Steps(mask=[np.ones((1, 5), dtype=bool)]), empty()))
    else:
        scene.frames.append((Steps(), empty()))
    assert smoke.main(["--episodes", "1", "--max-steps", "2", "--seed", "42", "--include-traces"]) == 0
    output = messages(capsys)
    record = output[-2]
    assert record["outcome"] == outcome and record["trace_validated"]
    assert record["steps"] == (2 if ending == "budget" else 1)
    assert len(record["trace"]["steps"]) == len(scene.sent)
    assert record["trace"]["steps"][-1]["game_outcome"] is None
    assert output[-1]["technical_errors"] == 0
    assert scene.close_count == 1


def test_no_actions_on_reset_does_not_call_policy_or_step(unity, monkeypatch, capsys):
    unity.decision.action_mask = [np.ones((1, 5), dtype=bool)]

    def unexpected_choice(*args, **kwargs):
        raise AssertionError("Policy must not be called on an empty action list")

    monkeypatch.setattr(RandomPolicy, "select_action", unexpected_choice)
    assert smoke.main(["--episodes", "3", "--seed", "42", "--include-traces"]) == 0
    output = messages(capsys)
    assert output[-1]["outcomes"] == {"NO_VALID_ACTIONS": 3}
    assert output[-1]["total_steps"] == 0
    assert all(r["trace"]["steps"] == [] for r in output if r["kind"] == "episode")
    assert not unity.sent and unity.reset_count == 3 and unity.close_count == 1


@pytest.mark.parametrize("operation", ["reset", "get_steps", "set_actions", "step"])
@pytest.mark.parametrize("error", [TimeoutError("timeout"), OSError("disconnect")])
def test_technical_error_stops_batch_and_closes(scene, monkeypatch, capsys, operation, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(scene, operation, fail)
    assert smoke.main(["--episodes", "3", "--seed", "42", "--include-traces"]) == 1
    output = messages(capsys)
    record = output[-2]
    assert record["outcome"] == "TECHNICAL_ERROR" and record["trace_validated"]
    assert record["steps"] == len(record["trace"]["steps"]) == 0
    assert str(error) in record["issues"][0]
    assert output[-1]["completed_episodes"] == 0 and output[-1]["technical_errors"] == 1
    assert output[-1]["unattempted_episodes"] == 2
    assert scene.reset_count <= 1 and scene.close_count == 1


def test_disconnect_preserves_partial_trace_and_total_returned_steps(scene, monkeypatch, capsys):
    step = scene.step
    scene.frames.append((Steps(values=(7, 0)), empty()))

    def fail_second():
        if scene.step_count == 1:
            raise OSError("disconnect after first action")
        step()

    monkeypatch.setattr(scene, "step", fail_second)
    assert smoke.main(["--episodes", "3", "--seed", "42", "--include-traces"]) == 1
    output = messages(capsys)
    record = output[-2]
    assert record["outcome"] == "TECHNICAL_ERROR" and record["steps"] == 1
    assert record["trace"]["steps"][0]["next_observation"] == {"observations": [[7.0, 0.0]]}
    assert output[-1]["total_steps"] == 1 and scene.reset_count == 1
    # The failed transition has no returned observation; it is not invented in the trace.
    assert len(scene.sent) == 2 and scene.close_count == 1


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_keyboard_interrupt_is_nonzero_and_closes(scene, monkeypatch, capsys, operation):
    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(scene, operation, interrupt)
    assert smoke.main(["--episodes", "3"]) == 130
    output = messages(capsys)
    assert output[-2]["outcome"] == "INTERRUPTED"
    assert output[-1]["interrupted"] and output[-1]["completed_episodes"] == 0
    assert scene.close_count == 1


def test_close_error_is_reported_and_fails(scene, monkeypatch, capsys):
    def fail_close():
        scene.close_count += 1
        raise OSError("close failed")

    monkeypatch.setattr(scene, "close", fail_close)
    assert smoke.main(["--episodes", "1", "--max-steps", "1"]) == 1
    summary = messages(capsys)[-1]
    assert summary["completed_episodes"] == 1 and summary["technical_errors"] == 1
    assert "close failed" in summary["close_error"] and scene.close_count == 1


def test_static_level_selected_once_without_runtime_seed(scene, monkeypatch):
    load = UnityMLAgentsAdapter.load_level
    calls = []

    def load_static(self, level_ref, seed=None):
        calls.append((level_ref, seed))
        return load(self, level_ref, seed=seed)

    monkeypatch.setattr(UnityMLAgentsAdapter, "load_level", load_static)
    assert smoke.main(["--episodes", "3", "--max-steps", "1", "--seed", "42"]) == 0
    assert calls == [("static", None)] and scene.reset_count == 3


def test_load_failure_closes_and_reports_unknown_average(unity, monkeypatch, capsys):
    closes = []
    close = UnityMLAgentsAdapter.close

    def fail_load(*args):
        raise RuntimeError("load failed")

    def tracked_close(self):
        closes.append(True)
        return close(self)

    monkeypatch.setattr(UnityMLAgentsAdapter, "load_level", fail_load)
    monkeypatch.setattr(UnityMLAgentsAdapter, "close", tracked_close)
    assert smoke.main(["--episodes", "3"]) == 1
    summary = messages(capsys)[-1]
    assert summary["completed_episodes"] == 0 and summary["technical_errors"] == 1
    assert summary["average_episode_seconds"] is None
    assert "load failed" in summary["batch_error"]
    assert len(closes) == 1 and unity.kwargs is None


@pytest.mark.parametrize("argv", [
    ["--episodes", "0"], ["--episodes", "-1"], ["--episodes", "1.5"],
    ["--max-steps", "0"], ["--max-steps", "-1"],
    ["--timeout-wait", "0"], ["--timeout-wait", "1.5"],
    ["--max-wait-steps", "-1"], ["--max-wait-steps", "1.5"], ["--seed", "oops"],
])
def test_cli_validation_happens_before_connecting(unity, argv):
    with pytest.raises(SystemExit) as error:
        smoke.main(argv)
    assert error.value.code == 2
    assert unity.kwargs is None and unity.close_count == 0


def test_cli_connection_options_are_forwarded(scene):
    assert smoke.main([
        "--episodes", "1", "--max-steps", "1", "--file-name", "Game.exe",
        "--behavior", "Game", "--timeout-wait", "42", "--max-wait-steps", "0", "--seed", "-3",
    ]) == 0
    assert scene.kwargs == {"file_name": "Game.exe", "timeout_wait": 42}


def test_invalid_policy_action_stops_without_sending(scene, monkeypatch, capsys):
    monkeypatch.setattr(RandomPolicy, "select_action", lambda *args, **kwargs: Action("invalid"))
    assert smoke.main(["--episodes", "3"]) == 1
    output = messages(capsys)
    assert output[-2]["outcome"] == "INVALID_ACTION" and output[-2]["steps"] == 0
    assert output[-1]["completed_episodes"] == 0 and output[-1]["technical_errors"] == 0
    assert scene.sent == [] and scene.close_count == 1


def test_trace_validation_failure_stops_and_returns_nonzero(scene, monkeypatch, capsys):
    monkeypatch.setattr(EpisodeRunner, "run", lambda *args, **kwargs: EpisodeResult(
        outcome="GAME_TERMINAL", steps=1, duration_seconds=0.5,
        trace=EpisodeTrace(seed=None),
    ))
    assert smoke.main(["--episodes", "3"]) == 1
    output = messages(capsys)
    assert not output[-2]["trace_validated"]
    assert output[-2]["outcome"] == "TECHNICAL_ERROR"
    assert "Trace step count" in output[-2]["issues"][0]
    assert output[-1]["technical_errors"] == 1 and scene.close_count == 0
    assert output[-1]["total_steps"] is None


@pytest.mark.parametrize("fault", ["index", "stop", "outcome", "terminal", "budget", "flags"])
def test_trace_validator_detects_inconsistent_transitions(fault):
    trace = EpisodeTrace(seed=None, steps=[TraceStep(
        step_index=1, observation={}, action=Action("step"), next_observation={},
        game_terminal=False, test_boundary_reached=False,
    )])
    result = EpisodeResult("STEP_LIMIT_REACHED", 1, 0.5, trace=trace)
    max_steps = 1
    if fault == "index":
        trace.steps[0].step_index = 2
    elif fault == "stop":
        trace.steps[0].game_terminal = True
        trace.steps.append(TraceStep(2, {}, Action("step"), {}, game_terminal=False, test_boundary_reached=False))
        result.steps = max_steps = 2
    elif fault == "outcome":
        trace.steps[0].game_terminal = True
        result.outcome = "NO_VALID_ACTIONS"
    elif fault == "terminal":
        result.outcome = "GAME_TERMINAL"
    elif fault == "budget":
        max_steps = 2
    else:
        trace.steps[0].game_terminal = None
    with pytest.raises(ValueError):
        smoke.validate_trace(result, max_steps)


def test_cli_process_exits_nonzero_without_a_real_unity_connection():
    code = (
        "import runpy, sys; sys.modules['mlagents_envs.environment'] = None; "
        "sys.argv = ['scripts/unity_runner_smoke.py', '--episodes', '3']; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert output[-1]["technical_errors"] == 1 and output[-1]["completed_episodes"] == 0


def test_100_episode_batch_reports_diversity_with_compact_default_output(scene, capsys):
    for _ in range(100):
        scene.frames.extend([
            (Steps(), empty()),
            (Steps(), empty()),
            (empty(), Steps(values=(0, 0))),
        ])
    assert smoke.main(["--episodes", "100", "--max-steps", "50", "--seed", "42"]) == 0
    output = messages(capsys)
    records = [r for r in output if r["kind"] == "episode"]
    assert len(records) == 100 and all("trace" not in r for r in records)
    assert [r["policy_seed"] for r in records] == list(range(42, 142))
    assert scene.reset_count == 100 and scene.step_count == len(scene.sent) == 300
    assert scene.close_count == 1
    summary = output[-1]
    assert summary["completed_episodes"] == summary["diversity_episodes"] == 100
    assert summary["total_steps"] == 300 and summary["unique_action_sequences"] > 1
    assert summary["technical_errors"] == 0 and summary["diversity_excluded_episodes"] == 0
    assert summary["diversity_supported"] and "engine-index" in summary["action_sequence_basis"]
    assert all(len(json.dumps(record)) < 1500 for record in records)


def test_optional_total_budget_stops_without_failure_or_extra_episode(scene, capsys):
    assert smoke.main([
        "--episodes", "10", "--max-steps", "3", "--max-total-steps", "5", "--seed", "42",
    ]) == 0
    output = messages(capsys)
    records = [r for r in output if r["kind"] == "episode"]
    assert [r["step_budget"] for r in records] == [3, 2]
    assert scene.step_count == len(scene.sent) == output[-1]["total_steps"] == 5
    assert scene.reset_count == 2 and scene.close_count == 1
    assert output[-1]["stop_reason"] == "TOTAL_STEP_LIMIT_REACHED"
    assert output[-1]["completed_episodes"] == 2 and output[-1]["unattempted_episodes"] == 8
    assert output[-1]["technical_errors"] == 0


def test_zero_total_budget_opens_no_connection(unity, capsys):
    assert smoke.main(["--episodes", "10", "--max-total-steps", "0", "--seed", "42"]) == 0
    output = messages(capsys)
    assert output[-1]["stop_reason"] == "TOTAL_STEP_LIMIT_REACHED"
    assert output[-1]["attempted_episodes"] == output[-1]["total_steps"] == 0
    assert unity.kwargs is None and unity.reset_count == unity.step_count == unity.close_count == 0


def test_omitted_seed_is_generated_recorded_and_used(scene, monkeypatch, capsys):
    monkeypatch.setattr("playtest.orchestration.batch.secrets.randbits", lambda bits: 54321)
    assert smoke.main(["--episodes", "2", "--max-steps", "1"]) == 0
    output = messages(capsys)
    assert output[-1]["base_seed"] == 54321
    assert [r["policy_seed"] for r in output if r["kind"] == "episode"] == [54321, 54322]


@pytest.mark.parametrize("value", ["-1", "1.5", "oops"])
def test_invalid_total_budget_is_rejected_before_connection(unity, value):
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--max-total-steps", value])
    assert exc.value.code == 2 and unity.kwargs is None


def test_unsupported_diversity_fails_smoke_without_inventing_a_count(unity, monkeypatch, capsys):
    from playtest.orchestration.batch import BatchRunner
    from test_batch import FakeAdapter, random_factory

    batch = BatchRunner().run(
        FakeAdapter(actions=[Action("pick", {"coords": (1, 2)})]), random_factory,
        episodes=1, base_seed=42,
    )
    monkeypatch.setattr(BatchRunner, "run", lambda *args, **kwargs: batch)
    assert smoke.main(["--episodes", "1", "--seed", "42"]) == 1
    summary = messages(capsys)[-1]
    assert summary["completed_episodes"] == 1 and summary["technical_errors"] == 0
    assert summary["unique_action_sequences"] is None and not summary["diversity_supported"]


def test_interrupt_during_diversity_keeps_episode_report_and_closes(scene, monkeypatch, capsys):
    def interrupt(actions):
        raise KeyboardInterrupt

    monkeypatch.setattr("playtest.orchestration.batch.action_sequence_signature", interrupt)
    assert smoke.main(["--episodes", "3", "--max-steps", "1", "--seed", "42"]) == 130
    output = messages(capsys)
    assert output[-2]["outcome"] == "STEP_LIMIT_REACHED" and output[-2]["steps"] == 1
    assert output[-1]["completed_episodes"] == 1 and output[-1]["stop_reason"] == "INTERRUPTED"
    assert output[-1]["unique_action_sequences"] is None
    assert scene.reset_count == scene.close_count == 1
