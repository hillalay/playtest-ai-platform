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
    assert smoke.main(["--episodes", "3", "--max-steps", "50", "--seed", "42"]) == 0
    output = messages(capsys)
    records = [m for m in output if m["kind"] == "episode"]
    assert len(runs) == len(records) == 3 and len(connections) == 1
    assert len({id(env) for env, _, _ in runs}) == 1
    assert len({id(policy) for _, policy, _ in runs}) == 3
    assert all(seed is None for _, _, seed in runs)
    assert scene.reset_count == 3 and scene.close_count == 1
    assert scene.step_count == len(scene.sent) == 6
    assert [sent[0][0] for sent in scene.sent] == [2, 0, 2, 0, 2, 0]
    assert all(r["outcome"] == "GAME_TERMINAL" and r["steps"] == 2 for r in records)
    assert all(r["trace_validated"] and r["policy_seed"] == 42 for r in records)
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
    assert smoke.main(["--episodes", "1", "--max-steps", "2", "--seed", "42"]) == 0
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
    assert smoke.main(["--episodes", "3", "--seed", "42"]) == 0
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
    assert smoke.main(["--episodes", "3", "--seed", "42"]) == 1
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
    assert smoke.main(["--episodes", "3", "--seed", "42"]) == 1
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
