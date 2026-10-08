import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.core.results import StepResult
from scripts import unity_adapter_stability as stability
from test_unity_mlagents_adapter import Steps, empty, unity


@pytest.fixture
def scene(unity, monkeypatch):
    initial = deepcopy(unity.decision)

    def reset():
        unity.reset_count += 1
        unity.decision = deepcopy(initial)
        unity.terminal = empty()

    monkeypatch.setattr(unity, "reset", reset)
    return unity


def output(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_100_episodes_use_one_connection_explicit_resets_and_legal_actions(scene, monkeypatch, capsys):
    module = sys.modules["mlagents_envs.environment"]
    connect = module.UnityEnvironment
    connections = []

    def tracked_connect(**kwargs):
        connections.append(kwargs)
        return connect(**kwargs)

    monkeypatch.setattr(module, "UnityEnvironment", tracked_connect)
    scene.frames.extend((Steps(values=(99, 99)), Steps(values=(0, 0))) for _ in range(100))
    assert stability.main([]) == 0
    messages = output(capsys)
    episodes = [m for m in messages if m["kind"] == "episode"]
    resets = [m for m in messages if m["kind"] == "reset"]
    steps = [m for m in messages if m["kind"] == "step"]
    assert len(connections) == 1
    assert scene.reset_count == scene.step_count == len(scene.sent) == 100
    assert scene.close_count == 1
    assert all(np.array_equal(sent, [[2]]) for sent in scene.sent)
    assert len(episodes) == len(resets) == len(steps) == 100
    assert [m["episode"] for m in episodes] == list(range(1, 101))
    assert all(m["outcome"] == "GAME_TERMINAL" and m["steps"] == 1 for m in episodes)
    assert all(m["last_observation"] == {"observations": [[0.0, 0.0]]} for m in episodes)
    assert all(m["game_outcome"] is None for m in episodes)
    assert all(m["baseline_match"] for m in resets)
    summary = messages[-1]
    assert summary["requested_episodes"] == summary["completed_episodes"] == 100
    assert summary["game_terminal"] == 100 and summary["technical_error"] == 0
    assert summary["average_reset_seconds"] >= 0
    assert summary["average_episode_seconds"] >= summary["average_reset_seconds"]


def test_summary_counts_normal_boundary_no_actions_and_budget_separately(scene, capsys):
    scene.frames.extend([
        (empty(), Steps(values=(0, 0))),
        (empty(), Steps(values=(7, 0), interrupted=True)),
        (Steps(mask=[np.ones((1, 5), dtype=bool)]), empty()),
        (Steps(), empty()),
        (Steps(), empty()),
    ])
    assert stability.main(["--episodes", "4", "--max-steps", "2"]) == 0
    messages = output(capsys)
    records = [m for m in messages if m["kind"] == "episode"]
    assert [r["outcome"] for r in records] == [
        "GAME_TERMINAL", "TEST_BOUNDARY_REACHED", "NO_VALID_ACTIONS", "STEP_LIMIT_REACHED"
    ]
    assert [r["steps"] for r in records] == [1, 1, 1, 2]
    assert all(r["game_outcome"] is None for r in records)
    summary = messages[-1]
    assert summary["completed_episodes"] == 4
    assert all(summary[key] == 1 for key in (
        "game_terminal", "test_boundary_reached", "no_valid_actions", "max_steps_reached"
    ))
    assert summary["technical_error"] == summary["reset_mismatch"] == 0
    assert scene.reset_count == 4 and scene.step_count == 5 and scene.close_count == 1


def test_no_actions_at_reset_never_sends_an_action(unity, capsys):
    unity.decision.action_mask = [np.ones((1, 5), dtype=bool)]
    assert stability.main(["--episodes", "3"]) == 0
    summary = output(capsys)[-1]
    assert summary["completed_episodes"] == summary["no_valid_actions"] == 3
    assert summary["game_terminal"] == 0
    assert unity.reset_count == 3 and unity.close_count == 1
    assert unity.sent == [] and unity.step_count == 0


@pytest.mark.parametrize("interrupted", [False, True])
def test_terminal_on_reset_is_not_mislabeled_as_no_actions(unity, capsys, interrupted):
    unity.terminal = Steps(values=(0, 0), interrupted=interrupted)
    assert stability.main(["--episodes", "2"]) == 0
    messages = output(capsys)
    records = [m for m in messages if m["kind"] == "episode"]
    assert all(r["terminal_on_reset"] and r["steps"] == 0 for r in records)
    assert messages[-1]["test_boundary_reached"] == (2 if interrupted else 0)
    assert messages[-1]["game_terminal"] == (0 if interrupted else 2)
    assert messages[-1]["no_valid_actions"] == 0
    assert not unity.sent and unity.step_count == 0


@pytest.mark.parametrize("mismatch", ["observation", "actions"])
def test_reset_mismatch_stops_before_action_and_fails(scene, monkeypatch, capsys, mismatch):
    reset = scene.reset

    def changed_reset():
        reset()
        if scene.reset_count == 2:
            if mismatch == "observation":
                scene.decision.obs[0][0][0] = 999
            else:
                scene.decision.action_mask[0][0][2] = True

    monkeypatch.setattr(scene, "reset", changed_reset)
    assert stability.main(["--episodes", "100", "--max-steps", "1"]) == 1
    messages = output(capsys)
    assert messages[-2]["outcome"] == "RESET_MISMATCH"
    assert messages[-2]["steps"] == 0
    assert messages[-1]["reset_mismatch"] == 1
    assert messages[-1]["completed_episodes"] == 1
    assert messages[-1]["unattempted_episodes"] == 98
    assert scene.reset_count == 2 and scene.step_count == 1 and scene.close_count == 1


@pytest.mark.parametrize("operation", ["reset", "step", "get_steps", "set_actions"])
@pytest.mark.parametrize("error", [TimeoutError("timeout"), OSError("disconnected")])
def test_transport_error_is_nonzero_reports_context_and_never_reuses_connection(
    scene, monkeypatch, capsys, operation, error
):
    def fail(*args):
        raise error

    monkeypatch.setattr(scene, operation, fail)
    assert stability.main(["--episodes", "100"]) == 1
    messages = output(capsys)
    record = messages[-2]
    assert record["episode"] == 1 and record["steps"] == 0 and record["technical_error"]
    assert str(error) in record["error"]
    assert record["attempted_step"] == (None if operation in ("reset", "get_steps") else 1)
    assert messages[-1]["technical_error"] == 1 and messages[-1]["completed_episodes"] == 0
    assert messages[-1]["unattempted_episodes"] == 99
    assert scene.close_count == 1
    assert scene.reset_count <= 1


def test_error_after_completed_step_keeps_last_known_observation(scene, monkeypatch, capsys):
    original_step = scene.step
    scene.frames.append((Steps(values=(7, 0)), empty()))

    def fail_second_step():
        if scene.step_count == 1:
            raise OSError("disconnected")
        original_step()

    monkeypatch.setattr(scene, "step", fail_second_step)
    assert stability.main(["--episodes", "10"]) == 1
    record = output(capsys)[-2]
    assert record["steps"] == 1 and record["attempted_step"] == 2
    assert record["last_observation"] == {"observations": [[7.0, 0.0]]}
    assert scene.reset_count == 1 and scene.close_count == 1


@pytest.mark.parametrize("during_reset", [False, True])
def test_wait_budget_exhaustion_is_technical_failure(scene, monkeypatch, capsys, during_reset):
    if during_reset:
        def reset():
            scene.reset_count += 1
            scene.decision, scene.terminal = empty(), empty()

        monkeypatch.setattr(scene, "reset", reset)
    else:
        scene.frames.append((empty(), empty()))
    assert stability.main(["--episodes", "10", "--max-wait-steps", "2"]) == 1
    messages = output(capsys)
    assert "after 2 waiting steps" in messages[-2]["error"]
    assert messages[-1]["technical_error"] == 1
    assert scene.step_count == (2 if during_reset else 3)
    assert scene.close_count == 1


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_keyboard_interrupt_closes_and_returns_130(scene, monkeypatch, capsys, operation):
    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(scene, operation, interrupt)
    assert stability.main(["--episodes", "100"]) == 130
    messages = output(capsys)
    assert messages[-2]["outcome"] == "INTERRUPTED"
    assert messages[-1]["interrupted"] and messages[-1]["completed_episodes"] == 0
    assert scene.close_count == 1


def test_close_error_fails_even_when_episodes_completed(scene, monkeypatch, capsys):
    def fail_close():
        scene.close_count += 1
        raise OSError("close failed")

    monkeypatch.setattr(scene, "close", fail_close)
    assert stability.main(["--episodes", "1", "--max-steps", "1"]) == 1
    summary = output(capsys)[-1]
    assert summary["completed_episodes"] == 1 and summary["technical_error"] == 1
    assert "close failed" in summary["close_error"] and scene.close_count == 1


def test_rejected_enumerated_action_is_technical_failure(scene, monkeypatch, capsys):
    monkeypatch.setattr(UnityMLAgentsAdapter, "step", lambda *args: StepResult(
        next_observation={"observations": [[1, 2]]}, invalid_reason="mask changed"
    ))
    assert stability.main(["--episodes", "2"]) == 1
    messages = output(capsys)
    assert "Enumerated action was rejected" in messages[-2]["error"]
    assert messages[-1]["technical_error"] == 1 and not scene.sent
    assert scene.close_count == 1


@pytest.mark.parametrize("argv", [
    ["--episodes", "0"], ["--episodes", "-1"], ["--episodes", "1.5"],
    ["--max-steps", "0"], ["--max-steps", "-1"], ["--max-steps", "oops"],
    ["--timeout-wait", "0"], ["--timeout-wait", "-5"], ["--timeout-wait", "1.5"],
    ["--max-wait-steps", "-1"], ["--max-wait-steps", "1.5"],
])
def test_invalid_cli_fails_before_connection(unity, argv):
    with pytest.raises(SystemExit) as exc:
        stability.main(argv)
    assert exc.value.code == 2
    assert unity.kwargs is None and unity.reset_count == unity.close_count == 0


def test_cli_forwards_connection_options_and_accepts_zero_wait(scene, capsys):
    assert stability.main([
        "--episodes", "1", "--max-steps", "1", "--behavior", "Game",
        "--file-name", "Example.exe", "--timeout-wait", "42", "--max-wait-steps", "0",
    ]) == 0
    assert scene.kwargs == {"file_name": "Example.exe", "timeout_wait": 42}
    assert scene.close_count == 1


def test_connection_failure_is_reported_and_closed_safely(unity, monkeypatch, capsys):
    def fail(**kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(sys.modules["mlagents_envs.environment"], "UnityEnvironment", fail)
    assert stability.main(["--episodes", "100"]) == 1
    messages = output(capsys)
    assert "connection refused" in messages[-2]["error"]
    assert messages[-1]["technical_error"] == 1 and messages[-1]["completed_episodes"] == 0
    assert unity.close_count == 0


def test_script_process_exits_nonzero_without_connecting_to_real_unity():
    code = (
        "import runpy, sys; "
        "sys.modules['mlagents_envs.environment'] = None; "
        "sys.argv = ['scripts/unity_adapter_stability.py', '--episodes', '1']; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[-1]["technical_error"] == 1
    assert messages[-1]["completed_episodes"] == 0
    assert "Install mlagents-envs==1.1.0" in messages[-2]["error"]


def test_unsupported_action_enumeration_is_technical_failure(scene, monkeypatch, capsys):
    monkeypatch.setattr(UnityMLAgentsAdapter, "valid_actions", lambda *args: None)
    assert stability.main(["--episodes", "3"]) == 1
    messages = output(capsys)
    assert "enumerated valid actions" in messages[-2]["error"]
    assert messages[-1]["technical_error"] == 1
    assert not scene.sent and scene.close_count == 1
