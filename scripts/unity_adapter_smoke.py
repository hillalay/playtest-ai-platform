"""Manual Unity integration check; never collected by the offline test suite."""

import argparse

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-name", default=None, help="Unity executable; omit for Unity Editor")
    parser.add_argument("--behavior", default=None, help="Exact ML-Agents behavior name")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout-wait", type=int, default=120)
    parser.add_argument("--max-wait-steps", type=int, default=100)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")

    adapter = UnityMLAgentsAdapter(
        file_name=args.file_name,
        timeout_wait=args.timeout_wait,
        behavior_name=args.behavior,
        max_wait_steps=args.max_wait_steps,
    )
    try:
        print("Waiting for Unity. For Editor mode, press Play when prompted.", flush=True)
        adapter.load_level("static")
        print("Initial observation:", adapter.reset())
        print("Capabilities:", adapter.capabilities())
        for _ in range(args.steps):
            actions = adapter.valid_actions()
            print("Valid actions:", actions)
            if not actions:
                print("No legal actions; no command sent.")
                break
            result = adapter.step(actions[0])
            if result.invalid_reason:
                raise RuntimeError(f"Enumerated action was rejected: {result.invalid_reason}")
            print("Step result:", result)
            if result.game_terminal or result.test_boundary_reached:
                break
        print("Explicit reset observation:", adapter.reset())
        print("Reset valid actions:", adapter.valid_actions())
        print("PASS: adapter connection/reset/action check completed; success semantics are codec-owned.")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
