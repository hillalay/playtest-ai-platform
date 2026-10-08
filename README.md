# Playtest AI

Playtest AI is a game-agnostic automated playtesting platform. This repository is being organized around the contracts and milestones in [the technical specification](dosyalar/Playtest_AI_Document.pdf).

## Repository layout

- `src/playtest/`: platform packages. These are package boundaries; implementation will be added as contracts are defined.
- `src/playtest/definition/playtest_definition.schema.json`: packaged game definition schema, shared by the CLI and tests.
- `schemas/`: placeholders for the remaining contracts; these reject all input until implemented.
- `cli/`, `api/`, `sdk/unity/`: planned integration surfaces.
- `examples/arrow_grid/`: planned first game example.
- `tests/`: unit, contract, integration, property, and benchmark test locations.
- `core/`, `agents/`, `solvers/`, `games/`: existing Python prototype, retained in place for now. It has not yet been migrated to `src/playtest/`.

## Setup

```bash
python -m pip install -e .
python test_arrow_adapter.py
python test_arrow_runner.py
```

The prototype runner uses multiprocessing, so the runner example needs an environment that allows child processes.

## Next milestone

Define the Universal Test Model contracts, replace the schema placeholders, and create a fake environment that completes one run end to end before moving the prototype modules.

## Unity ML-Agents adapter

`playtest.adapters.unity_mlagents.UnityMLAgentsAdapter` implements the current
`src/playtest/adapters/base.py` GameAdapter protocol. Use the same Python 3.10
environment and compatible `mlagents-envs` installation as `scripts/unity_spike.py`.
ML-Agents is imported lazily when connecting; ordinary unit tests need no Unity
process or ML-Agents installation.

Manual integration check: open the Arrow Puzzle scene in Unity and start Play
when the connection waits. Run this separately from the unit suite:

```python
from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter

adapter = UnityMLAgentsAdapter(file_name=None, timeout_wait=120)
try:
    adapter.load_level("static")
    observation = adapter.reset()
    print(observation, adapter.canonical_state())
    actions = adapter.valid_actions()
    if actions:
        result = adapter.step(actions[0])
        print(result, adapter.valid_actions(), adapter.goal_test())
finally:
    adapter.close()
```

This first version requires exactly one behavior, one agent, one discrete action
branch, and one vector observation with five integer values per arrow:
`arrowId, row, column, direction, isActive`. Each reset/step must return a decision
or terminal for that agent; delayed decisions and multiple agents are unsupported.
Observations become plain dictionaries of sorted arrow records; canonical state
is deterministic UTF-8 JSON bytes. Actions are
`Action("select_arrow", {"action_id": N})`, where N is the Unity branch index,
not a separately inferred arrow ID. Missing masks enable the whole discrete branch.

`load_level("static")` explicitly leaves the current scene in place. Other level
references, runtime seeds, arbitrary-state action queries and state restoration
raise `EnvironmentError`. Cloning returns `None`; events return an empty list.
Reset connects lazily and resets the Unity episode; close is idempotent and final.
Terminal episodes have no legal actions. Non-interrupted termination sets
`game_terminal`; interruption sets `test_boundary_reached`. Neither implies
success: `goal_test()` and `game_outcome` remain `None` until Unity exposes a
verified completion/failure signal. No gameplay or runner integration is included.
