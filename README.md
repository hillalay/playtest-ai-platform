# Playtest AI

Playtest AI is a game-agnostic automated playtesting platform. This repository is being organized around the contracts and milestones in [the technical specification](dosyalar/Playtest_AI_Document.pdf).

## Repository layout

- `src/playtest/`: platform packages. These are package boundaries; implementation will be added as contracts are defined.
- `schemas/`: versioned JSON Schema files. The current placeholders reject all input until their contracts are implemented.
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
