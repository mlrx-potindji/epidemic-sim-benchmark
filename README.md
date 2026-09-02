# Epidemic Simulation Benchmark

Research code for comparing an Esper Entity Component System epidemic model
with a Mesa agent-based implementation.

## Layout

- `src/epidemic_sim/ecs`: implemented Esper model, components, and systems
- `src/epidemic_sim/mesa`: Mesa implementation area
- `src/epidemic_sim/networks`: shared network utilities
- `src/epidemic_sim/data`: data-related package code
- `src/epidemic_sim/analysis`: analysis and visualization package code
- `scripts`: runnable benchmark and data-generation scripts
- `experiments`: exploratory modeling workflows
- `data`: raw, processed, and generated datasets
- `results` and `project_outputs`: generated artifacts

## Running

Install the project with uv, then run the ECS benchmark with:

```bash
uv run python scripts/run_ecs.py
```

The synthetic-data script expects its source CSV files under `data/raw`:

```bash
uv run python scripts/generate_data.py
```

The synthetic-data workflow additionally requires SDV, which is kept outside
the core dependencies because its current releases do not resolve for Python
3.12.

The Mesa implementation and automated tests are still under development.