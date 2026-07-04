# MP-VRDU Representation & Deployment Study

This repository implements the staged empirical pipeline described in
`docs/implementation_plan.md` for the MP-VRDU representation and deployment
study.

Stages 0-3 now provide the project skeleton, Kaya orchestration, complete
prestage/setup inventory, the concrete MMLongBench data layer, and the frozen
pipeline skeleton: all ABCs, the backend-agnostic `ModelInput`, and a caching
orchestrator that runs end to end on stubs. Real tools, models, metrics, and
experiment runners are filled in behind these frozen interfaces by the later
stages after the human checkpoints.

Run the stub pipeline over a tiny sample:

```
envs/mpvrdu/bin/python -m cli.run_experiment --sample 4
```

## Layout

- `docs/PROJECT_SPEC.md` defines what the experiments measure.
- `docs/implementation_plan.md` defines the staged build order.
- `docs/DECISIONS.md` records fixed decisions and stage findings.
- `kaya/` contains the Kaya config, generic Python sync/run/submit runner,
  task scripts, and Kaya guides.
- `kaya/KAYA_USER_GUIDE.md` is the human-facing Kaya quick guide.
- `kaya/KAYA_AGENT_GUIDE.md` is the agent-facing definitive Kaya operations guide.
- `docs/DATA.md` documents the Stage 2 normalized question schema and render cache.
- `docs/ARCHITECTURE.md` maps the tree to the paper and lists the Stage 3 frozen interfaces.

All machine-specific artifacts stay under the repository root and are ignored:
`.cache/`, `.data/`, `envs/`, `results/`, and `logs/`. The `data/` directory
is importable pipeline code, not a dataset store.
