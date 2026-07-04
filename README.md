# MP-VRDU Representation & Deployment Study

This repository implements the staged empirical pipeline described in
`docs/implementation_plan.md` for the MP-VRDU representation and deployment
study.

Stages M1-M3 now provide the smoke corpus, Kaya orchestration, complete
prestage/setup inventory, the concrete MMLongBench data layer, Marker/OCR/visual
tools, and the first real Qwen3-VL smoke reasoner behind the frozen
`Reasoner`/`ModelInput` boundary. Remaining model sizes, metrics, and experiment
runners are filled in behind these interfaces by later stages.

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
- `docs/implementation_plan.md` is the build plan; `docs/DECISIONS.md` records the
  tree-to-paper mapping, frozen interfaces, and every fixed decision.
- `docs/MODELS.md` records the M3 Qwen3-VL load path and frozen prompt template.

All machine-specific artifacts stay under the repository root and are ignored:
`.cache/`, `.data/`, `envs/`, `results/`, and `logs/`. The `data/` directory
is importable pipeline code, not a dataset store.
