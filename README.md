# MP-VRDU Representation & Deployment Study

This repository implements the staged empirical pipeline described in
`docs/implementation_plan.md` for the MP-VRDU representation and deployment
study.

Stage 0 creates the project skeleton only: importable modules, dependency
declarations, decision/runbook docs, and Kaya orchestration scripts. The actual
loader, schema, pipeline interfaces, tools, models, metrics, and experiment
runner are intentionally filled in by later stages after the human checkpoints.

## Layout

- `docs/PROJECT_SPEC.md` defines what the experiments measure.
- `docs/implementation_plan.md` defines the staged build order.
- `docs/DECISIONS.md` records fixed decisions and stage findings.
- `docs/KAYA.md` is the Kaya runbook for the pipeline mirror.
- `scripts/kaya/` contains the local-to-Kaya sync/setup/job wrappers.

All machine-specific artifacts stay under the repository root and are ignored:
`.cache/`, `.data/`, `envs/`, `results/`, and `logs/`. The `data/` directory
is importable pipeline code, not a dataset store.
