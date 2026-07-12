"""Pre-submission preflight: check on the login node that a generation spec will run.

Runs in the core env, offline, with no GPU. It parses the spec, imports the pipeline,
loads the dataset and resolves the corpus, and confirms every reasoner / retriever /
parser / classifier weight the run needs is already staged in the offline HF cache, so
a problem is caught before the job clears the SLURM queue rather than after. Exits
nonzero if any hard check fails, so `kaya submit` can gate on it.

    python -m ops.scripts.preflight --spec ops/specs/kaya_g2_full.yaml
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=true

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    """Collects check rows and decides the exit code (any FAIL -> nonzero)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.rows.append((level, name, detail))

    def ok(self, name: str, detail: str) -> None:
        self.add(PASS, name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add(WARN, name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(FAIL, name, detail)

    @property
    def failed(self) -> bool:
        return any(level == FAIL for level, _, _ in self.rows)

    def print(self) -> None:
        width = max((len(n) for _, n, _ in self.rows), default=10)
        for level, name, detail in self.rows:
            print(f"  [{level}] {name:<{width}}  {detail}")
        n_fail = sum(1 for level, _, _ in self.rows if level == FAIL)
        n_warn = sum(1 for level, _, _ in self.rows if level == WARN)
        verdict = "FAIL - do not submit" if self.failed else (
            "OK (with warnings)" if n_warn else "OK - clear to submit")
        print(f"\n[preflight] {verdict}  ({n_fail} fail, {n_warn} warn, {len(self.rows)} checks)")


# -- weight / env resolution -------------------------------------------------


def _is_staged(repo_id: str) -> bool:
    """True if the full snapshot for `repo_id` is in the offline HF cache."""

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def _reasoner_model_id(spec: str) -> str | None:
    """The HF repo id for a reasoner spec (quant suffix stripped), or None for stub."""

    from models import ModelSpec

    parsed = ModelSpec.parse(spec)
    if parsed.family == "stub":
        return None
    if parsed.family == "qwen3vl":
        from models.qwen3vl import model_id_for_spec
        return model_id_for_spec(spec)
    if parsed.family == "internvl3":
        from models.internvl import model_id_for_spec
        return model_id_for_spec(spec)
    raise ValueError(f"unknown reasoner family for spec {spec!r}")


def _retriever_model_id(name: str, kind: str) -> str | None:
    """The HF repo id for a retriever name, or None for bm25/none/unknown."""

    key = str(name).strip().lower()
    if key in ("", "none", "bm25"):
        return None
    if kind == "text":
        from retrievers.text import TEXT_RETRIEVERS
        cls = TEXT_RETRIEVERS.get(key)
    else:
        from retrievers.vision import VISION_RETRIEVERS
        cls = VISION_RETRIEVERS.get(key)
    if cls is None:
        return None
    return getattr(cls, "model_id", "") or None


def _parser_model_id(tool: str) -> str | None:
    """The parser's HF repo id from the Kaya config (or None if not listed)."""

    import json

    config_path = ROOT / "ops" / "kaya" / "config.json"
    parsers = json.loads(config_path.read_text()).get("parsers", {})
    return parsers.get(tool) if isinstance(parsers, dict) else None


# -- per-run checks ----------------------------------------------------------


def _check_weight(report: Report, label: str, repo_id: str | None, *, hard: bool, consequence: str = "") -> None:
    """Record a staged/missing verdict for one weight (hard = FAIL, else WARN)."""

    if not repo_id:
        return
    if _is_staged(repo_id):
        report.ok(label, f"{repo_id} staged")
    elif hard:
        report.fail(label, f"{repo_id} NOT staged (offline load will fail); run prestage")
    else:
        note = f" ({consequence})" if consequence else ""
        report.warn(label, f"{repo_id} not staged{note}")


def _check_run(report: Report, spec, gres_gpus: int | None, gres_type: str | None) -> None:
    from experiments.corpus.yaml_spec import config_from_spec, corpus_limit
    from experiments.registry import resolve

    tag = spec.run_tag or spec.task_name
    try:
        config = config_from_spec(spec)
    except Exception as exc:  # noqa: BLE001 - a bad config is a hard preflight failure
        report.fail(f"{tag}:config", f"config_from_spec raised {type(exc).__name__}: {exc}")
        return
    report.ok(f"{tag}:config", f"task={spec.task_name} dataset={config.dataset} parser={config.parser_tool}")

    # Dataset + corpus resolution (needs the staged data + a valid sampling scope).
    try:
        from ops.generate import load_corpus
        questions = load_corpus(config.dataset, config.paths.data_dir, require_complete=False, cache={})
    except Exception as exc:  # noqa: BLE001
        report.fail(f"{tag}:dataset", f"load {config.dataset} raised {type(exc).__name__}: {exc}")
        return
    tasks = resolve(spec.task_name)
    limit = corpus_limit(spec)
    total_cells = 0
    resolved = 0
    for task in tasks:
        try:
            pool = list(task.resolve_questions(config, questions))
        except Exception as exc:  # noqa: BLE001
            report.fail(f"{tag}:corpus", f"resolve_questions raised {type(exc).__name__}: {exc}")
            return
        if limit is not None:
            pool = pool[:limit]
        resolved = len(pool)
        specs = task.model_specs(config)
        if specs:
            from experiments.engine.driver import build_retrievers
            cells = task.generation_cells(config, pool, retrievers=build_retrievers(config))
            total_cells += len(cells) * len(specs) * max(1, len(config.visual_resolutions or (config.visual_resolution,)))
    if resolved == 0:
        report.fail(f"{tag}:corpus", "0 questions after scan/pool/sampling (nothing to run)")
    else:
        report.ok(f"{tag}:corpus", f"{resolved} questions -> ~{total_cells} reasoner cells")

    # Reasoner weights (every spec in the run) + classifier.
    for rspec in config.reasoner_specs or (config.reasoner_spec,):
        try:
            _check_weight(report, f"{tag}:reasoner:{rspec}", _reasoner_model_id(rspec), hard=True)
        except Exception as exc:  # noqa: BLE001
            report.fail(f"{tag}:reasoner:{rspec}", f"cannot resolve model id: {exc}")
    if config.classifier_spec:
        try:
            _check_weight(report, f"{tag}:classifier", _reasoner_model_id(config.classifier_spec), hard=True)
        except Exception as exc:  # noqa: BLE001
            report.fail(f"{tag}:classifier", f"cannot resolve model id: {exc}")

    # Parser env + model, only when a rung actually needs parser markdown (TL/TLV).
    reps = set(config.representations)
    if reps & {"TL", "TLV"}:
        env_python = ROOT / "envs" / f"parse-{config.parser_tool}" / "bin" / "python"
        if env_python.exists():
            report.ok(f"{tag}:parser-env", f"envs/parse-{config.parser_tool}/bin/python present")
        else:
            report.fail(f"{tag}:parser-env", f"{env_python} missing (TL/TLV cells will error)")
        # Soft: paddle loads from its own paddlex cache / an OCR floor, not always the
        # HF snapshot, so a "not staged" here is a hint, not proof. The parser smoke
        # (ops/scripts/final_probe.py) is the definitive parser check.
        _check_weight(report, f"{tag}:parser-model", _parser_model_id(config.parser_tool),
                      hard=False, consequence="paddle may use its own cache; confirm with the parser smoke")

    # Inference retriever arms are hard (a miss silently degrades ranking); the
    # benchmark method lists are soft (a missing method just drops its rows).
    _check_weight(report, f"{tag}:inf-text", _retriever_model_id(config.inference_text_retriever, "text"), hard=True)
    _check_weight(report, f"{tag}:inf-vision", _retriever_model_id(config.inference_vision_retriever, "vision"), hard=True)
    for name in config.text_retrievers:
        _check_weight(report, f"{tag}:bench-text:{name}", _retriever_model_id(name, "text"),
                      hard=False, consequence="its retrieval.jsonl rows will be skipped")
    for name in config.vision_retrievers:
        _check_weight(report, f"{tag}:bench-vision:{name}", _retriever_model_id(name, "vision"),
                      hard=False, consequence="its retrieval.jsonl rows will be skipped")

    # Resource sizing vs the requested --gres (soft: the operator may know better).
    if gres_gpus is not None:
        for rspec in config.reasoner_specs or (config.reasoner_spec,):
            from models import ModelSpec
            size = ModelSpec.parse(rspec).size
            if size == "8b" and (gres_type == "v100") and gres_gpus < 2:
                report.warn(f"{tag}:sizing", f"{rspec} (8B) usually needs 2 V100s; --gres has {gres_gpus}")
            if size == "32b" and gres_type == "v100":
                report.warn(f"{tag}:sizing", f"{rspec} (32B) does not fit V100s; run on the supervisor")


def _parse_gres(gres: str | None) -> tuple[int | None, str | None]:
    """Pull (gpu_count, gpu_type) out of a `gpu:v100:2` style --gres string."""

    if not gres:
        return None, None
    parts = gres.split(":")
    try:
        count = int(parts[-1])
    except ValueError:
        return None, None
    gpu_type = parts[1].lower() if len(parts) >= 3 else None
    return count, gpu_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="generation spec to preflight")
    parser.add_argument("--gres", help="the --gres you plan to submit with, for a sizing sanity check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = Report()

    # Importing the generate entry pulls the whole pipeline; a missing dep in the
    # core env surfaces here instead of after the job starts.
    try:
        from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs  # noqa: F401
        import experiments.engine.driver  # noqa: F401
        import models  # noqa: F401
        import ops.generate  # noqa: F401
        import pipeline.reasoner  # noqa: F401
        import retrievers.text  # noqa: F401
        import retrievers.vision  # noqa: F401
        import tools.parser  # noqa: F401
        report.ok("imports", "pipeline imports clean in the core env")
    except Exception as exc:  # noqa: BLE001
        report.fail("imports", f"{type(exc).__name__}: {exc}")
        report.print()
        return 1

    try:
        specs = load_yaml_specs(Path(args.spec))
    except Exception as exc:  # noqa: BLE001
        report.fail("spec.parse", f"{type(exc).__name__}: {exc}")
        report.print()
        return 1
    report.ok("spec.parse", f"{args.spec} -> {len(specs)} run(s)")

    # Point HF at the in-project offline cache the same way generate does, so the
    # staged-weight checks look in the right place on Kaya and locally.
    from ops.generate import ensure_cache_env
    ensure_cache_env(config_from_spec(specs[0]))

    gres_gpus, gres_type = _parse_gres(args.gres)
    for spec in specs:
        _check_run(report, spec, gres_gpus, gres_type)

    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
