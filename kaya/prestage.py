"""Stage Hugging Face models and MMLongBench-Doc on Kaya's login node."""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kaya.download_hf import snapshot, stage_mmlongbench_from_hub
from kaya.kaya import load_config


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dataset", action="store_true", help="do not stage MMLongBench-Doc")
    parser.add_argument("--skip-models", action="store_true", help="do not download configured models")
    parser.add_argument("--model-id", action="append", help="model repo id to stage; repeatable")
    parser.add_argument("--revision", help="optional Hugging Face revision for all downloads")
    parser.add_argument("--force-download", action="store_true", help="force Hugging Face redownload")
    parser.add_argument("--copy", action="store_true", help="copy staged MMLongBench files instead of symlinking")
    parser.add_argument("--max-workers", type=int, help="override config hf.max_workers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT / "kaya/config.json")
    cache_dir = Path(config.remote_path("cache"))
    data_dir = Path(config.remote_path("data"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_workers = args.max_workers or int(config.raw.get("hf", {}).get("max_workers", 8))
    print(
        "[prestage] start "
        f"remote_root={config.remote_root} cache_dir={cache_dir} data_dir={data_dir} "
        f"skip_models={args.skip_models} skip_dataset={args.skip_dataset} "
        f"max_workers={max_workers} force_download={args.force_download} "
        f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'missing'}"
    )

    if not args.skip_models:
        model_ids = args.model_id or list(config.raw["models"])
        print(f"[prestage] staging {len(model_ids)} model(s): {', '.join(model_ids)}")
        for model_id in model_ids:
            print(f"[prestage] downloading model {model_id}")
            path = snapshot(
                model_id,
                "model",
                args.revision,
                cache_dir,
                force_download=args.force_download,
                max_workers=max_workers,
            )
            print(f"model {model_id} -> {path}")
    else:
        print("[prestage] skipping model downloads")

    if not args.skip_dataset:
        dataset_id = config.raw["datasets"]["mmlongbench"]
        print(f"[prestage] downloading/staging dataset {dataset_id}")
        staged = stage_mmlongbench_from_hub(
            dataset_id,
            args.revision,
            cache_dir,
            data_dir,
            args.copy,
            args.force_download,
        )
        print(f"mmlongbench staged -> {staged}")
    else:
        print("[prestage] skipping dataset staging")

    print("[prestage] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
