# Kaya Setup and Pipeline Runbook

This is the end-to-end procedure for running MP-VRDU on UWA Kaya. Commands
marked **LOGIN** use the internet-facing login node. Commands marked **COMPUTE**
must run through SLURM. Never load models or run experiments on the login node.

Kaya module names, GPU partitions, and driver versions can change. Confirm them
with UWA HPC documentation or `module avail` before the first installation.

## 1. Clone and Configure the Repository (LOGIN)

Use `/group` for environments, datasets, models, results, and application logs:

```bash
ssh <username>@kaya.hpc.uwa.edu.au
mkdir -p /group/<project>/repos
cd /group/<project>/repos
git clone <repository-url> CITS4011
cd CITS4011
mkdir -p logs       # required before sbatch opens logs/%x_%j.{out,err}
```

Edit `scripts/kaya/env.sh` and replace `/group/CHANGE_ME`. Verify or override:

```bash
export MPVRDU_GROUP=/group/<project>
export MPVRDU_ANACONDA=Anaconda3/2021.05
export MPVRDU_CUDA=cuda/<available-version>
export MPVRDU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX
export MPVRDU_MINERU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX
```

Persist these values in `env.sh`; batch jobs source that file independently.
Select PyTorch wheel indexes compatible with Kaya's NVIDIA driver. MinerU needs
PyTorch 2.6 or newer, so its index may differ from the main environment.

Useful discovery commands:

```bash
module avail Anaconda
module avail cuda
sinfo --noheader --format="%P"
scontrol show partition gpu
```

The supplied SBATCH files assume `--partition=gpu` and `--gres=gpu:1`. Change
those directives if Kaya currently uses different names.

## 2. Create Both Environments (LOGIN)

The main environment cannot also host MinerU: MP-VRDU pins Transformers 5.3 for
Qwen/ColPali, while MinerU requires Transformers below 5.

```bash
bash scripts/kaya/setup_conda_env.sh
bash scripts/kaya/setup_mineru_env.sh
```

The resulting environments are:

```text
/group/<project>/conda_environments/mpvrdu
/group/<project>/conda_environments/mineru
```

The full grid also needs the Tesseract executable. Install it into the main
environment if `command -v tesseract` fails:

```bash
source scripts/kaya/env.sh
load_modules
conda install -p "$MPVRDU_ENV" -c conda-forge tesseract -y
```

Check the installations without running model inference:

```bash
source scripts/kaya/env.sh
load_modules
activate_env
python -m pip check
python -c "import torch, transformers, colpali_engine; print(torch.__version__, transformers.__version__)"
"$MPVRDU_MINERU_PYTHON" -m pip check
tesseract --version
```

## 3. Stage Data and Every Model (LOGIN)

Compute nodes have no internet. The following downloads the full
`yubo2333/MMLongBench-Doc` dataset and all models referenced by the Kaya grid,
hand-written dense configs, visual retrievers, LLM judge/reranker, and MinerU:

```bash
source scripts/kaya/env.sh
bash scripts/kaya/prestage.sh
```

This can require tens of gigabytes. Check quota first:

```bash
quota -s
du -sh "$HF_HOME" "$MPVRDU_MMLB_DIR"
```

`prestage.sh` is safe to rerun after an interrupted download. To use the 32B
generator, uncomment its download in that script before staging.

Important staged paths:

| Artifact | Location |
|---|---|
| Hugging Face and MinerU models | `$HF_HOME` |
| PDFs and benchmark parquet | `$MPVRDU_MMLB_DIR` |
| rendered page cache | `$MPVRDU_RENDER_CACHE` |
| results | `$MPVRDU_RESULTS` |
| application logs | `$MPVRDU_LOGS` |
| MinerU local-model map | `$MINERU_TOOLS_CONFIG_JSON` |

## 4. Validate Configuration (LOGIN)

These commands only inspect configuration and do not load models:

```bash
source scripts/kaya/env.sh
load_modules
activate_env
python scripts/run_grid.py \
  --suite experiments/grid_kaya_7b.yaml \
  --results-root "$MPVRDU_RESULTS/grid" \
  --logs-root "$MPVRDU_LOGS" \
  --dry-run
```

The suite should expand to 64 conditions. Confirm the dataset is visible:

```bash
python -c "from mpvrdu.config import DataConfig; from mpvrdu.data.load import load_dataset; d=load_dataset(DataConfig(slice='full', max_questions=1)); print(len(d), list(d.documents))"
```

## 5. Validate the GPU Stack (COMPUTE)

Submit the supplied five-minute job:

```bash
sbatch scripts/kaya/gpu_test.sbatch
squeue -u "$USER"
```

Inspect `logs/mpvrdu_gpu_test_<jobid>.out`. Both the main and MinerU
environments must report CUDA available. If either reports `False`, stop and
fix the CUDA module, driver-compatible PyTorch wheel, or GPU request.

For interactive debugging:

```bash
srun --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=1:00:00 --pty /bin/bash -l
source scripts/kaya/env.sh
load_modules
activate_env
set_offline
# run diagnostics, then always:
exit
```

## 6. Run Smoke Tests (COMPUTE)

First run the synthetic, mocked pipeline:

```bash
sbatch scripts/kaya/run_config.sbatch configs/smoke.yaml
```

Then run a small real-model check over two questions:

```bash
sbatch scripts/kaya/run_grid.sbatch \
  --only baselines --max-questions 2
```

Do not start the full grid until both jobs finish successfully. The small real
run verifies offline Qwen loading, PDF rendering, generation, scoring, result
writing, and per-condition logging.

## 7. Run One Pipeline Condition

Submit any single YAML configuration:

```bash
sbatch scripts/kaya/run_config.sbatch configs/oracle.yaml
sbatch scripts/kaya/run_config.sbatch configs/subA/colpali_image.yaml
sbatch scripts/kaya/run_config.sbatch configs/subB/dense_mineru_text.yaml
```

The script writes JSONL to `$MPVRDU_RESULTS/<config-name>.jsonl` and a detailed
application log to `$MPVRDU_LOGS/config/`. A rerun of the same single config
overwrites that fixed JSONL path.

## 8. Run the Full Resumable Grid

The default command uses Qwen2.5-VL-7B:

```bash
sbatch scripts/kaya/run_grid.sbatch
```

For easier scheduling, run sub-studies separately:

```bash
sbatch scripts/kaya/run_grid.sbatch --only baselines
sbatch scripts/kaya/run_grid.sbatch --only A_retrieval
sbatch scripts/kaya/run_grid.sbatch --only B_parser
sbatch scripts/kaya/run_grid.sbatch --only B_chunking
sbatch scripts/kaya/run_grid.sbatch --only C_modality
```

Run these sequentially unless you intentionally want several GPUs/jobs writing
to the shared summary files. Resubmitting the same command skips completed
conditions. If a wall-time limit kills a condition, that partial condition
restarts while earlier completed JSONL files are retained.

To use the staged 32B model:

```bash
sbatch --export=ALL,MPVRDU_MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
  scripts/kaya/run_grid.sbatch --only baselines
```

The supplied job requests one GPU. A 32B BF16 model requires an appropriately
sized GPU; do not assume it fits an A100 40GB. Adjust GRES or quantization only
as an explicit experimental change.

## 9. Monitor Runs and Read Errors

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES
scancel <jobid>
tail -f logs/mpvrdu_grid_<jobid>.out
```

SLURM captures the outer process in `logs/*.out` and `logs/*.err`. The
repository also writes:

```text
$MPVRDU_LOGS/grid_kaya_7b__<timestamp>.log
$MPVRDU_LOGS/grid_kaya_7b/<substudy>/<run>__<hash>.log
```

Per-run logs contain the resolved config, model output, Python traceback, and
subprocess exit code. Check these first when a condition is absent from the
summary.

## 10. Generate and Inspect Results

`run_grid.sbatch` generates a report after each job. Regenerate it manually:

```bash
source scripts/kaya/env.sh
load_modules
activate_env
python scripts/report.py \
  --results "$MPVRDU_RESULTS/grid" \
  --out "$MPVRDU_RESULTS/grid/report.md"
```

Outputs include:

```text
$MPVRDU_RESULTS/grid/summary.md
$MPVRDU_RESULTS/grid/report.md
$MPVRDU_RESULTS/grid/comparisons.csv
$MPVRDU_RESULTS/grid/figures/
```

Inspect one run or list all result files:

```bash
python scripts/inspect_run.py "$MPVRDU_RESULTS/grid"
python scripts/inspect_run.py <result.jsonl> --errors-only
python scripts/inspect_run.py <result.jsonl> --retrieval-misses --full
```

## 11. Common Failures

- **Offline model error:** rerun `prestage.sh` on the login node. Do not disable
  offline mode on compute nodes.
- **CUDA unavailable or undefined symbol:** the loaded CUDA module, NVIDIA
  driver, and selected PyTorch wheel are incompatible.
- **ColPali adapter mismatch/OOM:** verify the main environment retained
  `transformers>=5.3,<5.4` and `torch<2.12`.
- **MinerU tries to download:** verify `MINERU_MODEL_SOURCE=local`,
  `$MINERU_TOOLS_CONFIG_JSON`, and `$MPVRDU_MINERU_PYTHON`.
- **Tesseract condition missing:** verify `tesseract --version` inside
  `$MPVRDU_ENV`.
- **SLURM job never starts:** inspect partition/GRES availability with `sinfo`
  and `squeue -p gpu`.
- **No SLURM log:** create the repository's `logs/` directory before `sbatch`;
  SLURM opens output files before executing the script.
- **Quota or permission error:** run `quota -s`, `df -h`, and confirm every
  `MPVRDU_*` path points to writable `/group` storage.

## 12. Day-to-Day Sequence

```text
LOGIN:   git pull
LOGIN:   update env/model cache if dependencies or configs changed
LOGIN:   run the grid dry-run
COMPUTE: submit gpu test after environment/CUDA changes
COMPUTE: submit a small real-model run after model/config changes
COMPUTE: submit or resume the required sub-study
LOGIN:   inspect logs, regenerate report, archive results
```
