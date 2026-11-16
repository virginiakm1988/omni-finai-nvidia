# Demonstration playbook

## Intended result

Download and run the original Omni-FinAI 70B checkpoint for financial extraction, numerical question answering, and summarization. Train small LoRA adapters with NeMo AutoModel, reload the exported adapter, and compare the same held-out examples before and after training.

## 1. Prepare a reproducible session

Create an isolated environment, install the indicated extras, and record the Git commit with `git rev-parse HEAD`. Keep `configs/default.json` with the run. Inspect the source licenses before adding external data. The program saves a configuration SHA-256, mode, elapsed time, individual predictions, aggregate metrics, and failures. Use environment variables or Colab Secrets for credentials.

## 2. Inspect and test the core method

Run `prepare`, inspect the authored inputs, then run `demo --mode fixture`. Run `python -m unittest discover -s tests -v`. These tests check numerical scoring boundaries, prompt parity, fixture/live provenance, and data separation; they do not establish model quality.

The original profile selects the pinned 70B Omni-FinAI checkpoint. The small profile selects a different, explicitly labelled Qwen2.5 0.5B Instruct base for training mechanics. The authored financial reports use disjoint company IDs across training, validation, and test sets. AutoModel trains LoRA on training rows, validates separately, exports HF-compatible adapter weights, verifies nonzero LoRA-B updates, reloads them with PEFT, and scores the identical fixed test set.

## 3. Run the live demonstration

Set `HF_TOKEN` securely if the selected checkpoint requires gated access. The HF loader downloads the pinned checkpoint into its normal cache on first use. Optional `download --mode live --profile original --model-dir models` exports an additional local copy without GPU execution; inference still uses the normal cache, so this export needs extra disk space and is not required for the demo. The baseline and adapter evaluation need GPU resources. On Taipei-1, inspect current partitions and quota, then submit a bounded job. The provided `scripts/taipei1.slurm` intentionally leaves account/partition selection to the current environment.

Small-model job: one H100, at most one hour. Original 70B job: four H100s, at most two hours. Use an isolated environment with `nemo-automodel==0.7.0` or the official 26.10.00 container. The scripts fail if CUDA is absent or free memory is insufficient.

```bash
python -m project_demo prepare --output-dir runs/prepare
python -m project_demo train --mode live --profile small --gpus 1 --output-dir runs/automodel-small
# Separate multi-GPU allocation:
python -m project_demo demo --mode live --profile original --backend hf --output-dir runs/original-hf
python -m project_demo train --mode live --profile original --gpus 4 --output-dir runs/automodel-original
```

## 4. Review and evaluate

Check the scheduler exit code as well as `result.json`. Verify `training.log`, `before.json`, `after.json`, and the matching test hashes. A completed run must show nonzero LoRA-B tensors, saved adapter files, successful adapter reload, and before/after scores. Successful creation of YAML alone is not training. Use the three task metrics separately; low or worse scores are retained. The held-out test examples never become training examples.

## Acceptance and failure handling

Extraction checks every requested numeric field; QA checks the computed growth percentage. Summary scoring measures required numeric coverage only, not overall quality, faithfulness, or financial reasoning. Dataset and configuration hashes accompany each run.

A failure is part of the execution record. Resolve missing entitlement, invalid model ID, unsupported SDK release, malformed response, or insufficient GPU memory explicitly, then start a separate run. Do not edit old evidence to make it appear successful. The [validation file](../VALIDATION.json) is the source of truth for checks actually completed in this release.

## Method differences

This companion does not reproduce the paper training corpus, large-scale optimization, or benchmark results. It uses small synthetic financial reports with checkable arithmetic. The small profile is not Omni-FinAI. An optional NIM baseline is also labelled as a different model. A short LoRA run may reduce accuracy; the recorded result is retained.

## AutoModel 0.7 / recent Triton compatibility

The published 0.7 wheel contains a stale autotune key `S` for a kernel that has no `S` argument. Recent Triton rejects its import. In an isolated tutorial environment, run `python scripts/patch_automodel_0_7_triton.py` before training. This backports the exact upstream correction from commit `c57935fc64faea913ce9f5e38ea7e57e6416cc93`: the key becomes `[M, K]`. The script validates the package version, preserves the original file, and records before/after hashes. Taipei-1 evidence names this patch; it is not described as an unmodified wheel.

## Separate single-GPU and distributed environments

The measured small tutorial used `requirements-gpu-small-tested.txt`, plus the documented upstream Triton fix. The complete resolved package list is in `evidence/automodel-0.7-freeze.txt`. PyTorch2.15 removes an internal FSDP interface expected by the published AutoModel0.7 wheel, so the original70B distributed run uses a separate environment pinned by `requirements-gpu-distributed.txt`. Do not change packages in a running job. Check `VALIDATION.json` for the final measured status of that route.

```bash
python3.12 -m venv .venv-distributed
source .venv-distributed/bin/activate
python -m pip install -r requirements-gpu-distributed.txt
python scripts/patch_automodel_0_7_triton.py
python -m pip install --no-deps -e .
```

Set `PYTHONNOUSERSITE=1` so an unrelated user-site installation cannot shadow the isolated SDK environment. On shared clusters, run GPU work only inside the bounded scheduler allocation.

## Submit the prepared environment

From the repository root, select your current permitted partition and submit the separate scripts. For example, on the validated shared partition:

```bash
# Original70B environment prepared above; four GPUs, at most two hours:
sbatch --partition=shared --export=ALL,AUTOMODEL_ENV="$PWD/.venv-distributed" scripts/taipei1-original.slurm
# Separate small-model environment; one GPU, at most one hour:
sbatch --partition=shared --export=ALL,AUTOMODEL_ENV="$PWD/.venv-small" scripts/taipei1.slurm
```

Before and after inference run in isolated subprocesses. Each process exits before the next stage, releasing the original70B weights before FSDP training starts. Failed worker processes stop the pipeline and keep their logs.
