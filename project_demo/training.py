"""Actual NeMo AutoModel recipe launch followed by exported adapter validation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from .common import write_json, read_json, read_jsonl, digest
from .inference import selected_model, infer_hf, evaluate, gpu_preflight


def recipe(config, profile, data_dir, checkpoint_dir, gpus):
    spec = selected_model(config, profile)
    dataset = lambda split: {
        "_target_": "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset",
        "path_or_dataset_id": str((data_dir / (split + ".jsonl")).resolve()),
        "column_mapping": {
            "context": "context",
            "question": "question",
            "answer": "answer",
        },
        "seq_length": 512,
        "answer_only_loss_mask": True,
        "padding": "do_not_pad",
        "truncation": "do_not_truncate",
        "use_hf_chat_template": False,
    }
    loader = {
        "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
        "collate_fn": {
            "_target_": "nemo_automodel.components.datasets.utils.default_collater"
        },
        "shuffle": False,
    }
    return {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        "seed": config["seed"],
        "step_scheduler": {
            "global_batch_size": max(4, gpus),
            "local_batch_size": 1,
            "max_steps": config["train_steps"],
            "num_epochs": 2,
            "ckpt_every_steps": config["train_steps"],
            "val_every_steps": config["train_steps"],
        },
        "dist_env": {"backend": "nccl", "timeout_minutes": 10},
        "distributed": {
            "strategy": "fsdp2",
            "dp_size": None,
            "tp_size": 1,
            "cp_size": 1,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": spec["model_id"],
            "revision": spec["revision"],
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
        },
        "peft": {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "target_modules": "*_proj",
            "dim": 8,
            "alpha": 32,
            "use_triton": False,
        },
        "loss_fn": {
            "_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
        },
        "dataset": dataset("train"),
        "validation_dataset": dataset("validation"),
        "dataloader": loader,
        "validation_dataloader": loader,
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "model_save_format": "safetensors",
            "save_consolidated": "final",
        },
    }


def verify_adapter(path):
    from safetensors import safe_open

    config_path = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config_path.exists() or not weights.exists():
        raise RuntimeError("Expected HF-compatible exported adapter files are missing")
    with safe_open(weights, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        b_keys = [key for key in keys if "lora_B" in key or "lora_b" in key]
        nonzero = sum(
            bool(source.get_tensor(key).abs().sum().item() > 0) for key in b_keys
        )
    if not b_keys or not nonzero:
        raise RuntimeError(
            "No nonzero LoRA B update found; cannot claim successful training"
        )
    return {
        "tensor_count": len(keys),
        "nonzero_lora_B_tensors": nonzero,
        "exported_adapter": str(path),
    }


def isolated_hf_evaluation(config, profile, data_path, run_dir, stage, adapter=None):
    """Never retain a 70B baseline's GPU memory in the training parent process."""
    config_path = run_dir / "hf-worker-config.json"
    predictions_path = run_dir / (stage + "-predictions.json")
    write_json(config_path, config)
    command = [
        sys.executable,
        "-m",
        "project_demo.hf_worker",
        "--config",
        str(config_path),
        "--data",
        str(data_path),
        "--profile",
        profile,
        "--output",
        str(predictions_path),
    ]
    if adapter is not None:
        command += ["--adapter", str(adapter)]
    with (run_dir / (stage + "-inference.log")).open("w", encoding="utf-8") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800
        )
    return read_json(predictions_path)


def train(config, profile, data_dir, run_dir, gpus):
    import yaml
    from importlib.metadata import version

    if version("nemo-automodel") != "0.7.0":
        raise RuntimeError("Use pinned nemo-automodel==0.7.0")
    if gpus not in {1, 2, 4, 8}:
        raise ValueError("Supported tutorial GPU counts: 1,2,4,8")
    hardware = gpu_preflight(profile)
    test = read_jsonl(data_dir / "test.jsonl")
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val = read_jsonl(data_dir / "validation.jsonl")
    groups = [{row["company"] for row in items} for items in [train_rows, val, test]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Company leakage across train/validation/test")
    before = isolated_hf_evaluation(
        config, profile, data_dir / "test.jsonl", run_dir, "before"
    )
    write_json(
        run_dir / "before.json",
        {
            "predictions": before,
            "metrics": evaluate(before),
            "test_sha256": digest(test),
        },
    )
    checkpoint_dir = run_dir / "checkpoints"
    cfg = recipe(config, profile, data_dir, checkpoint_dir, gpus)
    recipe_path = run_dir / "automodel.yaml"
    recipe_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    executable = shutil.which("automodel")
    if executable is None:
        raise RuntimeError(
            "automodel CLI not found; install the training extra in the active environment"
        )
    command = [executable, f"--nproc-per-node={gpus}", str(recipe_path)]
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=7200
        )
    candidates = list(checkpoint_dir.glob("epoch_*/model/adapter_config.json"))
    if not candidates:
        raise RuntimeError("Training did not export an adapter")
    adapter = max(candidates, key=lambda p: p.stat().st_mtime).parent
    proof = verify_adapter(adapter)
    after = isolated_hf_evaluation(
        config, profile, data_dir / "test.jsonl", run_dir, "after", adapter
    )
    write_json(
        run_dir / "after.json",
        {"predictions": after, "metrics": evaluate(after), "test_sha256": digest(test)},
    )
    return {
        "hardware": hardware,
        "inference_process_isolation": True,
        "before": evaluate(before),
        "after": evaluate(after),
        "adapter_validation": proof,
        "test_sha256": digest(test),
        "training_log": str(run_dir / "training.log"),
        "note": "Any metric decrease is retained; this small authored dataset does not establish financial expertise.",
    }
