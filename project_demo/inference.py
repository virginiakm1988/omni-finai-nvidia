"""HF checkpoint inference and an explicit NVIDIA NIM baseline, never substituted."""

import json
import re
from pathlib import Path
from .common import clean_json
from .data import prompt


def selected_model(config, profile):
    if profile not in config["profiles"]:
        raise ValueError("profile must be small or original")
    return config["profiles"][profile]


def gpu_preflight(profile):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "HF and AutoModel execution requires a CUDA GPU. Use the Taipei-1 recipe."
        )
    free = sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()))
    required = 145 if profile == "original" else 6
    if free / (1024**3) < required:
        raise RuntimeError(
            f"{profile} BF16 inference requires approximately {required} GiB total free GPU memory; found {free/1024**3:.1f} GiB"
        )
    return {
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "total_free_gib": round(free / 1024**3, 2),
    }


def load_hf(config, profile, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_preflight(profile)
    spec = selected_model(config, profile)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    if adapter:
        from peft import PeftModel

        adapter_config = json.loads((Path(adapter) / "adapter_config.json").read_text())
        expected = adapter_config.get("base_model_name_or_path")
        if expected and expected != spec["model_id"]:
            raise ValueError("Adapter base model does not match selected model")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return tokenizer, model


def encode_prompt(tokenizer, row, prompt_format):
    if prompt_format == "checkpoint-chat":
        if not tokenizer.chat_template:
            raise ValueError(
                "Selected checkpoint has no chat template; explicitly select plain format"
            )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    if prompt_format != "plain":
        raise ValueError("Unknown prompt format")
    return tokenizer(prompt(row), return_tensors="pt")


def infer_hf(rows, config, profile, adapter=None, prompt_format="plain"):
    # AutoModel before/after comparisons deliberately default to the exact plain
    # ColumnMapped training format. The interactive HF demo selects checkpoint-chat.

    import torch

    tokenizer, model = load_hf(config, profile, adapter)
    predictions = []
    for row in rows:
        inputs = encode_prompt(tokenizer, row, prompt_format).to(
            model.get_input_embeddings().weight.device
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        predictions.append(
            {
                "id": row["id"],
                "task": row["task"],
                "prediction": text,
                "checks": row["checks"],
            }
        )
    del model
    torch.cuda.empty_cache()
    return predictions


def infer_nim(rows, config, output_dir):
    from .nim_client import NIMClient

    client = NIMClient(
        output_dir=output_dir,
        max_requests=config["max_requests"],
        base_url=config["base_url"],
    )
    return [
        {
            "id": row["id"],
            "task": row["task"],
            "checks": row["checks"],
            "prediction": client.chat_text(
                [{"role": "user", "content": prompt(row)}],
                model=config["model"],
                max_tokens=config["max_new_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        }
        for row in rows
    ]


def score_one(row):
    text = row["prediction"]
    checks = row["checks"]
    if row["task"] == "extract":
        try:
            parsed = clean_json(text)
            return all(
                type(parsed.get(k)) in {int, float} and abs(parsed[k] - v) < 0.005
                for k, v in checks.items()
            )
        except (ValueError, TypeError, AttributeError):
            return False
    if row["task"] == "qa":
        try:
            return (
                abs(float(text.strip().rstrip("%")) - checks["growth_percent"]) < 0.011
            )
        except ValueError:
            return False
    # A transparent numerical coverage check, not a summary quality or faithfulness score.
    numbers = {
        float(n.replace(",", ""))
        for n in re.findall(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?", text)
    }
    return all(v in numbers for v in checks.values()) and "million" in text.lower()


def evaluate(rows):
    if not rows:
        raise ValueError("Cannot evaluate empty results")
    grouped = {
        task: [row for row in rows if row["task"] == task]
        for task in ["extract", "qa", "summary"]
    }
    return {
        task: {
            "n": len(items),
            "pass_rate": (
                sum(score_one(row) for row in items) / len(items) if items else None
            ),
            "metric": (
                "numeric_coverage_only"
                if task == "summary"
                else "deterministic_correctness"
            ),
        }
        for task, items in grouped.items()
    }
