import argparse
import json
import shutil
from pathlib import Path
from .common import (
    evaluation_source,
    digest,
    Run,
    load_config,
    read_json,
    read_jsonl,
    write_json,
    sdk_versions,
)
from .data import prepare, example
from .inference import infer_hf, infer_nim, evaluate, selected_model


def main():
    parser = argparse.ArgumentParser(
        description="Omni-FinAI HF inference and NeMo AutoModel companion"
    )
    parser.add_argument(
        "command",
        choices=["prepare", "demo", "evaluate", "download", "train", "recipe"],
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    parser.add_argument(
        "--task", choices=["all", "extract", "qa", "summary"], default="all"
    )
    parser.add_argument("--profile", choices=["small", "original"], default=None)
    parser.add_argument(
        "--prompt-format",
        choices=["checkpoint-chat", "plain"],
        default="checkpoint-chat",
    )
    parser.add_argument("--backend", choices=["hf", "nim"], default="hf")
    parser.add_argument("--output-dir", default="runs/demo")
    parser.add_argument("--data-dir", default="data/generated")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--results")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    if args.command == "evaluate" and not args.results:
        parser.error("evaluate requires --results; it never calls a model")
    config = load_config(args.config)
    profile = args.profile or config["profile"]
    config["profile"] = profile
    config["prompt_format"] = "plain" if args.command == "train" else args.prompt_format
    run = Run(args.output_dir, config, args.mode, args.command)
    try:
        data_dir = Path(args.data_dir).resolve()
        if args.command == "prepare":
            result = {
                "datasets": prepare(config, data_dir),
                "model": selected_model(config, profile),
                "model_called": False,
            }
        elif args.command == "download":
            if args.mode != "live":
                raise ValueError("Model download requires --mode live")
            from huggingface_hub import snapshot_download

            spec = selected_model(config, profile)
            target = Path(args.model_dir).resolve() / profile
            target.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target).free / (1024**3) < spec["min_free_disk_gib"]:
                raise RuntimeError("Not enough disk space for the selected checkpoint")
            location = snapshot_download(
                spec["model_id"], revision=spec["revision"], local_dir=str(target)
            )
            result = {"model": spec, "snapshot": location, "model_called": False}
        elif args.command == "recipe":
            import yaml
            from .training import recipe

            prepare(config, data_dir)
            value = recipe(
                config, profile, data_dir, run.path / "checkpoints", args.gpus
            )
            (run.path / "automodel.yaml").write_text(
                yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
            )
            result = {"recipe": value, "executed": False}
        elif args.command == "train":
            if args.mode != "live":
                raise ValueError(
                    "Training requires explicit --mode live; no simulated weights are saved"
                )
            from .training import train

            if not (data_dir / "train.jsonl").exists():
                prepare(config, data_dir)
            result = train(config, profile, data_dir, run.path, args.gpus)
        elif args.command == "evaluate" and args.results:
            previous, provenance = evaluation_source(args.results)
            run.record.update(
                mode=previous["mode"],
                config=previous["config"],
                config_sha256=digest(previous["config"]),
            )
            result = {
                "metrics": evaluate(previous["result"]["predictions"]),
                **provenance,
            }
        else:
            rows = [example(i, "test", config["seed"]) for i in range(3)]
            if args.task != "all":
                rows = [row for row in rows if row["task"] == args.task]
            if args.mode == "fixture":
                predictions = [
                    {
                        "id": r["id"],
                        "task": r["task"],
                        "prediction": r["answer"],
                        "checks": r["checks"],
                    }
                    for r in rows
                ]
                label = "authored reference answers; not model output"
            elif args.backend == "nim":
                predictions = infer_nim(rows, config, run.path / "nim")
                label = "NIM baseline; NOT Omni-FinAI"
            else:
                predictions = infer_hf(
                    rows, config, profile, prompt_format=args.prompt_format
                )
                label = selected_model(config, profile)["model_id"]
            result = {
                "predictions": predictions,
                "metrics": evaluate(predictions),
                "model_label": label,
                "prompt_format": (
                    args.prompt_format if args.backend == "hf" else "NIM user message"
                ),
                "sdk_versions": sdk_versions(),
                "profile": profile,
            }
        record = run.finish(result)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "result_file": str(run.path / "result.json"),
                }
            )
        )
        return 0
    except Exception as error:
        run.fail(error)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": run.record["error"],
                    "result_file": str(run.path / "result.json"),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
