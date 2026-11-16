"""Run evidence and configuration, with no implicit network or mode switching."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_config(path):
    result = read_json(path)
    result["_config_dir"] = str(Path(path).resolve().parent)
    if (
        type(result.get("max_requests", 20)) is not int
        or result.get("max_requests", 20) < 1
    ):
        raise ValueError("max_requests must be a positive integer")
    return result


def data_path(config, key):
    path = Path(config[key])
    return path if path.is_absolute() else Path(config["_config_dir"]) / path


def clean_json(text):
    # JSON fences may wrap an otherwise valid provider response. No repair/fabrication.
    value = text.strip()
    if value.startswith("```"):
        value = "\n".join(value.splitlines()[1:-1])
    return json.loads(value)


class Run:
    def __init__(self, output_dir, config, mode, command):
        if mode not in {"fixture", "live"}:
            raise ValueError("mode must be fixture or live")
        self.base = Path(output_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.path.mkdir()
        self.started = time.monotonic()
        safe_config = {k: v for k, v in config.items() if not k.startswith("_")}
        self.record = {
            "command": command,
            "mode": mode,
            "status": "running",
            "config": safe_config,
            "config_sha256": digest(safe_config),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
        }
        self.save()

    def save(self):
        write_json(self.path / "result.json", self.record)
        write_json(
            self.base / "latest.json",
            {"run": str(self.path), "status": self.record["status"]},
        )

    def finish(self, result):
        self.record.update(
            status="succeeded",
            result=result,
            elapsed_seconds=round(time.monotonic() - self.started, 4),
        )
        self.save()
        return self.record

    def fail(self, error):
        message = str(error)
        for key in ("NVIDIA_API_KEY", "HF_TOKEN"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        self.record.update(
            status="failed",
            error=message,
            error_type=type(error).__name__,
            elapsed_seconds=round(time.monotonic() - self.started, 4),
        )
        self.save()


def sdk_versions():
    result = {}
    for name in (
        "nemo-fabric",
        "nemo-fabric-adapter-contract",
        "nemo-relay",
        "data-designer",
        "nemo-automodel",
        "transformers",
        "torch",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def evaluation_source(path):
    source_path = Path(path)
    saved = read_json(source_path)
    if saved.get("status") != "succeeded" or saved.get("mode") not in {
        "fixture",
        "live",
    }:
        raise ValueError(
            "Evaluation requires a successful, explicitly labelled source run"
        )
    if not isinstance(saved.get("config"), dict):
        raise ValueError("Source configuration is missing")
    return saved, {
        "source_file": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_mode": saved["mode"],
        "model_called": False,
    }
