"""Small, auditable NIM client shared as a vendored module by the demo repos.

Every HTTP attempt is reserved before transmission, including failed requests.
There are no implicit retries or fixture fallbacks. Credentials never enter logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class NIMClient:
    """Call NVIDIA's OpenAI-compatible API with a persistent attempt budget.

    Use one output directory per experiment. Reusing it resumes the SAME budget.
    The SQLite transaction makes reservations safe across client processes.
    """

    def __init__(
        self,
        *,
        api_key=None,
        base_url="https://integrate.api.nvidia.com/v1",
        output_dir="runs/nim",
        max_requests=20,
        timeout=90,
    ):
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.timeout = timeout
        self.database = self.output_dir / "attempts.sqlite3"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, started TEXT, endpoint TEXT, status TEXT)"
            )

    def _redact(self, value):
        return value.replace(self.api_key, "[REDACTED]") if self.api_key else value

    def _reserve(self, endpoint):
        with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if count >= self.max_requests:
                raise RuntimeError(
                    f"NIM attempt budget exhausted ({count}/{self.max_requests})"
                )
            cursor = db.execute(
                "INSERT INTO attempts(started,endpoint,status) VALUES(?,?,?)",
                (utc_now(), endpoint, "reserved"),
            )
            return cursor.lastrowid

    def request(self, endpoint, payload):
        if not self.api_key:
            raise RuntimeError(
                "Set NVIDIA_API_KEY in the environment or Colab Secrets before live execution"
            )
        if not endpoint.startswith("/") or ".." in endpoint:
            raise ValueError("endpoint must be an absolute API path")
        attempt_id = self._reserve(endpoint)
        started = time.monotonic()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        record = {
            "attempt_id": attempt_id,
            "mode": "live",
            "started_at": utc_now(),
            "endpoint": self.base_url + endpoint,
            "model": payload.get("model"),
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "status": "running",
        }
        path = self.output_dir / f"attempt-{attempt_id:06d}.json"
        write_json(path, record)
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
            record.update(status="succeeded", response=result)
            return result
        except urllib.error.HTTPError as exc:
            detail = self._redact(exc.read().decode("utf-8", errors="replace")[:3000])
            record.update(status="failed", http_status=exc.code, error=detail)
            raise RuntimeError(f"NIM HTTP {exc.code}: {detail}") from None
        except Exception as exc:
            record.update(status="failed", error=self._redact(str(exc)))
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["finished_at"] = utc_now()
            # Redact echoed credentials even if a remote server returns them in JSON.
            safe_record = json.loads(
                self._redact(json.dumps(record, ensure_ascii=False))
            )
            write_json(path, safe_record)
            with closing(sqlite3.connect(self.database)) as db, db:
                db.execute(
                    "UPDATE attempts SET status=? WHERE id=?",
                    (record["status"], attempt_id),
                )

    def chat(self, messages, *, model, max_tokens=512, temperature=0, extra_body=None):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if extra_body:
            if {"model", "messages", "stream"} & set(extra_body):
                raise ValueError(
                    "extra_body cannot override model, messages, or stream"
                )
            payload.update(extra_body)
        return self.request("/chat/completions", payload)

    def chat_text(self, messages, **kwargs):
        response = self.chat(messages, **kwargs)
        text = response["choices"][0]["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "NIM returned no final text; inspect saved response/finish_reason"
            )
        return text

    def embeddings(self, texts, *, model, input_type="passage"):
        response = self.request(
            "/embeddings",
            {
                "model": model,
                "input": texts,
                "input_type": input_type,
                "truncate": "NONE",
            },
        )
        return [
            row["embedding"]
            for row in sorted(response["data"], key=lambda row: row["index"])
        ]

    def rerank(self, query, passages, *, model):
        return self.request(
            "/ranking",
            {
                "model": model,
                "query": {"text": query},
                "passages": [{"text": text} for text in passages],
            },
        )
