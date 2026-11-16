"""Numerically checkable authored finance tasks; never imply real company results."""

import random
from decimal import Decimal, ROUND_HALF_UP
from .common import write_jsonl, digest


def example(index, split, seed):
    rng = random.Random(f"{seed}:{split}:{index}")
    previous = rng.randint(40, 150) * 10
    current = previous + rng.randint(-10, 40) * 5
    profit = rng.randint(5, 25) * 5
    growth = ((Decimal(current) / Decimal(previous) - 1) * 100).quantize(
        Decimal(".01"), rounding=ROUND_HALF_UP
    )
    company = f"Fictional {split.title()} Company {index:03d}"
    context = (
        f"{company} reported revenue of USD {current} million for 2025, "
        f"compared with USD {previous} million in 2024. Net income for 2025 was USD {profit} million. "
        "These are authored teaching figures, not a real company report."
    )
    kind = ["extract", "qa", "summary"][index % 3]
    if kind == "extract":
        question = "Extract 2025 revenue and net income. Return JSON with revenue_million and net_income_million."
        answer = __import__("json").dumps(
            {"revenue_million": current, "net_income_million": profit}
        )
        checks = {"revenue_million": current, "net_income_million": profit}
    elif kind == "qa":
        question = "What was year-over-year revenue growth in percent? Round to two decimals and answer only the number."
        answer = str(growth)
        checks = {"growth_percent": float(growth)}
    else:
        question = "Summarize 2025 revenue and net income in one factual sentence. Include both numbers and their units."
        answer = f"2025 revenue was USD {current} million and net income was USD {profit} million."
        checks = {"revenue_million": current, "net_income_million": profit}
    return {
        "id": f"{split}-{index:03d}",
        "split": split,
        "company": company,
        "task": kind,
        "context": context,
        "question": question,
        "answer": answer,
        "checks": checks,
        "provenance": "authored_synthetic",
    }


def prepare(config, output):
    result = {}
    for split in ["train", "validation", "test"]:
        rows = [
            example(i, split, config["seed"]) for i in range(config[split + "_size"])
        ]
        write_jsonl(output / (split + ".jsonl"), rows)
        result[split] = {
            "rows": len(rows),
            "sha256": digest(rows),
            "file": str(output / (split + ".jsonl")),
        }
    return result


def prompt(row):
    return f"{row['context']} {row['question']} "
