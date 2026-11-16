"""One-process HF evaluation; process exit releases all model CUDA contexts."""

import argparse
from pathlib import Path
from .common import read_json, read_jsonl, write_json
from .inference import infer_hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--profile", required=True, choices=["small", "original"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    result = infer_hf(
        rows, read_json(args.config), args.profile, args.adapter, prompt_format="plain"
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
