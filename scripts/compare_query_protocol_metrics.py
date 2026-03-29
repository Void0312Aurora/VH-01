#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "cond_acc",
    "cond_true_prob",
    "cond_true_in90",
    "cond_support_ratio",
    "query_exec_mse",
    "query_oracle_mse",
    "query_match_true",
    "query_exec_set_size",
)


def parse_entry(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Invalid entry '{raw}'. Expected NAME=PATH.")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip())


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"Summary at {path} is not a JSON object.")
    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare protocol-B query/readout metric summary JSON files.")
    parser.add_argument("--entry", action="append", required=True, help="Entry in the form NAME=PATH.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for raw in args.entry:
        name, path = parse_entry(raw)
        summary = load_summary(path)
        row = {"name": name, "path": str(path)}
        for key in METRIC_KEYS:
            row[key] = summary.get(key)
        rows.append(row)

    headers = ["name", *METRIC_KEYS]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [row["name"], *[format_value(row.get(key)) for key in METRIC_KEYS]]
        print("| " + " | ".join(values) + " |")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
        print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
