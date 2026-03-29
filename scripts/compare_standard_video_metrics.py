#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "recon_mse",
    "recon_psnr",
    "recon_ssim",
    "future_mse",
    "future_psnr",
    "future_ssim",
)


def parse_entry(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Invalid entry '{raw}'. Expected NAME=PATH."
        )
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"Invalid entry '{raw}'. Expected NAME=PATH."
        )
    return name, Path(path)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"Summary at {path} is not a JSON object.")
    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["name", *METRIC_KEYS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row["name"], *[format_value(row.get(key)) for key in METRIC_KEYS]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standard video metric summary JSON files."
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Comparison entry in the form NAME=PATH_TO_SUMMARY_JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the comparison as JSON.",
    )
    args = parser.parse_args()

    entries = [parse_entry(raw) for raw in args.entry]
    rows: list[dict[str, Any]] = []
    for name, path in entries:
        summary = load_summary(path)
        row = {"name": name, "path": str(path)}
        for key in METRIC_KEYS:
            row[key] = summary.get(key)
        rows.append(row)

    print(build_markdown_table(rows))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
        print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
