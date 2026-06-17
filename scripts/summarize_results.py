"""Generate a compact Markdown summary from result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_metrics(results: dict) -> str:
    rows = ["| Head | Accuracy | AUC | F1 | Recall |", "|---|---:|---:|---:|---:|"]
    for head in ("video", "audio", "any"):
        m = results["metrics"][head]
        rows.append(
            f"| {head} | {fmt(m['accuracy'])} | {fmt(m['auc'])} | {fmt(m['f1'])} | {fmt(m['recall'])} |"
        )
    return "\n".join(rows)


def render_categories(results: dict) -> str:
    rows = ["| Category | n | any accuracy | any recall | any F1 |", "|---|---:|---:|---:|---:|"]
    for row in results["per_category"]:
        rows.append(
            f"| {row['category']} | {row['n']} | {fmt(row['any_acc'])} | {fmt(row['any_recall'])} | {fmt(row['any_f1'])} |"
        )
    return "\n".join(rows)


def render_ablation(path: Path) -> str:
    if not path.exists():
        return "_Ablation file not found._"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = ["| Config | any AUC | any F1 | video AUC | audio AUC |", "|---|---:|---:|---:|---:|"]
        for row in reader:
            rows.append(
                f"| {row['config']} | {fmt(float(row['any_auc']))} | {fmt(float(row['any_f1']))} | "
                f"{fmt(float(row['video_auc']))} | {fmt(float(row['audio_auc']))} |"
            )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize evaluation artifacts as Markdown.")
    parser.add_argument("--results-dir", default="results", help="Directory containing test_results.json and ablation.csv.")
    parser.add_argument("--output", help="Optional Markdown output file.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_json(results_dir / "test_results.json")
    text = "\n\n".join(
        [
            "# Evaluation Summary",
            "## Task Metrics",
            render_metrics(results),
            "## Category Breakdown",
            render_categories(results),
            "## Ablation",
            render_ablation(results_dir / "ablation.csv"),
        ]
    )
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
