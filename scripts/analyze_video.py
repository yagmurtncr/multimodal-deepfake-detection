r"""Analyze a single video from the command line.

Example:
    python scripts/analyze_video.py --model C:\path\best_model.pt --video sample.mp4 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "demo_site", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepfake_detector.reporting import build_report
from inference import DeepfakePredictor, category_from_path, interpret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal deepfake inference on one video.")
    parser.add_argument("--model", required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--output", help="Optional path to save the JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = DeepfakePredictor(args.model)
    scores, info = predictor.predict(args.video)
    if scores is None:
        print(f"Preprocess failed: {info.get('fail_reason', 'unknown')}", file=sys.stderr)
        return 2

    decision = interpret(scores, threshold=args.threshold)
    category = category_from_path(args.video)
    ground_truth = None
    if category[0]:
        ground_truth = {
            "category": category[0],
            "video_fake": bool(category[3]) and category[1] == "sahte",
            "audio_fake": bool(category[3]) and category[2] == "sahte",
            "any_fake": bool(category[3]),
        }
    report = build_report(scores, decision, info, ground_truth, threshold=args.threshold)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(report["summary"])
        print(f"Scores: video={report['scores']['video']:.4f} audio={report['scores']['audio']:.4f} any={report['scores']['any']:.4f}")
        for finding in report["findings"]:
            print(f"- {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
