#!/usr/bin/env python3
"""Generate unified-vs-baseline comparison matrix from evaluation results.

Usage:
  python3 generate_comparison_matrix.py \
      --eval-results eval_results/unified_YYYYMMDD_HHMMSS/results.json

If --eval-results is omitted, it picks the latest unified run under eval_results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


BASELINES = {
    # Stage 1 best models on same benchmarks
    "classification_trashnet_accuracy": {
        "task": "Classification (in-domain)",
        "baseline_model": "ViT-Base",
        "baseline_metric": 0.9644,
        "metric_name": "Accuracy",
    },
    "classification_realwaste_accuracy": {
        "task": "Classification (cross-domain)",
        "baseline_model": "CLIP",
        "baseline_metric": 0.4268,
        "metric_name": "Accuracy",
    },
    "detection_taco_f1": {
        "task": "Detection (in-domain)",
        "baseline_model": "YOLOv8",
        "baseline_metric": 0.6410,
        "metric_name": "F1",
    },
    "detection_icra19_f1": {
        "task": "Detection (cross-domain)",
        "baseline_model": "Grounding DINO",
        "baseline_metric": 0.3720,
        "metric_name": "F1",
    },
    "segmentation_taco_miou": {
        "task": "Segmentation (in-domain)",
        "baseline_model": "DeepLabV3+",
        "baseline_metric": 0.4541,
        "metric_name": "mIoU",
    },
}


def _find_latest_eval_results(stage2_root: Path) -> Optional[Path]:
    eval_root = stage2_root / "eval_results"
    if not eval_root.exists():
        return None
    candidates = sorted(list(eval_root.glob("unified_*/results.json")) + list(eval_root.glob("zeroshot_*/results.json")))
    return candidates[-1] if candidates else None


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _delta_str(delta: float, is_percent_metric: bool) -> str:
    if is_percent_metric:
        return f"{delta * 100:+.2f} pts"
    return f"{delta:+.4f}"


def build_matrix(results: Dict) -> List[Dict]:
    rows = []

    c_tr = results.get("classification_trashnet", {})
    c_rw = results.get("classification_realwaste", {})
    d_ta = results.get("detection_taco", {})
    d_ic = results.get("detection_icra19", {})
    s_ta = results.get("segmentation_taco", {})

    unified_values = {
        "classification_trashnet_accuracy": c_tr.get("accuracy"),
        "classification_realwaste_accuracy": c_rw.get("accuracy"),
        "detection_taco_f1": d_ta.get("f1"),
        "detection_icra19_f1": d_ic.get("f1"),
        "segmentation_taco_miou": s_ta.get("mean_iou"),
    }

    for key, base in BASELINES.items():
        unified_metric = unified_values.get(key)
        if unified_metric is None:
            continue

        baseline_metric = base["baseline_metric"]
        metric_name = base["metric_name"]
        is_percent_metric = metric_name in {"Accuracy", "F1"}
        delta = unified_metric - baseline_metric

        if is_percent_metric:
            baseline_display = f"{_pct(baseline_metric)} {metric_name}"
            unified_display = f"{_pct(unified_metric)} {metric_name}"
        else:
            baseline_display = f"{baseline_metric:.4f} {metric_name}"
            unified_display = f"{unified_metric:.4f} {metric_name}"

        rows.append(
            {
                "task": base["task"],
                "baseline_model": base["baseline_model"],
                "baseline": baseline_display,
                "unified": unified_display,
                "delta": _delta_str(delta, is_percent_metric),
            }
        )

    return rows


def to_markdown(rows: List[Dict], model_label: str) -> str:
    lines = [
        "# Unified vs Stage 1 Baseline Matrix",
        "",
        f"| Task | Baseline Best | Baseline Metric | {model_label} | Delta |",
        "|------|---------------|----------------|--------------------|-------|",
    ]

    for row in rows:
        lines.append(
            f"| {row['task']} | {row['baseline_model']} | {row['baseline']} | {row['unified']} | {row['delta']} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- Delta = Unified - Baseline.",
            "- Positive delta means unified model is better on that metric.",
            "- Detection comparison uses F1 (not mAP) for apples-to-apples with unified evaluator output.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline comparison matrix from unified eval results")
    parser.add_argument("--eval-results", type=str, default=None, help="Path to eval results.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same folder as results.json)")
    parser.add_argument("--model-label", type=str, default="Florence-2 Unified", help="Column label for evaluated model")
    args = parser.parse_args()

    stage2_root = Path(__file__).resolve().parent

    if args.eval_results:
        eval_path = Path(args.eval_results).expanduser().resolve()
    else:
        latest = _find_latest_eval_results(stage2_root)
        if latest is None:
            raise SystemExit("No evaluation results found under stage_2/eval_results")
        eval_path = latest

    if not eval_path.exists():
        raise SystemExit(f"Evaluation file not found: {eval_path}")

    with eval_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    rows = build_matrix(results)
    if not rows:
        raise SystemExit("No comparable metrics found in evaluation results")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else eval_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_json_path = out_dir / "comparison_matrix.json"
    matrix_md_path = out_dir / "comparison_matrix.md"

    matrix_payload = {
        "source_eval_results": str(eval_path),
        "model_label": args.model_label,
        "rows": rows,
    }
    matrix_json_path.write_text(json.dumps(matrix_payload, indent=2), encoding="utf-8")
    matrix_md_path.write_text(to_markdown(rows, args.model_label), encoding="utf-8")

    print(f"Saved: {matrix_json_path}")
    print(f"Saved: {matrix_md_path}")
    print("\nPreview:")
    for row in rows:
        print(f"- {row['task']}: {row['unified']} vs {row['baseline']} ({row['delta']})")


if __name__ == "__main__":
    main()
