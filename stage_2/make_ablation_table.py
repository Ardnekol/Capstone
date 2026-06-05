#!/usr/bin/env python3
"""Build the E1 multi-task ablation comparison table.

Loads per-LoRA results from:

  Main 5-benchmark eval:  eval_results/e1_ablation/<lora_name>/results.json
  DWSD cross-domain seg:  eval_results/dwsd_<lora_name>/results.json
                          (unified model used eval_results/dwsd_unified_v3/)

and emits a comparison file in two formats:

  eval_results/e1_ablation/comparison.md
  eval_results/e1_ablation/comparison.json

Columns: cls-only | det-only | unified
Rows:    benchmark × metric pairs from both source files.
"""

from __future__ import annotations

import json
from pathlib import Path

STAGE2 = Path(__file__).resolve().parent
EVAL_DIR = STAGE2 / "eval_results"
ABLATION_DIR = EVAL_DIR / "e1_ablation"

# Display order for columns (must match dir names under eval_results/e1_ablation/)
# Note: seg-only LoRA was excluded after mode-collapse during training (see paper
# discussion). The seg-only run is archived as florence2_seg_only_lora_collapsed.
MODEL_ORDER = [
    ("florence2_cls_only_lora", "cls-only"),
    ("florence2_det_only_lora", "det-only"),
    ("florence2_unified_multitask_lora", "unified"),
]

# Per-model paths for the supplementary DWSD cross-domain segmentation eval.
# The unified run was named dwsd_unified_v3 (after debug iterations); the
# single-task runs follow the dwsd_<lora_name> pattern.
DWSD_DIRS = {
    "florence2_cls_only_lora": EVAL_DIR / "dwsd_florence2_cls_only_lora",
    "florence2_det_only_lora": EVAL_DIR / "dwsd_florence2_det_only_lora",
    "florence2_unified_multitask_lora": EVAL_DIR / "dwsd_unified_v3",
}

# Each row: (display name, source, results.json top key, metric key, format-fn)
# source: "main" => eval_results/e1_ablation/<lora>/results.json
#         "dwsd" => DWSD_DIRS[<lora>]/results.json
ROWS = [
    ("TrashNet cls (in-domain) — accuracy",        "main", "classification_trashnet",  "accuracy",       "{:.4f}"),
    ("TrashNet cls (in-domain) — macro F1",        "main", "classification_trashnet",  "macro_f1",       "{:.4f}"),
    ("RealWaste cls (cross-domain) — accuracy",    "main", "classification_realwaste", "accuracy",       "{:.4f}"),
    ("RealWaste cls (cross-domain) — macro F1",    "main", "classification_realwaste", "macro_f1",       "{:.4f}"),
    ("TACO det (in-domain) — F1",                  "main", "detection_taco",           "f1",             "{:.4f}"),
    ("TACO det (in-domain) — precision",           "main", "detection_taco",           "precision",      "{:.4f}"),
    ("TACO det (in-domain) — recall",              "main", "detection_taco",           "recall",         "{:.4f}"),
    ("ICRA19 det (cross-domain) — F1",             "main", "detection_icra19",         "f1",             "{:.4f}"),
    ("TACO seg (in-domain) — mIoU",                "main", "segmentation_taco",        "mean_iou",       "{:.4f}"),
    ("TACO seg (in-domain) — pixel acc",           "main", "segmentation_taco",        "pixel_accuracy", "{:.4f}"),
    ("DWSD seg (cross-domain) — mIoU",             "dwsd", "segmentation_dwsd",        "mean_iou",       "{:.4f}"),
    ("DWSD seg (cross-domain) — pixel acc",        "dwsd", "segmentation_dwsd",        "pixel_accuracy", "{:.4f}"),
]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_results_for_model(model_dir: str) -> dict[str, dict | None]:
    return {
        "main": _load_json(ABLATION_DIR / model_dir / "results.json"),
        "dwsd": _load_json(DWSD_DIRS[model_dir] / "results.json") if model_dir in DWSD_DIRS else None,
    }


def get_metric(results: dict | None, top_key: str, metric_key: str) -> float | None:
    if results is None:
        return None
    section = results.get(top_key)
    if not isinstance(section, dict):
        return None
    val = section.get(metric_key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def main() -> None:
    if not ABLATION_DIR.exists():
        raise SystemExit(f"No ablation eval results found at {ABLATION_DIR}. Run eval_single_task_ablation.sh first.")

    # Load every model's main + DWSD results once
    all_results: dict[str, dict[str, dict | None]] = {}
    for model_dir, _label in MODEL_ORDER:
        all_results[model_dir] = load_results_for_model(model_dir)

    # Build markdown table
    md_lines: list[str] = []
    md_lines.append("# E1 — Multi-Task Ablation Comparison\n")
    md_lines.append("Each column is a Florence-2-large-ft LoRA trained on a single task (or all)."
                    " Same hyperparams across all three runs.\n")
    md_lines.append("")

    header = "| Benchmark / Metric | " + " | ".join(label for _d, label in MODEL_ORDER) + " | Best |"
    sep = "|" + "---|" * (len(MODEL_ORDER) + 2)
    md_lines.append(header)
    md_lines.append(sep)

    json_rows: list[dict] = []

    for row_name, source, top_key, metric_key, fmt in ROWS:
        cells = []
        values: list[tuple[str, float]] = []
        for model_dir, label in MODEL_ORDER:
            results = all_results[model_dir].get(source)
            v = get_metric(results, top_key, metric_key)
            if v is None:
                cells.append("—")
            else:
                cells.append(fmt.format(v))
                values.append((label, v))

        best_label = max(values, key=lambda x: x[1])[0] if values else "—"
        md_lines.append("| " + row_name + " | " + " | ".join(cells) + f" | **{best_label}** |")
        json_rows.append({
            "row": row_name,
            "source": source,
            "metric_key": metric_key,
            "section": top_key,
            "values": {label: get_metric(all_results[d].get(source), top_key, metric_key) for d, label in MODEL_ORDER},
            "best": best_label,
        })

    md_lines.append("")
    md_lines.append("## Notes\n")
    md_lines.append("- Best column per row is bolded.")
    md_lines.append("- Dashes mean that LoRA's eval didn't produce that metric (missing dir, eval skipped, or DWSD seg not run for that LoRA).")
    md_lines.append("- DWSD seg rows use eval_results/dwsd_<lora>/results.json (unified uses dwsd_unified_v3).")
    md_lines.append("- Cross-domain rows (RealWaste cls, ICRA19 det, DWSD seg) are what reviewers care about for the lab-to-field gap claim.")

    md_out = ABLATION_DIR / "comparison.md"
    md_out.write_text("\n".join(md_lines) + "\n")

    json_out = ABLATION_DIR / "comparison.json"
    json_out.write_text(json.dumps({
        "models": [{"dir": d, "label": label} for d, label in MODEL_ORDER],
        "rows": json_rows,
    }, indent=2))

    print(f"Wrote {md_out}")
    print(f"Wrote {json_out}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
