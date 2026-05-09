#!/usr/bin/env python3
"""
Generate a three-way comparison report:
  Florence-2 Zero-Shot  vs  Florence-2 Fine-Tuned  vs  YOLOv8 Fine-Tuned
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def latest_file(pattern: str) -> Path | None:
    files = sorted(PROJECT_DIR.glob(pattern))
    return files[-1] if files else None


def load_florence2_results(tag: str, split: str = "test") -> dict | None:
    matches = sorted(PROJECT_DIR.glob(f"eval_results/florence2_{tag}_{split}_*/results.json"))
    if not matches:
        return None
    with open(matches[-1]) as f:
        return json.load(f)


def load_yolo_results() -> dict | None:
    p = PROJECT_DIR / "eval_results" / "yolo" / "yolo_eval_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    # prefer test split
    return data.get("test") or data.get("validation")


def fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.4f}" if not pct else f"{v*100:.2f}%"


def main():
    zs  = load_florence2_results("zeroshot")
    ft  = load_florence2_results("finetuned")
    yl  = load_yolo_results()

    if not any([zs, ft, yl]):
        print("No evaluation results found. Run the evaluation scripts first.")
        sys.exit(1)

    def get(d, key):
        return d.get(key) if d else None

    rows = {
        "Florence-2 Zero-Shot":  zs,
        "Florence-2 Fine-Tuned": ft,
        "YOLOv8m Fine-Tuned":    yl,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# CCTV Garbage Detection — Model Comparison",
        "",
        "| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        p  = fmt(get(r, "precision"))
        rc = fmt(get(r, "recall"))
        f1 = fmt(get(r, "f1"))
        m5 = fmt(get(r, "mAP@0.5"))
        m95= fmt(get(r, "mAP@0.5:0.95"))
        lines.append(f"| {name} | {p} | {rc} | {f1} | {m5} | {m95} |")

    lines += [
        "",
        "## TP / FP / FN (Florence-2 models)",
        "",
        "| Model | TP | FP | FN | Images |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in [("Florence-2 Zero-Shot", zs), ("Florence-2 Fine-Tuned", ft)]:
        if r:
            lines.append(f"| {name} | {r.get('tp','N/A')} | {r.get('fp','N/A')} | {r.get('fn','N/A')} | {r.get('evaluated','N/A')} |")

    # Delta row
    if zs and ft:
        dp  = (ft.get("precision", 0) - zs.get("precision", 0))
        drc = (ft.get("recall", 0)    - zs.get("recall", 0))
        df1 = (ft.get("f1", 0)        - zs.get("f1", 0))
        lines += [
            "",
            "## Fine-Tuning Improvement (Florence-2)",
            "",
            f"- Precision: {dp:+.4f}",
            f"- Recall:    {drc:+.4f}",
            f"- F1:        {df1:+.4f}",
        ]

    lines += [
        "",
        "## Conclusion",
        "",
    ]
    if yl and ft:
        if yl.get("f1", 0) > ft.get("f1", 0):
            lines.append(f"YOLOv8m achieves the best F1 ({fmt(get(yl,'f1'))}) vs Florence-2 fine-tuned ({fmt(get(ft,'f1'))}).")
        else:
            lines.append(f"Florence-2 fine-tuned achieves the best F1 ({fmt(get(ft,'f1'))}) vs YOLOv8m ({fmt(get(yl,'f1'))}).")

    report_md = out_dir / f"comparison_{ts}.md"
    report_md.write_text("\n".join(lines))
    print(f"Markdown report: {report_md}")

    # ── JSON report ───────────────────────────────────────────────────────────
    report_json = out_dir / f"comparison_{ts}.json"
    payload = {
        "timestamp": ts,
        "florence2_zeroshot":  zs,
        "florence2_finetuned": ft,
        "yolo_finetuned":      yl,
    }
    report_json.write_text(json.dumps(payload, indent=2))
    print(f"JSON report:     {report_json}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
