# Cross-Domain 3-Regime Evaluation (Stage 2)

Compares **Task-Specific vs Foundation vs Florence-2-FT** on two NEW cross-domain
datasets, using ONE consistent metric per task (no mAP-vs-F1 mismatch).

| Task | Dataset | Task-Specific | Foundation | Unified FT | Metric |
|---|---|---|---|---|---|
| Classification | WaRP-C (1551 test) | ViT/ResNet/EffNet | CLIP | Florence-2+LoRA | Accuracy / Macro-F1 |
| Detection | ZeroWaste-f (929), WaRP-D (522) | YOLOv8m | Grounding DINO | Florence-2+LoRA | P/R/F1 @IoU0.5 (class-agnostic) |
| Segmentation | ZeroWaste-f (929) | DeepLabV3+ | SAM ViT-H | Florence-2+LoRA | binary mIoU / pixel-acc |

## How to run (overnight, on the GPU node)

```bash
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
tmux new -s eval
cd ~/Capstone/stage_2/cross_domain_eval

python3 check_env.py          # 1. confirm which regimes are runnable HERE
bash run_all.sh               # 2. run everything; logs → ../eval_results/logs_<stamp>/
# Ctrl-b d to detach; `tmux attach -t eval` in the morning
```

Knobs (env vars): `PYTHON=python3.11`, `DEVICE=cuda:0`, `MAX=20` (smoke test),
`SKIP_CLS=clip`, `SKIP_DET=grounding_dino`, `SKIP_SEG=sam`.

## IMPORTANT — environment
No single login-node interpreter has every dependency. `check_env.py` tells you
what the current python can run. Missing regimes are **skipped cleanly** (the run
never aborts). If a regime you want says SKIP, install it:
```bash
pip install --user open_clip_torch ultralytics segmentation_models_pytorch segment_anything
```

## Outputs
Each step writes `*_results.json` + `*_summary.md` under `../eval_results/<task>_<dataset>/`.
`run_all.sh` prints all summary tables at the end.

## Caveats to remember
- **WaRP-C mapping**: bottle/detergent/canister→plastic, cans→metal, cardboard→cardboard
  (glass bottles are folded into "bottle"→plastic per WaRP's own taxonomy; classes
  are imbalanced: plastic 1291 / cardboard 162 / metal 98). Edit `WARPC_TO_TRASHNET`
  in `eval_warpc_classification.py` to change.
- Test splits only — these datasets were never trained on.
- Grounding DINO downloads weights from HF on first run (needs network/cache).
