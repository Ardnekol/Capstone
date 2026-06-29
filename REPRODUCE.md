# REPRODUCE — Capstone: AI for Waste Management (Unified Foundation Model)

> **Purpose of this file.** This is the single, self-contained guide to rebuild the
> environment, obtain the data, and reproduce every number in the paper after the
> SLURM scratch space is wiped. Hand this file back to Claude after `git clone` and it
> contains enough context to drive the whole pipeline.
>
> **Author:** Lokendra Mandloi (cs24mtech11024) · Guide: Prof. Srijith P. K. · IIT Hyderabad
> **GitHub:** `git@github.com:Ardnekol/Capstone.git`
> **Hardware used:** 1× NVIDIA A100-40GB on the IITH DGX node `dgx-a100-02`.

---

## 0. TL;DR — fastest path to the headline numbers

The fine-tuned LoRA adapter (`stage_2/finetuned/florence2_unified_multitask_lora/`,
~14 MB) **is committed to git**, so you do NOT need to retrain to reproduce the paper
results. You only need: (1) the conda env, (2) the test datasets, (3) run the
cross-domain eval.

```bash
# 1. clone
git clone git@github.com:Ardnekol/Capstone.git ~/Capstone
cd ~/Capstone

# 2. recreate the env (see §2)  -> ~/.conda/envs/Capstone

# 3. download datasets into ./datasets/  (see §3)

# 4. get a GPU and run the full cross-domain evaluation (see §5)
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
tmux new -s eval
cd ~/Capstone/stage_2/cross_domain_eval
python3 check_env.py          # confirm every regime says RUNNABLE
bash run_all.sh               # writes ../eval_results/<task>_<dataset>/*_summary.md
```

Everything else (Stage 1 benchmark training, Stage 2 retraining, Playground demo) is
optional and documented below.

---

## 1. What this project is (so the numbers make sense)

Three components, one storyline: **Stage 1 proves the problem → Stage 2 solves it →
Playground demos it.**

| Path | What it is |
|---|---|
| `stage_1/` | **Benchmark.** 15 task-specific specialists (5 per task) vs zero-shot foundation models, in-domain vs cross-domain. Shows specialists collapse under domain shift (−58% to −80%). |
| `stage_2/` | **Unified model.** One `Florence-2-large-ft` + LoRA (rank 16, α 32, ~5.2M params, ~14 MB adapter) does classification, detection, AND segmentation via task prompts. |
| `stage_2/cross_domain_eval/` | The **3-regime cross-domain evaluation** that produces the paper's headline table. |
| `Playground/` | FastAPI + React demo serving the one unified model for all tasks. |
| `paper_drafts/waste_vision_ieee.tex` | The IEEE paper (compiles standalone on Overleaf). |

**Headline result to reproduce:** the single unified Florence-2-FT is **best, or
statistically tied for best, on all 7 cross-domain benchmarks** (2/2 classification,
3/3 detection, 2/2 segmentation incl. one tie). Target numbers are in §6.

**Trained only on:** TrashNet (classification) + TACO (detection & segmentation).
**Never trained on any cross-domain test set** — that is the whole point.

---

## 2. Environment

All experiments ran in the conda env **`Capstone`** (Python 3.11). The exact stack
that produced the paper numbers:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| python | 3.11.15 | | peft | 0.10.0 |
| torch | 2.0.1+cu117 | | accelerate | 1.13.0 |
| torchvision | 0.15.2+cu117 | | timm | 1.0.22 |
| CUDA (torch) | 11.7 | | ultralytics | 8.3.241 |
| transformers | 4.40.2 | | open_clip_torch | 3.2.0 |
| numpy | 1.26.4 | | segment-anything | 1.0 |
| pillow | 12.0.0 | | segmentation-models-pytorch | 0.5.0 |
| pycocotools | 2.0.11 | | scikit-learn | 1.8.0 |
| opencv-python(-headless) | 4.11.0.86 | | scipy | 1.17.1 |
| einops | 0.4.1 | | datasets | 4.4.2 |
| safetensors | 0.7.0 | | matplotlib | 3.10.8 |

> `transformers==4.40.2` and `einops` are **required** for Florence-2 to load. Newer
> transformers can break Florence-2's custom modeling code, so pin this version.

### Recreate the env

```bash
conda create -n Capstone python=3.11 -y
conda activate Capstone     # or: export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"

# Torch built for CUDA 11.7 (matches the A100 driver used)
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

pip install transformers==4.40.2 peft==0.10.0 accelerate==1.13.0 \
    timm==1.0.22 einops==0.4.1 safetensors==0.7.0 \
    ultralytics==8.3.241 open_clip_torch==3.2.0 \
    segment-anything segmentation-models-pytorch \
    pycocotools==2.0.11 scikit-learn scipy numpy==1.26.4 \
    opencv-python-headless pillow matplotlib tqdm \
    huggingface_hub gdown datasets
```

There are also per-component `requirements.txt` files (`stage_1/requirements.txt`,
`stage_1/{classification,detection,segmentation}/requirements.txt`,
`stage_2/requirements.txt`, `Playground/backend/requirements.txt`) if you prefer to
install per stage.

### flash-attn note
Florence-2 references flash-attn but the repo ships a **pure-Python shim** at
`stage_2/flash_attn/` so you do **not** need to compile real flash-attn. If a script
can't find `flash_attn`, run it from inside `stage_2/` (or add `stage_2/` to
`PYTHONPATH`) so the shim resolves.

### GPU access (IITH cluster)
The login node has **no GPU**. Get an interactive shell on the A100 first:
```bash
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Run long jobs inside `tmux` (`tmux new -s eval`; detach `Ctrl-b d`; reattach
`tmux attach -t eval`).

---

## 3. Datasets

Datasets are **git-ignored** (too large). Download them into `./datasets/`. Helper:
```bash
python stage_1/download_all_datasets.py --task all
# or per task: --task classification | detection | segmentation
```

### Expected on-disk layout

| Dataset | Role | Expected path (hardcoded in eval scripts) |
|---|---|---|
| **TrashNet** | cls train / in-domain | `datasets/classification/trashnet/` |
| **RealWaste** | cls cross-domain (cluttered) | `datasets/classification/realwaste/realwaste-main/RealWaste/` |
| **TACO** | det+seg train / in-domain (urban litter) | `datasets/detection/taco/` |
| **Trash-ICRA19** | det cross-domain (underwater) | `datasets/detection/trash_icra19/` |
| **ZeroWaste-f** | det+seg cross-domain (recycling conveyor) | `datasets/zerowaste-f/splits_final_deblurred/test/` |
| **WaRP** (Warp-C / Warp-D) | cls+det cross-domain (recycling plant) | `datasets/WARP/Warp-C/test_crops/`, `datasets/WARP/Warp-D/test/` |
| **DWSD** (Dense Waste Seg. Dataset) | seg cross-domain | `datasets/segmentation/Dense Waste Segmentation Dataset/` |

> Eval scripts use these relative paths. If you place data elsewhere, symlink into
> `datasets/` or pass the path flag (see `--help` on each `eval_*.py`). After
> downloading, run `check_env.py` then a smoke test (`MAX=20 bash run_all.sh`) to
> confirm paths resolve before the full run.

### Download links

**A. Public URLs verified from the repo's own downloader** (`stage_1/download_all_datasets.py`)
— these are authoritative for this project:

| Dataset | Public source / direct link | DOI / id |
|---|---|---|
| TrashNet | HuggingFace: https://huggingface.co/datasets/garythung/trashnet (repo `garythung/trashnet`) | — |
| RealWaste | UCI: https://archive.ics.uci.edu/dataset/908/realwaste · direct zip: https://archive.ics.uci.edu/static/public/908/realwaste.zip | UCI #908 |
| TACO | Zenodo: https://zenodo.org/records/3587843 · zip: https://zenodo.org/records/3587843/files/TACO.zip · code: https://github.com/pedropro/TACO | 10.5281/zenodo.3587843 |
| Trash-ICRA19 | UMN Conservancy: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649 · direct: https://conservancy.umn.edu/bitstreams/0239b06a-512e-49c3-80aa-ba33371e11de/download | 10.13020/x0qn-y082 |
| BePLi v1 ᵃ | SEANOE: https://www.seanoe.org/data/00811/92297/ · zip: https://www.seanoe.org/data/00811/92297/data/98753.zip · code: https://github.com/earthlab-be/BePLi | 10.17882/92297 |

ᵃ BePLi v1 was the *original* segmentation cross-domain plan in the downloader; the
final paper uses **DWSD** + **ZeroWaste-f** for segmentation instead. Kept here for
completeness.

**B. Cross-domain sets used in Stage 2 — public, but their URLs are NOT recorded in
the repo.** These are the canonical public sources (⚠️ **verify the exact mirror/split
matches what's on disk** before trusting numbers — these are the well-known releases,
not necessarily byte-identical to the local copy):

| Dataset | Canonical public source |
|---|---|
| ZeroWaste-f | Project page: http://ai.bu.edu/zerowaste/ · GitHub: https://github.com/dbash/zerowaste (use the `splits_final_deblurred` release; ~the conveyor-belt recycling dataset) |
| WaRP (Warp-C / Warp-D / Warp-S) | Kaggle: https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset (Waste Recycling Plant dataset) |
| DWSD (Dense Waste Segmentation Dataset) | ⚠️ **Source not recorded in the repo and not confirmed by me — do not assume a URL.** Locate the exact copy you downloaded (check your browser history / Kaggle / Roboflow / the dataset's own paper) and record it here before re-running segmentation. |

**C. Extra datasets present in `datasets/` but not part of the paper's headline tables**
(YOLO / surveillance side-experiments) — also public, from Roboflow Universe:

| Dataset | Source |
|---|---|
| garbage_best (YOLOv8) | https://universe.roboflow.com/smart-india-hackathon-2023/garbage_best |
| Garbage Detection using CCTV | Roboflow Universe (search "Garbage Detection using CCTV"); large (~3 GB) and git-ignored |
| AllGVPImages / Trashcan | Local/auxiliary sets — **source not recorded; treat as private/manual** unless you can confirm a public origin |

**Only the test splits** of the cross-domain datasets are needed to reproduce the
headline results — none were used in training (training = TrashNet + TACO only).

> **If any dataset above is private or you got it via a manual/restricted mirror,**
> leave its row as-is and just note "private — obtained manually" next to it; the
> reproduction still works as long as the files land at the expected path in the table.

---

## 4. Reproduce Stage 1 (benchmark) — *optional, heavy*

Stage 1 trains the 15 specialists. This is the most compute-heavy part. Skip it if you
only need the unified-model headline numbers — the specialist scores are already
recorded in `stage_1/MINI_REPORT.md` and `stage_2/cross_domain_eval/PROJECT_MASTER_RESULTS.md`.

```bash
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"

# Classification (ViT-Base, ResNet-50, EfficientNet-B0, CLIP)
cd ~/Capstone/stage_1/classification && bash quick_start.sh        # --quick for 5 epochs

# Detection (YOLOv8m, Faster R-CNN, RetinaNet, Grounding DINO, Florence-2)
cd ~/Capstone/stage_1/detection && bash run_detection_full.sh

# Segmentation (U-Net, DeepLabV3+, Mask R-CNN, SAM)
cd ~/Capstone/stage_1/segmentation && bash run_all_models.sh
```
Specialist weights (`*_best.pth`, YOLO `runs/`, `sam_vit_h_4b8939.pth`) are git-ignored
and regenerated by these scripts. Reports land next to each task; the consolidated view
is `stage_1/MINI_REPORT.md`.

---

## 5. Reproduce Stage 2 — the unified model

### 5a. Cross-domain evaluation (reproduces the paper's headline table)

**You do NOT need to retrain.** The adapter is committed. Just evaluate:

```bash
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
tmux new -s eval
cd ~/Capstone/stage_2/cross_domain_eval

python3 check_env.py     # every regime should say RUNNABLE under the Capstone env
bash run_all.sh          # picks the freest GPU; logs -> ../eval_results/logs_<stamp>/
```

`run_all.sh` runs four steps and prints all summary tables at the end:
1. `01_classification_warpc` — WaRP-C: ViT/ResNet/EffNet + CLIP + Florence-FT
2. `02_detection_zerowaste` — ZeroWaste-f detection: YOLOv8 + Grounding-DINO + Florence-FT
3. `03_detection_warpd` — WaRP-D detection
4. `04_segmentation_zerowaste` — ZeroWaste-f seg: DeepLabV3+ + SAM + Florence-FT

Each step writes `*_results.json` + `*_summary.md` under
`stage_2/eval_results/<task>_<dataset>/`.

**Useful env knobs:**
```bash
MAX=20 bash run_all.sh                # smoke test on 20 images per regime
PYTHON=python3.11 bash run_all.sh     # override interpreter
DEVICE=cuda:0 bash run_all.sh
SKIP_CLS=clip SKIP_DET=grounding_dino SKIP_SEG=sam bash run_all.sh   # skip a regime
LORA=../finetuned/florence2_unified_multitask_lora bash run_all.sh   # which adapter
```

**Run a single regime directly** (defaults to the committed adapter):
```bash
python eval_warpc_classification.py --lora ../finetuned/florence2_unified_multitask_lora --device cuda:0
python eval_detection.py --dataset zerowaste --lora ../finetuned/florence2_unified_multitask_lora
python eval_detection.py --dataset warpd    --lora ../finetuned/florence2_unified_multitask_lora
python eval_segmentation.py --dataset zerowaste --seg-method cascade --lora ../finetuned/florence2_unified_multitask_lora
```

RealWaste classification and ICRA19/DWSD rows come from
`eval_realwaste_classification.py` and the earlier four-regime eval — see
`stage_2/STAGE2_REPORT.md`.

**Statistical significance** (the ZeroWaste-f segmentation tie):
```bash
python diag_seg_significance.py --lora ../finetuned/florence2_unified_multitask_lora
# paired Wilcoxon Florence vs Mask R-CNN -> p≈0.98, bootstrap 95% CI of diff includes 0, n=929
```

### 5b. Retrain the unified model from scratch — *optional*

Only needed if you want to rebuild the adapter rather than use the committed one.
~3 hours on one A100-40GB for the v1 config.

```bash
cd ~/Capstone/stage_2
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"

bash train_unified.sh          # full: prep JSONL -> train -> evaluate
# bash train_unified.sh --quick        # 50-image smoke test
# bash train_unified.sh --skip-prep    # reuse existing finetune_data/*.jsonl
# bash train_unified.sh --eval-only     # just evaluate
```

**v1 hyperparameters (the paper model):** `microsoft/Florence-2-large-ft`,
epochs 3, lr 1e-4, batch 1 × grad-accum 8, LoRA rank 16 / α 32, balance none.
Output → `stage_2/finetuned/florence2_unified_multitask_lora/`.

The training JSONLs are committed (`stage_2/finetune_data/unified_multitask_{train,val}.jsonl`),
so `--skip-prep` works without re-downloading TACO/TrashNet. To rebuild them:
```bash
python prepare_taco_florence2_od_jsonl.py
python prepare_taco_florence2_seg_jsonl.py
python prepare_trashnet_florence2_caption_jsonl.py
python prepare_unified_multitask_jsonl.py --balance none
```

**v2 (stronger) config** lives in `train_unified_v2.sh` (epochs 10, rank 64 / α 128,
oversample balance, bf16) → writes to `..._lora_v2/` and leaves v1 untouched. The paper
headline uses **v1**.

> Retraining will not reproduce numbers bit-for-bit (CUDA nondeterminism, no fixed
> global seed across all regimes). Use the committed adapter for exact paper numbers;
> multi-seed runs are aggregated via `aggregate_seeds.py` / `run_multiseed.sh`.

---

## 6. Target numbers (what a correct reproduction should print)

From `stage_2/cross_domain_eval/PROJECT_MASTER_RESULTS.md`. Bold = best in column.

**Classification — Accuracy % (macro-F1)**

| Model | TrashNet *(in)* | RealWaste *(cross)* | WaRP-C *(cross)* |
|---|--:|--:|--:|
| ViT-Base (TS) | **96.44** | 39.98 | 21.99 (0.215) |
| EfficientNet-B0 (TS) | 89.53 | 32.89 | 46.29 (0.346) |
| ResNet-50 (TS) | 80.83 | 17.85 | 17.28 (0.164) |
| CLIP ViT-B/16 (FM) | 67.83 | 42.68 | 43.07 (0.395) |
| **Florence-2 FT (UNI)** | 85.24 | **56.68** | **60.35 (0.415)** |

**Detection — F1 @ IoU≥0.5 (class-agnostic)**

| Model | TACO *(in)* | ICRA19 | ZeroWaste-f | WaRP-D |
|---|--:|--:|--:|--:|
| YOLOv8m (TS) | **0.641** | 0.139 | 0.220 | 0.184 |
| Faster R-CNN (TS) | 0.195 | 0.137 | 0.146 | 0.123 |
| Grounding DINO (FM) | 0.253 | 0.372 | 0.208 | 0.192 |
| **Florence-2 FT (UNI)** | 0.366 | **0.505** | **0.272** | **0.281** |

**Segmentation — binary mIoU**

| Model | TACO *(in)* | DWSD | ZeroWaste-f |
|---|--:|--:|--:|
| DeepLabV3+ (TS) | **0.454** | 0.048 | 0.132 |
| U-Net (TS) | 0.329 | 0.063 | 0.131 |
| Mask R-CNN (TS) | 0.289 | 0.084 | 0.169 |
| SAM ViT-H (FM) | 0.038 | 0.102 | 0.027 |
| **Florence-2 FT (UNI)** | 0.222 | **0.180** | 0.160 ᵗ |

ᵗ ZeroWaste-f: Florence 0.160 vs Mask R-CNN 0.169 is a **statistical tie**
(paired Wilcoxon p≈0.98; bootstrap 95% CI of the difference [−0.004, 0.022] includes 0,
n=929).

→ **Florence-2-FT is best or tied-for-best on all 7 cross-domain benchmarks.**

**Caveats that affect the numbers:**
- WaRP-C is plastic-heavy (plastic 1291 / cardboard 162 / metal 98) — cite **macro-F1**,
  not raw accuracy. Class mapping is `WARPC_TO_TRASHNET` in
  `eval_warpc_classification.py` (bottle/detergent/canister→plastic, cans→metal,
  cardboard→cardboard).
- Florence detection uses the class-agnostic `<REGION_PROPOSAL>` head.
- Florence segmentation uses the multi-instance cascade (phrase-grounding →
  per-region segmentation → union), `--seg-method cascade`.

---

## 7. Playground demo — *optional*

```bash
cd ~/Capstone/Playground
pip install -r backend/requirements.txt
./start.sh                      # builds frontend + serves UI+API at http://localhost:7860
# dev mode: python backend/main.py  (API :8000) + (cd frontend && npm install && npm run dev) (UI :5173)
```
Overrides: `PORT=9000`, `MODEL_CACHE_DIR=/fast/disk`, `TORCH_DTYPE=bfloat16`,
`DEFAULT_MODEL=microsoft/Florence-2-base`, `HF_HUB_OFFLINE=1`. Full notes:
`Playground/how to run.txt`, `Playground/PLAN.md`.

---

## 8. Reference docs already in the repo

- `README.md` — project overview.
- `stage_1/MINI_REPORT.md`, `stage_1/EXPERIMENT_DESIGN.md` — Stage 1 design + results.
- `stage_2/STAGE2_REPORT.md`, `stage_2/PLAN.md` — Stage 2 design + results.
- `stage_2/cross_domain_eval/{README.md, PROJECT_MASTER_RESULTS.md, RESULTS.md}` — the eval.
- `full_paper.md`, `paper_drafts/waste_vision_ieee.tex` — the paper.

---

## 9. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` | You're on the login node. `srun ... --pty bash` onto the A100 first. |
| `ImportError: flash_attn` | Run from inside `stage_2/` so the Python shim resolves, or add it to `PYTHONPATH`. |
| Florence-2 fails to load / modeling error | Pin `transformers==4.40.2` and ensure `einops` is installed. |
| A regime says `SKIP` in `check_env.py` | Install its dep (e.g. `pip install open_clip_torch ultralytics segmentation_models_pytorch segment_anything`). |
| Dataset `FileNotFoundError` | Paths in §3 are hardcoded-relative; symlink into `datasets/` or pass the path flag (`--help`). |
| Grounding DINO hangs on first run | It downloads weights from HF — needs network/cache; pre-warm or set `HF_HUB_OFFLINE` after first fetch. |
| Shared GPU OOM | `run_all.sh`/`train_*.sh` auto-pick the freest GPU; or pin `CUDA_VISIBLE_DEVICES=<id>`. |
