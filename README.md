# Real-Time Driver Monitoring System v2.0
**Akkiraju Pruthvi | 22BCE2098 | VIT Vellore**

---

## The complete sequence — 7 commands total

```bash
# 1. Setup (once)
bash setup.sh

# 2. See exactly what data you have
bash inspect_datasets.sh

# 3. Prepare data (extracts YawDD, builds manifest)
bash prepare_data.sh

# 4. Train overnight
bash train.sh

# 5. Evaluate next morning
bash evaluate.sh

# 6. Live demo on your face
bash demo.sh

# 7. Full web dashboard
bash dashboard.sh
```

---

## Before step 2: download your datasets

See **docs/dataset_guide.md** for every download link and command.

| Dataset | Required | What it gives | Where to put it |
|---|---|---|---|
| AUC v2 | YES | Distraction + 2 angles | `data/raw/auc_v2` |
| State Farm | YES | Distraction + 1 angle | `data/raw/state_farm` |
| YawDD | YES | Yawn + 2 angles | `data/raw/yawdd` |
| MRL Eye | Recommended | Eye state (84k images) | `data/raw/mrl_eye` |
| CEW | Optional | Eye state (2.4k images) | `data/raw/cew` |
| NTHU-DDD | Optional | Eye state in-car | `data/raw/nthu_ddd` |

**Minimum to start:** AUC + State Farm + YawDD.
**For 90%+ eye accuracy:** Also add MRL Eye (5 min Kaggle download).

---

## Expected accuracy

| Task | With 3 datasets | With MRL Eye added |
|---|---|---|
| Distraction | **90-94%** | 90-94% |
| Yawn | **88-93%** | 88-93% |
| Eye state | 0% (no labels) | **80-88%** |
| Weighted F1 | 0.86-0.91 | **0.88-0.93** |

---

## Scripts reference

| Script | When to run | What it does |
|---|---|---|
| `setup.sh` | Once | Python venv + pip install |
| `inspect_datasets.sh` | Before prepare | Shows ALL files, folders, counts, issues |
| `prepare_data.sh` | Before train | Extract YawDD, build manifest |
| `train.sh` | Overnight | Smoke test + 50-epoch training |
| `evaluate.sh` | After training | Metrics, plots, benchmark table, ONNX |
| `demo.sh` | After training | Live webcam overlay |
| `dashboard.sh` | After training | Web dashboard at localhost:8080 |

---

## What inspect_datasets.sh tells you

Run it before anything else. It shows:
- Exact path found for each dataset
- Number of images / videos at every folder level
- Sample filenames
- Whether Camera 1 and Camera 2 are present (AUC)
- Whether driver_imgs_list.csv is present (State Farm)
- Whether YawDD needs frame extraction
- Whether eye-state folders are correctly named (MRL/CEW)
- A READY / NOT READY verdict with specific instructions

---

## Novel contributions (patent filing)

1. **Measured-angle stratification** — camera angle measured per-frame via MediaPipe solvePnP, splits stratified by (dataset, angle) at subject level
2. **Source-aware loss weighting** — dataset=1.0, pseudo=0.5, assumed=0.0
3. **Gated-EMA P-DSI** — baseline only updates during safe periods (score < gate)

See `docs/technical_disclosure.md` → email VIT IPR Cell to file provisional.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | `brew install ffmpeg` |
| `No module named src` | Run from project root, not from data/ |
| Training very slow (>5s/it) | Laptop overheated — close lid, cool 10 min, reopen |
| `MPS out of memory` | Set `batch_size: 16` in configs/config.yaml |
| Eye accuracy = 0% | No eye dataset — download MRL: `bash inspect_datasets.sh` |
| distraction_acc falling | Old checkpoint — `rm checkpoints/*.pth` then retrain |
