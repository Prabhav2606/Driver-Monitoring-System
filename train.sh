#!/bin/bash
# Smoke test + training. Run: bash train.sh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }
header(){ echo -e "\n${BLUE}=== $1 ===${NC}"; }

[ -f "data/processed/manifest.csv" ] || fail "manifest.csv missing. Run: bash prepare_data.sh"

# ---- Remove old contaminated checkpoints ----
header "Checking for old checkpoints"
OLD_CKPTS=$(ls checkpoints/best_epoch00[0-1]_*.pth 2>/dev/null | head -3 || true)
if [ -n "$OLD_CKPTS" ]; then
    BEST_F1=$(ls checkpoints/best_epoch*.pth 2>/dev/null | \
        python3 -c "
import sys,re
lines=[l.strip() for l in sys.stdin if l.strip()]
if not lines: print(0)
else:
    def s(p): m=re.search(r'(\d+\.\d+)\.pth',p); return float(m.group(1)) if m else 0
    print(max(s(l) for l in lines))
" 2>/dev/null || echo "0")
    if python3 -c "exit(0 if float('$BEST_F1') < 0.50 else 1)" 2>/dev/null; then
        warn "Removing low-quality checkpoints (F1=$BEST_F1, contaminated labels)"
        rm -f checkpoints/*.pth
        ok "Old checkpoints removed"
    else
        ok "Existing checkpoints look good (F1=$BEST_F1) — will resume from best"
    fi
fi

# ---- Smoke test ----
header "Smoke test (2 epochs on 200 samples each — ~5 min)"
info "This proves every part of the training loop works before the long run."

# Write smoke config
python3 -c "
import yaml; from pathlib import Path
p=Path('configs/config.yaml'); cfg=yaml.safe_load(p.read_text())
orig=(cfg['training']['epochs'],cfg['training']['batch_size'],
      cfg['training']['freeze_backbone_epochs'],cfg['training']['early_stopping_patience'])
cfg.update({'training': {**cfg['training'],'epochs':2,'batch_size':16,
            'freeze_backbone_epochs':2,'early_stopping_patience':99}})
p.write_text(yaml.dump(cfg,default_flow_style=False))
open('.smoke_orig','w').write(' '.join(map(str,orig)))
print('smoke config written')
"

# Run smoke
python3 train.py --manifest data/processed/manifest.csv --epochs 2
SMOKE_EXIT=$?

# Restore config
python3 -c "
import yaml; from pathlib import Path, os
p=Path('configs/config.yaml'); cfg=yaml.safe_load(p.read_text())
try:
    e,b,fr,es=open('.smoke_orig').read().strip().split()
    cfg['training'].update({'epochs':int(e),'batch_size':int(b),
        'freeze_backbone_epochs':int(fr),'early_stopping_patience':int(es)})
    import os as _os; _os.remove('.smoke_orig')
except: pass
p.write_text(yaml.dump(cfg,default_flow_style=False))
print('config restored')
"

[ $SMOKE_EXIT -eq 0 ] || fail "Smoke test failed — do not proceed"
ls checkpoints/*.pth &>/dev/null || fail "No checkpoint from smoke test — check errors above"
ok "Smoke test passed"

# ---- Confirm full run ----
header "Starting full training"
EPOCHS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/config.yaml'))['training']['epochs'])")
SAMPLES=$(python3 -c "import pandas as pd; df=pd.read_csv('data/processed/manifest.csv'); print(len(df[df['split']=='train']))")
ETA_LOW=$((EPOCHS * 35 / 60))
ETA_HIGH=$((EPOCHS * 55 / 60))
DATASETS=$(python3 -c "import pandas as pd; df=pd.read_csv('data/processed/manifest.csv'); print(', '.join(sorted(df['dataset'].unique())))")

echo ""
echo "  Epochs          : $EPOCHS"
echo "  Training samples: $SAMPLES"
echo "  Datasets        : $DATASETS"
echo "  M1 estimate     : ${ETA_LOW}–${ETA_HIGH} hours"
echo "  caffeinate      : prevents sleep with lid closed"
echo "  Sound+notif     : plays when done"
echo ""
read -p "Start overnight training? (y/N): " CONFIRM
[ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ] || { echo "Cancelled."; exit 0; }

mkdir -p logs
echo "Training started at $(date)" | tee logs/overnight.log

caffeinate -i python3 train.py \
    --manifest data/processed/manifest.csv \
    --epochs "$EPOCHS" \
    2>&1 | tee -a logs/overnight.log
TRAIN_EXIT=${PIPESTATUS[0]}

MSG="Training $([ $TRAIN_EXIT -eq 0 ] && echo 'COMPLETE' || echo 'FAILED') at $(date)"
say "$MSG" 2>/dev/null || true
osascript -e "display notification \"$MSG\" with title \"DMS\" sound name \"Glass\"" 2>/dev/null || true
echo "$MSG"

[ $TRAIN_EXIT -eq 0 ] && echo -e "\nNext: bash evaluate.sh" || fail "Training failed — check logs/overnight.log"
