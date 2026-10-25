#!/bin/bash
# Prepare datasets and build manifest. Run: bash prepare_data.sh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }
header(){ echo -e "\n${BLUE}=== $1 ===${NC}"; }

# ---- Find datasets ----
header "Finding datasets"
find_dir(){
    for ROOT in "$PROJECT_DIR/data/raw" "$HOME/Documents/data/raw" \
                "$HOME/Downloads" "$HOME/Desktop" "$HOME/Documents"; do
        [ -d "$ROOT" ] || continue
        for PAT in "$@"; do
            FOUND=$(find "$ROOT" -maxdepth 4 -type d -iname "$PAT" 2>/dev/null | head -1)
            [ -n "$FOUND" ] && { echo "$FOUND"; return; }
        done
    done
    for PAT in "$@"; do
        [ -d "$PROJECT_DIR/data/raw/$PAT" ] && echo "$PROJECT_DIR/data/raw/$PAT" && return
    done
    echo ""
}
AUC=$(find_dir "auc_v2" "auc*v2*" "auc*distracted*")
SF=$(find_dir  "state_farm" "*state*farm*")
YAWDD_VID=$(find_dir "yawdd" "yawdd*")
YAWDD_IMG=$(find_dir "yawdd_frames")
MRL=$(find_dir "mrl_eye" "mrl*eye*" "MRL*")
CEW=$(find_dir "cew" "*closed*eye*" "CEW*")
NTHU=$(find_dir "nthu_ddd" "nthu*")

[ -z "$AUC" ] && fail "AUC v2 not found. See docs/dataset_guide.md"
[ -z "$SF"  ] && fail "State Farm not found. See docs/dataset_guide.md"
[ -z "$YAWDD_VID" ] && [ -z "$YAWDD_IMG" ] && fail "YawDD not found. See docs/dataset_guide.md"

ok "AUC v2:      $AUC"
ok "State Farm:  $SF"
[ -n "$YAWDD_IMG" ] && ok "YawDD frames: $YAWDD_IMG" || ok "YawDD videos: $YAWDD_VID"
[ -n "$MRL"  ] && ok "MRL Eye:      $MRL"  || warn "MRL Eye not found (optional)"
[ -n "$CEW"  ] && ok "CEW:          $CEW"  || warn "CEW not found (optional)"
[ -n "$NTHU" ] && ok "NTHU-DDD:     $NTHU" || info "NTHU-DDD not found (optional)"

# Symlink into data/raw
mkdir -p data/raw
[ ! -e "data/raw/auc_v2"     ] && ln -sfn "$AUC" data/raw/auc_v2
[ ! -e "data/raw/state_farm" ] && ln -sfn "$SF"  data/raw/state_farm
[ -n "$MRL"  ] && [ ! -e "data/raw/mrl_eye"   ] && ln -sfn "$MRL"  data/raw/mrl_eye
[ -n "$CEW"  ] && [ ! -e "data/raw/cew"        ] && ln -sfn "$CEW"  data/raw/cew
[ -n "$NTHU" ] && [ ! -e "data/raw/nthu_ddd"   ] && ln -sfn "$NTHU" data/raw/nthu_ddd

# ---- YawDD frame extraction ----
header "YawDD frame extraction"
if [ -n "$YAWDD_IMG" ] && [ "$(find "$YAWDD_IMG" -name '*.jpg' | wc -l | tr -d ' ')" -gt 100 ]; then
    ok "YawDD frames already extracted ($YAWDD_IMG)"
    [ ! -e "data/raw/yawdd_frames" ] && ln -sfn "$YAWDD_IMG" data/raw/yawdd_frames
    YAWDD_FINAL="data/raw/yawdd_frames"
elif [ -n "$YAWDD_VID" ]; then
    AVI_COUNT=$(find "$YAWDD_VID" \( -name "*.avi" -o -name "*.avi.avi" \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$AVI_COUNT" -gt 0 ]; then
        command -v ffmpeg &>/dev/null || fail "ffmpeg required: brew install ffmpeg"
        info "Extracting $AVI_COUNT videos to frames (~20 min)..."
        python3 scripts/extract_yawdd_frames.py \
            --src "$YAWDD_VID" --dst data/raw/yawdd_frames --fps 5
        ok "YawDD frames extracted"
    fi
    YAWDD_FINAL="data/raw/yawdd_frames"
fi

# ---- Build manifest ----
header "Building manifest"
ROOTS="data/raw/auc_v2 data/raw/state_farm $YAWDD_FINAL"
[ -n "$MRL"  ] && [ -d "data/raw/mrl_eye"   ] && ROOTS="$ROOTS data/raw/mrl_eye"
[ -n "$CEW"  ] && [ -d "data/raw/cew"        ] && ROOTS="$ROOTS data/raw/cew"
[ -n "$NTHU" ] && [ -d "data/raw/nthu_ddd"   ] && ROOTS="$ROOTS data/raw/nthu_ddd"

# Check if existing manifest has contaminated labels
if [ -f "data/processed/manifest.csv" ]; then
    ASSUMED=$(python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/manifest.csv')
print(int((df.get('distraction_source',pd.Series(dtype=str))=='assumed').sum()))
" 2>/dev/null || echo "0")
    if [ "$ASSUMED" -gt 5000 ]; then
        warn "Existing manifest has $ASSUMED assumed distraction labels (old contaminated version)"
        warn "Rebuilding clean manifest..."
        python3 data/download_datasets.py --roots $ROOTS \
            --out data/processed/manifest.csv --no-pseudo-label
    else
        info "Manifest exists and looks clean ($ASSUMED assumed labels)"
        read -p "  Rebuild manifest? (y/N): " CHOICE
        if [ "$CHOICE" = "y" ] || [ "$CHOICE" = "Y" ]; then
            python3 data/download_datasets.py --roots $ROOTS \
                --out data/processed/manifest.csv --no-pseudo-label
        fi
    fi
else
    python3 data/download_datasets.py --roots $ROOTS \
        --out data/processed/manifest.csv --no-pseudo-label
fi
ok "Manifest ready: data/processed/manifest.csv"

echo ""
echo "================================================"
echo " Data ready! Next:"
echo "   bash train.sh"
echo "================================================"
