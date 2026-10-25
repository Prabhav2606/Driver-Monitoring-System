#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
# ============================================================
# DATASET INSPECTOR
# Run this BEFORE prepare_data.sh to see exactly what you have.
# Shows every folder, file count, sample names, and issues.
# bash inspect_datasets.sh
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
header(){ echo -e "\n${BLUE}╔══════════════════════════════════════════════════════╗${NC}"; \
          echo -e "${BLUE}║  $1${NC}"; \
          echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

# ============================================================
# Find all datasets
# ============================================================
header "SCANNING FOR DATASETS"

# Search locations
SEARCH_PATHS=(
    "$PROJECT_DIR/data/raw"
    "$HOME/Documents/data/raw"
    "$HOME/Downloads"
    "$HOME/Desktop"
    "$HOME/Documents"
)

echo ""
echo "Searching in:"
for p in "${SEARCH_PATHS[@]}"; do
    [ -d "$p" ] && echo "  $p"
done
echo ""

# Find each dataset
find_dir() {
    local PATTERNS=("$@")
    for ROOT in "${SEARCH_PATHS[@]}"; do
        [ -d "$ROOT" ] || continue
        for PAT in "${PATTERNS[@]}"; do
            FOUND=$(find "$ROOT" -maxdepth 4 -type d -iname "$PAT" 2>/dev/null | head -1)
            if [ -n "$FOUND" ]; then echo "$FOUND"; return; fi
        done
    done
    # also check if symlink in data/raw
    for PAT in "${PATTERNS[@]}"; do
        [ -d "$PROJECT_DIR/data/raw/$PAT" ] && echo "$PROJECT_DIR/data/raw/$PAT" && return
    done
    echo ""
}

AUC_PATH=$(find_dir "auc_v2" "auc*v2*" "auc*distracted*" "AUC*")
SF_PATH=$(find_dir  "state_farm" "*state*farm*" "StateFarm")
YAWDD_PATH=$(find_dir "yawdd" "yawdd*" "YawDD")
YAWDD_FRAMES=$(find_dir "yawdd_frames" "yawdd_frames*")
MRL_PATH=$(find_dir "mrl_eye" "mrl*eye*" "MRL*" "mrl*")
CEW_PATH=$(find_dir "cew" "*closed*eye*wild*" "*CEW*")
NTHU_PATH=$(find_dir "nthu_ddd" "nthu*" "NTHU*")

# Use yawdd_frames if yawdd extraction already done
[ -z "$YAWDD_FRAMES" ] && YAWDD_FRAMES=""
[ -n "$YAWDD_FRAMES" ] && YAWDD_EFFECTIVE="$YAWDD_FRAMES (extracted frames)" || \
    { [ -n "$YAWDD_PATH" ] && YAWDD_EFFECTIVE="$YAWDD_PATH (videos — needs extraction)" || YAWDD_EFFECTIVE="NOT FOUND"; }

echo "Dataset locations:"
echo "  AUC v2      : ${AUC_PATH:-NOT FOUND}"
echo "  State Farm  : ${SF_PATH:-NOT FOUND}"
echo "  YawDD       : $YAWDD_EFFECTIVE"
echo "  MRL Eye     : ${MRL_PATH:-NOT FOUND (optional)}"
echo "  CEW         : ${CEW_PATH:-NOT FOUND (optional)}"
echo "  NTHU-DDD    : ${NTHU_PATH:-NOT FOUND (optional)}"

# ============================================================
# Inspect each found dataset
# ============================================================

inspect_dir() {
    local NAME="$1"
    local PATH="$2"
    local REQUIRED="$3"

    header "INSPECTING: $NAME"

    if [ -z "$PATH" ] || [ ! -d "$PATH" ]; then
        if [ "$REQUIRED" = "yes" ]; then
            fail "NOT FOUND — required dataset! See docs/dataset_guide.md"
        else
            warn "Not found — optional (improves accuracy)"
        fi
        return
    fi

    info "Path: $PATH"
    echo ""

    # File counts
    JPG=$(find "$PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
    PNG=$(find "$PATH" -type f -iname "*.png" 2>/dev/null | wc -l | tr -d ' ')
    AVI=$(find "$PATH" -type f \( -iname "*.avi" -o -iname "*.avi.avi" \) 2>/dev/null | wc -l | tr -d ' ')
    CSV=$(find "$PATH" -type f -iname "*.csv" 2>/dev/null | tr '\n' ' ')
    TOTAL_IMG=$((JPG + PNG))

    echo "  File counts:"
    echo "    JPG/JPEG : $JPG"
    echo "    PNG      : $PNG"
    echo "    AVI      : $AVI"
    [ -n "$CSV" ] && echo "    CSV      : $CSV"
    echo "    TOTAL IMG: $TOTAL_IMG"

    echo ""
    echo "  Folder structure (3 levels deep):"
    find "$PATH" -maxdepth 3 -type d 2>/dev/null | sort | while read DIR; do
        DEPTH=$(echo "$DIR" | sed "s|$PATH||g" | tr -cd '/' | wc -c | tr -d ' ')
        INDENT=$(printf '%*s' "$((DEPTH*2))" '')
        BASENAME=$(basename "$DIR")
        # Count files directly in this folder
        NFILES=$(find "$DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.avi" -o -iname "*.avi.avi" \) 2>/dev/null | wc -l | tr -d ' ')
        if [ "$NFILES" -gt 0 ]; then
            echo "    ${INDENT}${BASENAME}/  ($NFILES files)"
        else
            echo "    ${INDENT}${BASENAME}/"
        fi
    done

    echo ""
    echo "  Sample filenames (first 5):"
    find "$PATH" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.avi" -o -iname "*.avi.avi" \) 2>/dev/null | \
        head -5 | while read F; do echo "    $(basename "$F")"; done

    echo ""

    # Dataset-specific checks
    case "$NAME" in
        "AUC v2")
            CAM1=$(find "$PATH" -type d -iname "Camera 1" -o -type d -iname "Camera1" 2>/dev/null | wc -l | tr -d ' ')
            CAM2=$(find "$PATH" -type d -iname "Camera 2" -o -type d -iname "Camera2" 2>/dev/null | wc -l | tr -d ' ')
            V1=$(find "$PATH" -type d -iname "v1_cam1_no_split" 2>/dev/null | wc -l | tr -d ' ')
            [ "$CAM1" -gt 0 ] && ok "Camera 1 found" || warn "Camera 1 not found"
            [ "$CAM2" -gt 0 ] && ok "Camera 2 found (second real angle!)" || warn "Camera 2 not found"
            [ "$V1" -gt 0 ] && ok "v1 human-named folders found (extra data)"
            [ "$TOTAL_IMG" -gt 10000 ] && ok "$TOTAL_IMG images found" || warn "Only $TOTAL_IMG images — expected ~32,000"
            ;;
        "State Farm")
            [ -f "$PATH/driver_imgs_list.csv" ] && ok "driver_imgs_list.csv found (subject IDs available)" \
                || warn "driver_imgs_list.csv missing — subject splits will be approximate"
            TRAIN=$(find "$PATH/imgs/train" -type d -name "c0" 2>/dev/null | wc -l | tr -d ' ')
            [ "$TRAIN" -gt 0 ] && ok "imgs/train/c0..c9 structure found" \
                || warn "imgs/train structure not found — check extraction"
            [ "$TOTAL_IMG" -gt 20000 ] && ok "$TOTAL_IMG images found" \
                || warn "Only $TOTAL_IMG images — expected ~102,000"
            ;;
        "YawDD")
            DASH=$(find "$PATH" -type d -iname "Dash" 2>/dev/null | wc -l | tr -d ' ')
            MIRROR=$(find "$PATH" -type d -iname "Mirror" 2>/dev/null | wc -l | tr -d ' ')
            [ "$DASH" -gt 0 ] && ok "Dash (dashboard angle) found" || warn "Dash folder not found"
            [ "$MIRROR" -gt 0 ] && ok "Mirror (mirror angle) found" || warn "Mirror folder not found"
            if [ "$AVI" -gt 0 ] && [ "$TOTAL_IMG" -eq 0 ]; then
                warn "$AVI video files found but NO images"
                warn "You need to extract frames first!"
                echo ""
                echo "    Run this command:"
                echo "    python scripts/extract_yawdd_frames.py \\"
                echo "        --src $PATH \\"
                echo "        --dst data/raw/yawdd_frames \\"
                echo "        --fps 5"
            elif [ "$TOTAL_IMG" -gt 0 ]; then
                ok "$TOTAL_IMG extracted frames found"
            fi
            ;;
        "MRL Eye")
            OPEN_COUNT=$(find "$PATH" -type d \( -iname "open_eye" -o -iname "open" -o -iname "openlefteyes" \) 2>/dev/null | wc -l | tr -d ' ')
            CLOSED_COUNT=$(find "$PATH" -type d \( -iname "close_eye" -o -iname "closed" -o -iname "closedlefteyes" \) 2>/dev/null | wc -l | tr -d ' ')
            SUBJECT_COUNT=$(find "$PATH" -maxdepth 1 -type d -name "s[0-9]*" 2>/dev/null | wc -l | tr -d ' ')
            [ "$OPEN_COUNT" -gt 0 ] && ok "Open-eye folder found" || warn "No open-eye folder"
            [ "$CLOSED_COUNT" -gt 0 ] && ok "Closed-eye folder found" || warn "No closed-eye folder"
            [ "$SUBJECT_COUNT" -gt 0 ] && info "Subject subfolders found ($SUBJECT_COUNT subjects)"
            [ "$TOTAL_IMG" -gt 5000 ] && ok "$TOTAL_IMG images found" \
                || warn "Only $TOTAL_IMG images — MRL should have ~84,000"
            ;;
        "CEW")
            CLOSED_COUNT=$(find "$PATH" -type d \( -iname "closedLeftEyes" -o -iname "closed_eyes" -o -iname "closed" \) 2>/dev/null | wc -l | tr -d ' ')
            OPEN_COUNT=$(find "$PATH" -type d \( -iname "openLeftEyes" -o -iname "open_eyes" -o -iname "open" \) 2>/dev/null | wc -l | tr -d ' ')
            [ "$CLOSED_COUNT" -gt 0 ] && ok "Closed-eye folder found" || warn "No closed-eye folder found"
            [ "$OPEN_COUNT" -gt 0 ] && ok "Open-eye folder found" || warn "No open-eye folder found"
            [ "$TOTAL_IMG" -gt 1000 ] && ok "$TOTAL_IMG images" || warn "Only $TOTAL_IMG images"
            ;;
        "NTHU-DDD")
            LABEL_FILES=$(find "$PATH" -name "*drowsiness*.txt" 2>/dev/null | wc -l | tr -d ' ')
            [ "$LABEL_FILES" -gt 0 ] && ok "$LABEL_FILES drowsiness label files found" \
                || warn "No *drowsiness*.txt label files — eye labels will be missing"
            ;;
    esac
}

# Run inspection
inspect_dir "AUC v2"     "$AUC_PATH"    "yes"
inspect_dir "State Farm" "$SF_PATH"     "yes"
[ -n "$YAWDD_FRAMES" ] && \
    inspect_dir "YawDD"  "$YAWDD_FRAMES" "yes" || \
    inspect_dir "YawDD"  "$YAWDD_PATH"   "yes"
inspect_dir "MRL Eye"    "$MRL_PATH"    "no"
inspect_dir "CEW"        "$CEW_PATH"    "no"
inspect_dir "NTHU-DDD"   "$NTHU_PATH"   "no"

# ============================================================
# Summary and readiness check
# ============================================================
header "READINESS SUMMARY"

READY=1
[ -z "$AUC_PATH"   ] && { fail "AUC v2 missing (required)";   READY=0; }
[ -z "$SF_PATH"    ] && { fail "State Farm missing (required)"; READY=0; }
[ -z "$YAWDD_PATH" ] && [ -z "$YAWDD_FRAMES" ] && \
                         { fail "YawDD missing (required)";     READY=0; }

[ -n "$AUC_PATH"   ] && ok "AUC v2      — distraction + 2 camera angles"
[ -n "$SF_PATH"    ] && ok "State Farm  — distraction + 1 camera angle"
{ [ -n "$YAWDD_FRAMES" ] || [ -n "$YAWDD_PATH" ]; } && \
                         ok "YawDD       — yawn labels + 2 camera angles"
[ -n "$MRL_PATH"   ] && ok "MRL Eye     — eye state (84k images)"
[ -n "$CEW_PATH"   ] && ok "CEW         — eye state (2.4k images)"
[ -n "$NTHU_PATH"  ] && ok "NTHU-DDD    — eye state in-car"

echo ""
if [ -z "$MRL_PATH" ] && [ -z "$CEW_PATH" ] && [ -z "$NTHU_PATH" ]; then
    warn "No eye-state dataset found"
    warn "Eye accuracy will be 0%. Download MRL or CEW for eye detection."
    warn "See docs/dataset_guide.md for download instructions."
fi

echo ""
if [ $READY -eq 1 ]; then
    echo -e "${GREEN}✓ Ready to proceed!${NC}"
    echo ""
    echo "Next step:"
    echo "   bash prepare_data.sh"
else
    echo -e "${RED}✗ Missing required datasets. Fix above issues first.${NC}"
    echo ""
    echo "See: docs/dataset_guide.md"
fi
echo ""
