#!/bin/bash
# One-time environment setup. Run: bash setup.sh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[!!]${NC} $1"; }
fail(){ echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo "================================================"
echo " DMS Setup — Pruthvi 22BCE2098 VIT Vellore"
echo "================================================"
echo ""

# Python check
PYVER=$(python3 --version 2>&1 | awk '{print $2}')
PYMAJ=$(echo $PYVER | cut -d. -f1)
PYMIN=$(echo $PYVER | cut -d. -f2)
[ "$PYMAJ" -ge 3 ] && [ "$PYMIN" -ge 9 ] || \
    fail "Python 3.9+ required (you have $PYVER). Run: brew install python@3.11"
ok "Python $PYVER"

# Virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv; ok "Created .venv"
else
    ok ".venv exists"
fi
source .venv/bin/activate

# Install deps
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# PyTorch + MPS check
python3 -c "
import torch
mps = torch.backends.mps.is_available()
print(f'  torch {torch.__version__}  MPS={mps}')
assert mps, 'MPS not available — check PyTorch version'
" || warn "MPS check failed — training will use CPU (slower)"

# ffmpeg check
command -v ffmpeg &>/dev/null && ok "ffmpeg available" || \
    warn "ffmpeg not found — run: brew install ffmpeg (needed for YawDD)"

mkdir -p data/raw data/processed checkpoints logs exports events
ok "Directories created"

echo ""
echo "================================================"
echo " Setup done! Next:"
echo "   bash inspect_datasets.sh"
echo "================================================"
