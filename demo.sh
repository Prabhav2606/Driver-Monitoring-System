#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"; source .venv/bin/activate
BEST=$(ls checkpoints/best_epoch*.pth 2>/dev/null | python3 -c "
import sys,re; lines=[l.strip() for l in sys.stdin if l.strip()]
if not lines: print('checkpoints/best.pth')
else:
    def s(p): m=re.search(r'(\d+\.\d+)\.pth',p); return float(m.group(1)) if m else 0
    print(sorted(lines,key=s)[-1])" 2>/dev/null || echo "checkpoints/best.pth")
[ -f "$BEST" ] || { echo "No checkpoint. Run: bash train.sh"; exit 1; }
echo "Live webcam demo — ESC to quit"
python3 inference_demo.py --model "$BEST"
