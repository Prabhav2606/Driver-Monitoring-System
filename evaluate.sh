#!/bin/bash
# Full evaluation after training. Run: bash evaluate.sh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
header(){ echo -e "\n${BLUE}=== $1 ===${NC}"; }

# Find best checkpoint
BEST=$(ls checkpoints/best_epoch*.pth 2>/dev/null | \
    python3 -c "
import sys,re
lines=[l.strip() for l in sys.stdin if l.strip()]
if not lines: print('checkpoints/best.pth')
else:
    def s(p): m=re.search(r'(\d+\.\d+)\.pth',p); return float(m.group(1)) if m else 0
    print(sorted(lines,key=s)[-1])
" 2>/dev/null || echo "checkpoints/best.pth")
[ -f "$BEST" ] || { echo "No checkpoint found. Run: bash train.sh"; exit 1; }
ok "Using: $BEST"

# Training summary
header "Training summary (last 30 lines)"
tail -60 logs/overnight.log 2>/dev/null | grep -E "(epoch|dist_acc|yawn_acc|w_f1|improved|early)" | tail -15

# Evaluation
header "Test-set evaluation (~10 min)"
python3 evaluate.py --ckpt "$BEST" \
    --manifest data/processed/manifest.csv \
    --out logs/eval_report
ok "Evaluation complete"

# Print results
header "RESULTS"
python3 - << 'PYEOF'
import json; from pathlib import Path
r=json.loads(Path("logs/eval_report/test_results.json").read_text())
s=r["summary"]
print()
print("╔══════════════════════════════════════════════════════╗")
print("║            FINAL PERFORMANCE METRICS                ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Distraction Accuracy  :  {s.get('dist_acc',0)*100:6.2f}%                ║")
print(f"║  Distraction F1        :  {s.get('dist_f1',0):.4f}                  ║")
print(f"║  Yawn Accuracy         :  {s.get('yawn_acc',0)*100:6.2f}%                ║")
print(f"║  Yawn AUC              :  {s.get('yawn_auc',0):.4f}                  ║")
print(f"║  Eye State Accuracy    :  {s.get('eye_acc',0)*100:6.2f}%                ║")
print(f"║  Head Pose MAE         :  {s.get('pose_mae',0):6.2f}°                ║")
print(f"║  Weighted F1 (overall) :  {s.get('weighted_f1',0):.4f}                  ║")
print("╠══════════════════════════════════════════════════════╣")
print("║  PER CAMERA ANGLE                                    ║")
for ang,m in sorted(r.get("per_angle",{}).items()):
    print(f"║    {ang:<22}: dist={m['dist_acc']*100:.0f}%  yawn={m['yawn_acc']*100:.0f}%  n={m['n']:,} ║")
print("╠══════════════════════════════════════════════════════╣")
print("║  PER DATASET                                         ║")
for ds,m in sorted(r.get("per_dataset",{}).items()):
    print(f"║    {ds:<22}: dist={m['dist_acc']*100:.0f}%  yawn={m['yawn_acc']*100:.0f}%  n={m['n']:,} ║")
print("╚══════════════════════════════════════════════════════╝")
print(); print("Plots: logs/eval_report/*.png")
PYEOF

# Benchmark table
header "Patent claims validation"
python3 benchmark.py --ckpt "$BEST" \
    --manifest data/processed/manifest.csv \
    --json-out logs/benchmark_report.json

# ONNX export
header "ONNX export"
mkdir -p exports
python3 -m src.deployment.export_onnx --ckpt "$BEST" \
    --out exports/dms_model.onnx 2>&1 | tail -3
ls -lh exports/dms_model.onnx 2>/dev/null && ok "ONNX exported"

# Open plots
echo ""
echo "Opening result plots..."
open logs/eval_report/*.png 2>/dev/null || echo "  (open logs/eval_report/*.png manually)"

echo ""
echo "================================================"
echo " Done! Next:"
echo "   bash demo.sh       ← live webcam test"
echo "   bash dashboard.sh  ← web dashboard"
echo "================================================"
