"""Patent-claims validation table.

Runs (all that apply):
  1. Model-size introspection (FP32).
  2. Latency benchmark using the real-time pipeline on synthetic frames.
  3. If --manifest is provided: evaluation metrics on the test split.
  4. Privacy logger self-test: writes a few events, verifies chain, asserts
     no raw frame data was persisted.

Every row prints the ACHIEVED value from the current run, not a hardcoded
number. Targets come from ``configs/config.yaml``. If a test cannot be run
on this machine (e.g., no GPU for INT8 TRT) the row is marked N/A, not
silently PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# ------------------------------------------------------------------
# Pretty-printing helpers
# ------------------------------------------------------------------
_BORDER_T = "╔" + "═" * 74 + "╗"
_BORDER_M = "╠" + "═" * 74 + "╣"
_BORDER_B = "╚" + "═" * 74 + "╝"


def _row(metric: str, target: str, achieved: str, status: str) -> str:
    metric = metric[:22].ljust(22)
    target = target[:14].ljust(14)
    achieved = achieved[:14].ljust(14)
    status = status[:14].ljust(14)
    return f"║ {metric} ║ {target} ║ {achieved} ║ {status} ║"


def _status(ok: Optional[bool], na: bool = False) -> str:
    if na:
        return "N/A"
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------
# Individual benchmarks
# ------------------------------------------------------------------
def bench_model_size() -> Dict[str, float]:
    """FP32 model size by parameter count (no torch state_dict write needed)."""
    try:
        import torch   # noqa: F401
        from src.models.multitask_model import MultiTaskDMSModel
    except ImportError as e:
        return {"fp32_mb": float("nan"), "int8_mb": float("nan"),
                "error": f"torch unavailable: {e}"}
    m = MultiTaskDMSModel(pretrained=False)
    return {"fp32_mb": m.model_size_mb(4), "int8_mb": m.model_size_mb(1),
            "params": sum(p.numel() for p in m.parameters())}


def bench_latency(model_path: Optional[str], backend: str, n_frames: int = 200) -> Dict[str, float]:
    if not model_path or not os.path.exists(model_path):
        return {"status": "no_checkpoint"}
    try:
        from src.inference.realtime_pipeline import PipelineConfig, RealTimeDMSPipeline
        from src.pdsi.scoring import PDSIConfig
    except ImportError as e:
        return {"status": f"import_error: {e}"}
    pipe = RealTimeDMSPipeline(
        model_path=model_path, backend=backend,
        pipeline_cfg=PipelineConfig(),
        pdsi_cfg=PDSIConfig(calibration_frames=30),
    )
    return pipe.benchmark_latency(n_frames=n_frames)


def bench_eval(ckpt: Optional[str], manifest: Optional[str],
               config: Dict[str, Any]) -> Dict[str, float]:
    if not ckpt or not os.path.exists(ckpt) or not manifest or not os.path.exists(manifest):
        return {"status": "skipped", "reason": "missing ckpt or manifest"}
    try:
        from torch.utils.data import DataLoader
        from src.data.unified_dataset import UnifiedDMSDataset
        from src.evaluation.evaluator import evaluate_checkpoint
    except ImportError as e:
        return {"status": f"import_error: {e}"}
    ds = UnifiedDMSDataset(
        manifest, split="test",
        input_size=tuple(config["model"]["input_size"]), augment=False,
    )
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"],
                        shuffle=False, num_workers=2)
    res = evaluate_checkpoint(ckpt, loader, output_dir="./logs/bench_eval")
    return res["summary"]


def bench_privacy(tmp_dir: str) -> Dict[str, Any]:
    from src.privacy.logger import LoggerConfig, PrivacyPreservingLogger
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    logger = PrivacyPreservingLogger(LoggerConfig(log_dir=tmp_dir, rotate_every_mb=1))
    for i in range(3):
        logger.log_event(
            event_type="BENCHMARK_EVENT",
            pdsi_score=20.0 + i * 15.0,
            task_scores={"eye": 0.1, "yawn": 0.1, "pose": 0.1, "distraction": 0.1},
            alert_level="NORMAL", driver_id="bench_driver",
        )
    ok, n = logger.verify_chain_integrity()

    # Scan every log line and assert none contain forbidden keys or look like
    # base64 image blobs (>1KB string fields are suspicious).
    forbidden = {"frame_data", "frame", "image", "embedding",
                 "raw_landmarks", "landmarks", "pixels"}
    raw_bytes_found = 0
    for fp in sorted(Path(tmp_dir).glob("events_*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, v in rec.items():
                    if k in forbidden:
                        raw_bytes_found += 1
                    if isinstance(v, str) and len(v) > 1024:
                        raw_bytes_found += 1
    return {
        "chain_valid": bool(ok),
        "records_verified": int(n),
        "raw_bytes_found": raw_bytes_found,
        "privacy_ok": bool(ok and raw_bytes_found == 0),
    }


def bench_pdsi_overhead(n: int = 5000) -> Dict[str, float]:
    """Measure pure-Python P-DSI update() cost."""
    from src.pdsi.scoring import PDSIConfig, PDSIScorer
    scorer = PDSIScorer(PDSIConfig(calibration_frames=10))
    sample = {
        "eye_state": {"confidence_open": 0.8},
        "yawn": {"active": False, "confidence": 0.1},
        "head_pose": {"yaw": 5.0, "pitch": 3.0},
        "distraction": {"label": "safe", "confidence": 0.05},
    }
    t = time.perf_counter()
    for _ in range(n):
        scorer.update(sample)
    elapsed = time.perf_counter() - t
    return {"mean_ms_per_update": (elapsed / n) * 1000.0, "n": n}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./configs/config.yaml")
    ap.add_argument("--ckpt", default="./checkpoints/best.pth")
    ap.add_argument("--manifest", default="./data/processed/manifest.csv")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"])
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--json-out", default="./logs/benchmark_report.json")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    targets = {
        "eye_acc": 0.88, "eye_auc": 0.90,
        "yawn_acc": 0.88, "yawn_auc": 0.90,
        "pose_mae": 8.0, "dist_acc": 0.90, "weighted_f1": 0.88,
        "infer_fp32_ms": 20.0, "infer_int8_ms": 8.0,
        "e2e_ms": float(config["deployment"]["target_latency_ms"]),
        "fp32_model_mb": 25.0, "int8_model_mb": 7.0, "pdsi_overhead_ms": 5.0,
    }

    size = bench_model_size()
    latency = bench_latency(args.ckpt, args.backend, n_frames=args.n_frames)
    evaluated = bench_eval(args.ckpt, args.manifest, config) if not args.skip_eval else {"status": "skipped"}
    privacy = bench_privacy("./events_bench_tmp")
    pdsi = bench_pdsi_overhead()

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "targets": targets,
        "model_size": size,
        "latency": latency,
        "eval": evaluated,
        "privacy": privacy,
        "pdsi": pdsi,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=float))

    # ---------------- Table ---------------------------------------
    lines: List[str] = []
    lines.append(_BORDER_T)
    lines.append("║" + "      PATENT CLAIMS VALIDATION TABLE — RealTime-DMS v1.0".ljust(75) + "║")
    lines.append(_BORDER_M)
    lines.append(_row("METRIC", "TARGET", "ACHIEVED", "STATUS"))
    lines.append(_BORDER_M)

    def _cls(name: str, target_key: str, unit: str, higher_better: bool,
             fmt: str = "{:.4f}") -> Tuple[str, str, str]:
        if not isinstance(evaluated, dict) or evaluated.get("status") == "skipped":
            return f"> {targets[target_key]}{unit}", "N/A", _status(None, na=True)
        val = evaluated.get(name)
        if val is None:
            return f"> {targets[target_key]}{unit}", "N/A", _status(None, na=True)
        ok = val > targets[target_key] if higher_better else val < targets[target_key]
        return f"{'>' if higher_better else '<'} {targets[target_key]}{unit}", fmt.format(val) + unit, _status(ok)

    for row_name, key in (("Eye State Acc", "eye_acc"), ("Eye AUC", "eye_auc"),
                          ("Yawn Acc", "yawn_acc"), ("Yawn AUC", "yawn_auc"),
                          ("Distraction Acc", "dist_acc"),
                          ("Weighted Avg F1", "weighted_f1")):
        t, a, s = _cls(key, key, "", True, "{:.4f}")
        lines.append(_row(row_name, t, a, s))
    # Pose (lower is better). When eval is skipped/missing, _cls returns
    # a target string prefixed with ">"; fix that up for this row.
    t, a, s = _cls("pose_mae", "pose_mae", "°", False, "{:.2f}")
    t = t.replace(">", "<")
    lines.append(_row("Head Pose MAE", t, a, s))

    # Latency
    if isinstance(latency, dict) and "mean_ms" in latency:
        lines.append(_row(
            "E2E Latency (mean)", f"< {targets['e2e_ms']:.0f}ms",
            f"{latency['mean_ms']:.2f}ms",
            _status(latency['mean_ms'] < targets['e2e_ms']),
        ))
        lines.append(_row(
            "E2E Latency (P95)", f"< {targets['e2e_ms']*1.5:.0f}ms",
            f"{latency['p95_ms']:.2f}ms",
            _status(latency['p95_ms'] < targets['e2e_ms'] * 1.5),
        ))
        lines.append(_row(
            "FPS", f"> {config['deployment']['target_fps']}",
            f"{latency['fps']:.2f}",
            _status(latency['fps'] > config['deployment']['target_fps']),
        ))
    else:
        for r in ("E2E Latency (mean)", "E2E Latency (P95)", "FPS"):
            lines.append(_row(r, "—", "N/A", "N/A"))

    # Model sizes
    if not np.isnan(size.get("fp32_mb", float("nan"))):
        lines.append(_row(
            "FP32 Model Size", f"< {targets['fp32_model_mb']}MB",
            f"{size['fp32_mb']:.2f}MB",
            _status(size['fp32_mb'] < targets['fp32_model_mb']),
        ))
        lines.append(_row(
            "INT8 Model Size (est)", f"< {targets['int8_model_mb']}MB",
            f"{size['int8_mb']:.2f}MB",
            _status(size['int8_mb'] < targets['int8_model_mb']),
        ))
    else:
        lines.append(_row("FP32 Model Size", f"< {targets['fp32_model_mb']}MB", "N/A", "N/A"))
        lines.append(_row("INT8 Model Size (est)", f"< {targets['int8_model_mb']}MB", "N/A", "N/A"))

    # P-DSI overhead
    lines.append(_row(
        "P-DSI Overhead", f"< {targets['pdsi_overhead_ms']}ms",
        f"{pdsi['mean_ms_per_update']:.3f}ms",
        _status(pdsi["mean_ms_per_update"] < targets["pdsi_overhead_ms"]),
    ))

    # Privacy rows — these are always checkable
    lines.append(_row(
        "Privacy: no raw data", "0 raw keys",
        f"{privacy['raw_bytes_found']}",
        _status(privacy["raw_bytes_found"] == 0),
    ))
    lines.append(_row(
        "Chain Integrity", "100% valid",
        "VALID" if privacy["chain_valid"] else "BROKEN",
        _status(privacy["chain_valid"]),
    ))

    # ----- robustness rows (per-angle / per-lighting minima) -----
    # These answer: "what's the WORST angle/lighting accuracy?" — a system
    # that's 85% overall but 30% at mirror-mount has failed the angle claim.
    per_angle_info = None
    per_lighting_info = None
    if isinstance(evaluated, dict) and evaluated.get("status") != "skipped":
        # evaluate_checkpoint stored per-angle in a separate file; reload if present.
        import json as _json
        rep_path = Path("./logs/bench_eval/test_results.json")
        if rep_path.exists():
            try:
                rep = _json.loads(rep_path.read_text())
                per_angle_info = rep.get("per_angle") or {}
                per_lighting_info = rep.get("per_lighting") or {}
            except Exception:
                pass
    if per_angle_info:
        eye_accs = [v.get("eye_acc", 0) for v in per_angle_info.values()]
        min_angle_eye = min(eye_accs) if eye_accs else 0.0
        lines.append(_row(
            "Min per-angle eye acc", "> 0.70",
            f"{min_angle_eye:.4f}",
            _status(min_angle_eye > 0.70),
        ))
        # Spread (max - min) between angles — closer to 0 = more robust.
        spread = max(eye_accs) - min(eye_accs) if len(eye_accs) > 1 else 0.0
        lines.append(_row(
            "Angle spread (eye)", "< 0.15",
            f"{spread:.4f}",
            _status(spread < 0.15),
        ))
    if per_lighting_info and len(per_lighting_info) > 1:
        eye_accs = [v.get("eye_acc", 0) for v in per_lighting_info.values()]
        min_light_eye = min(eye_accs) if eye_accs else 0.0
        lines.append(_row(
            "Min per-lighting eye acc", "> 0.70",
            f"{min_light_eye:.4f}",
            _status(min_light_eye > 0.70),
        ))

    lines.append(_BORDER_B)
    print("\n".join(lines))
    print(f"\nFull JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
