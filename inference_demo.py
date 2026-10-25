"""Webcam live demo for the DMS pipeline.

    python inference_demo.py --model ./checkpoints/best.pth
    python inference_demo.py --model ./exports/dms_model.onnx --backend onnx
"""
from __future__ import annotations

import argparse

from src.inference.realtime_pipeline import PipelineConfig, RealTimeDMSPipeline
from src.pdsi.scoring import PDSIConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./checkpoints/best.pth")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibration-seconds", type=int, default=60,
                    help="Shorter calibration for demo; use 300 for production")
    args = ap.parse_args()

    pipe = RealTimeDMSPipeline(
        model_path=args.model,
        backend=args.backend,
        pipeline_cfg=PipelineConfig(driver_id="demo_driver"),
        pdsi_cfg=PDSIConfig(calibration_frames=args.calibration_seconds * 30),
    )
    print(f"[demo] starting webcam={args.camera}. Press Esc to quit.")
    pipe.run_webcam_demo(camera_id=args.camera)


if __name__ == "__main__":
    main()
