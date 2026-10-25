"""Evaluate a trained checkpoint on the test split."""
from __future__ import annotations

import argparse

import yaml
from torch.utils.data import DataLoader

from src.data.unified_dataset import UnifiedDMSDataset
from src.evaluation.evaluator import evaluate_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./checkpoints/best.pth")
    ap.add_argument("--manifest", default="./data/processed/manifest.csv")
    ap.add_argument("--config", default="./configs/config.yaml")
    ap.add_argument("--out", default="./logs/eval_report")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = UnifiedDMSDataset(
        args.manifest, split="test",
        input_size=tuple(cfg["model"]["input_size"]), augment=False,
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=cfg["data"]["num_workers"])
    evaluate_checkpoint(args.ckpt, loader, output_dir=args.out)


if __name__ == "__main__":
    main()
