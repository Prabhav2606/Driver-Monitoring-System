"""Master training script.

Usage::
    python train.py --manifest ./data/processed/manifest.csv --config ./configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.unified_dataset import UnifiedDMSDataset
from src.models.multitask_model import MultiTaskDMSModel
from src.training.trainer import Trainer, TrainerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic off by default because it kills throughput; opt in by
    # setting env var DMS_DETERMINISTIC=1 if you need exact reproduction.
    import os
    if os.environ.get("DMS_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(manifest: str, cfg: dict):
    common = dict(
        input_size=tuple(cfg["model"]["input_size"]),
        angle_prob=cfg["data"]["augmentation"]["angle_simulation_prob"],
        pseudo_label_weight=cfg["data"].get("pseudo_label_weight", 0.5),
        min_face_quality=cfg["data"].get("min_face_quality", 0.0),
    )
    train_ds = UnifiedDMSDataset(manifest, split="train", augment=True, **common)
    val_ds = UnifiedDMSDataset(manifest, split="val", augment=False, **common)
    test_ds = UnifiedDMSDataset(manifest, split="test", augment=False, **common)

    # Print split stats once.
    print("[data] TRAIN split stats:")
    print(train_ds.get_split_statistics().to_string(index=False))

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    pm = cfg["data"]["pin_memory"]
    # pin_memory only works on CUDA — disable on MPS/CPU to avoid warnings.
    import torch as _torch
    _pm = pm and _torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=_pm, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=_pm)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=_pm)
    return train_ds, train_loader, val_loader, test_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="./data/processed/manifest.csv")
    ap.add_argument("--config", default="./configs/config.yaml")
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    ap.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["project"]["seed"])

    train_ds, train_loader, val_loader, test_loader = build_loaders(args.manifest, cfg)
    class_weights = train_ds.get_class_weights()
    print(f"[loss] class weights: " + ", ".join(f"{k}={v.tolist()}" for k, v in class_weights.items()))

    model = MultiTaskDMSModel(
        pretrained=cfg["model"]["pretrained"],
        shared_dim=cfg["model"]["shared_dim"],
        dropout=cfg["model"]["dropout"],
    )
    print(f"[model] params: {model.count_parameters()}")
    print(f"[model] FP32 size (est): {model.model_size_mb(4):.2f} MB")
    print(f"[model] INT8 size (est): {model.model_size_mb(1):.2f} MB")

    tr_cfg_dict = cfg["training"]
    trainer_cfg = TrainerConfig(
        epochs=args.epochs or tr_cfg_dict["epochs"],
        batch_size=tr_cfg_dict["batch_size"],
        lr=tr_cfg_dict["learning_rate"],
        weight_decay=tr_cfg_dict["weight_decay"],
        gradient_clip=tr_cfg_dict["gradient_clip"],
        amp=tr_cfg_dict["amp"],
        freeze_backbone_epochs=tr_cfg_dict["freeze_backbone_epochs"],
        warmup_epochs=tr_cfg_dict["scheduler"]["warmup_epochs"],
        min_lr_ratio=tr_cfg_dict["scheduler"]["min_lr"] / tr_cfg_dict["learning_rate"],
        pose_anneal_epochs=tr_cfg_dict["loss"]["pose_anneal_epochs"],
        focal_alpha=tr_cfg_dict["loss"]["focal_alpha"],
        focal_gamma=tr_cfg_dict["loss"]["focal_gamma"],
        early_stopping_patience=tr_cfg_dict["early_stopping_patience"],
        ckpt_dir=cfg["paths"]["checkpoints"],
        log_dir=cfg["paths"]["logs"],
        task_weights={k: v["weight"] for k, v in cfg["model"]["task_heads"].items()},
    )

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[resume] loaded weights from {args.resume}")

    trainer = Trainer(model, train_loader, val_loader, trainer_cfg, class_weights=class_weights)
    result = trainer.fit()

    # Final test set evaluation.
    from src.evaluation.evaluator import evaluate_checkpoint
    best = Path(trainer_cfg.ckpt_dir) / "best.pth"
    if best.exists():
        print("[test] evaluating best.pth on held-out test set…")
        evaluate_checkpoint(str(best), test_loader, output_dir=trainer_cfg.log_dir)

    print("\n[done] best val weighted F1:", result["best_val_weighted_f1"])


if __name__ == "__main__":
    main()
