#!/usr/bin/env python3
"""Train the TCN per-queue hotspot predictor from a dataset produced
by ``scripts/gen_tcn_dataset.py``.

All design choices (loss, optimizer, LR schedule, batch size, early
stopping, RNG seed, weight decay, gradient clip) live in
``configs/ml/tcn_pred.yaml``. Every run of this script reads that YAML
plus the npz dataset and writes:

    models/tcn_pred.pt             : inference weights + norm + W
    models/tcn_pred_metrics.json   : train/val curves + final metrics

The saved weights file contains only what ``TCNPredictor`` needs at
inference; architecture constants are fixed in code (keeps the file
small and avoids shape drift).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.predictors.tcn import _TCN  # noqa: E402


def _build_model(model_cfg: Dict) -> nn.Module:
    return _TCN(
        in_ch=int(model_cfg["input_channels"]),
        ch=int(model_cfg["channels"]),
        k=int(model_cfg["kernel"]),
        layers=int(model_cfg["layers"]),
        sigmoid_output=False,
    )


def _build_optim(params, train_cfg: Dict):
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=float(train_cfg.get("momentum", 0.9)))
    raise ValueError(f"unknown optimizer {name!r}")


def _lr_at(step: int, total_steps: int, warmup_steps: int,
           base_lr: float, schedule: str) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if schedule == "cosine":
        # Cosine decay from base_lr down to base_lr/100 after warmup.
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        floor = base_lr / 100.0
        return floor + 0.5 * (base_lr - floor) * (1.0 + math.cos(math.pi * t))
    return base_lr


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mse": 0.0, "mae": 0.0, "pearson": 0.0,
                "hot_precision@0.35": 0.0}
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    if y_true.std() > 0 and y_pred.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0
    mask = y_true > 0.35
    if mask.any():
        hot_precision = float(
            np.mean(y_pred[mask] > 0.35))  # of truly-hot labels, fraction
        # that the model also flags as hot (threshold 0.35).
    else:
        hot_precision = 0.0
    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "hot_recall@0.35": hot_precision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-config", default="configs/ml/tcn_pred.yaml")
    args = ap.parse_args()

    ml_path = args.ml_config
    if not os.path.isabs(ml_path):
        ml_path = os.path.join(ROOT, ml_path)
    with open(ml_path) as f:
        ml = yaml.safe_load(f)

    model_cfg = ml["model"]
    train_cfg = ml["training"]
    data_cfg = ml["data"]
    out_cfg = ml["output"]

    seed = int(train_cfg.get("seed", 0))
    _set_seeds(seed)

    device = torch.device("cpu")
    print(f"device: {device}")

    ds_path = os.path.join(ROOT, data_cfg["dataset_path"])
    if not os.path.exists(ds_path):
        raise SystemExit(f"missing dataset: {ds_path}; run "
                         f"scripts/gen_tcn_dataset.py first")
    z = np.load(ds_path, allow_pickle=True)
    X_train = torch.from_numpy(z["X_train"]).float()
    y_train = torch.from_numpy(z["y_train"]).float()
    w_train = torch.from_numpy(z["w_train"]).float()
    X_val = torch.from_numpy(z["X_val"]).float()
    y_val = torch.from_numpy(z["y_val"]).float()
    w_val = torch.from_numpy(z["w_val"]).float()
    ch_mean = torch.from_numpy(z["channel_mean"]).float().view(1, -1, 1)
    ch_std = torch.from_numpy(z["channel_std"]).float().view(1, -1, 1)
    W_ckpt = int(z["W"])

    if W_ckpt != int(model_cfg["window"]):
        raise SystemExit(
            f"dataset W={W_ckpt} but model.window={model_cfg['window']}; "
            f"regenerate dataset or change the YAML")

    # Z-score normalize inputs (train stats).
    X_train = (X_train - ch_mean) / ch_std
    X_val = (X_val - ch_mean) / ch_std

    print(f"train: X {tuple(X_train.shape)}  y {tuple(y_train.shape)}")
    print(f"val  : X {tuple(X_val.shape)}    y {tuple(y_val.shape)}")

    model = _build_model(model_cfg).to(device)
    print(f"model params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_name = str(train_cfg.get("loss", "bce_with_logits")).lower()
    if loss_name == "bce_with_logits":
        def _loss_fn(logits, target, weight):
            per = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            return (per * weight).sum() / (weight.sum() + 1e-12)
    elif loss_name == "mse":
        def _loss_fn(logits, target, weight):
            pred = torch.sigmoid(logits)
            per = (pred - target) ** 2
            return (per * weight).sum() / (weight.sum() + 1e-12)
    else:
        raise SystemExit(f"unknown loss {loss_name!r}")

    opt = _build_optim(model.parameters(), train_cfg)

    batch_size = int(train_cfg["batch_size"])
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    base_lr = float(train_cfg["lr"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, w_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, w_val),
        batch_size=1024,
        shuffle=False,
        drop_last=False,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    history = {"train_loss": [], "val_loss": [], "val": []}
    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience_left = patience
    step = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for X, y, w in train_loader:
            lr = _lr_at(step, total_steps, warmup_steps, base_lr, schedule)
            for pg in opt.param_groups:
                pg["lr"] = lr
            logits = model(X)
            loss = _loss_fn(logits, y, w)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
        train_loss = float(np.mean(losses)) if losses else 0.0

        model.eval()
        with torch.no_grad():
            preds = []
            for X, _, _ in val_loader:
                preds.append(torch.sigmoid(model(X)).cpu().numpy())
            y_pred = (np.concatenate(preds, axis=0)
                      if preds else np.zeros(0, dtype=np.float32))
        val = _metrics(y_val.numpy(), y_pred)

        # Use MSE on (y, sigmoid(logit)) as the val stopping metric for
        # comparability across loss choices.
        val_loss = val["mse"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val"].append(val)

        improved = val_loss < best_val - 1e-6
        tag = " *" if improved else ""
        print(f"  epoch {epoch+1:03d}/{max_epochs:03d}  "
              f"lr={lr:.2e}  train={train_loss:.5f}  "
              f"val_mse={val['mse']:.5f}  "
              f"val_mae={val['mae']:.5f}  "
              f"val_pearson={val['pearson']:+.3f}  "
              f"hot_recall={val['hot_recall@0.35']:.3f}{tag}")

        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_preds, tr_targets = [], []
        for X, y, _ in DataLoader(
                TensorDataset(X_train, y_train, w_train),
                batch_size=1024, shuffle=False):
            tr_preds.append(torch.sigmoid(model(X)).cpu().numpy())
            tr_targets.append(y.cpu().numpy())
        y_tr_pred = (np.concatenate(tr_preds) if tr_preds
                     else np.zeros(0, dtype=np.float32))
        y_tr = (np.concatenate(tr_targets) if tr_targets
                else np.zeros(0, dtype=np.float32))
        train_final = _metrics(y_tr, y_tr_pred)

        va_preds = []
        for X, _, _ in val_loader:
            va_preds.append(torch.sigmoid(model(X)).cpu().numpy())
        y_va_pred = (np.concatenate(va_preds) if va_preds
                     else np.zeros(0, dtype=np.float32))
        val_final = _metrics(y_val.numpy(), y_va_pred)

    dt = time.time() - t0
    print()
    print("=== training complete ===")
    print(f"  elapsed: {dt:.1f}s")
    print(f"  TRAIN: {train_final}")
    print(f"  VAL  : {val_final}")

    weights_path = os.path.join(ROOT, out_cfg["weights_file"])
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "norm_mean": ch_mean.view(-1).cpu().numpy(),
        "norm_std": ch_std.view(-1).cpu().numpy(),
        "W": int(W_ckpt),
    }, weights_path)
    print(f"  wrote weights -> {weights_path}")

    metrics_path = os.path.join(ROOT, out_cfg["metrics_file"])
    with open(metrics_path, "w") as f:
        json.dump({
            "history": history,
            "train_final": train_final,
            "val_final": val_final,
            "loss": loss_name,
            "num_params": int(sum(p.numel()
                                  for p in model.parameters()
                                  if p.requires_grad)),
            "elapsed_s": dt,
        }, f, indent=2)
    print(f"  wrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
