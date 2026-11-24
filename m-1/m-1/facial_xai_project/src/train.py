from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASS_NAMES, FER2013
from .model import FERClassifier
from .muon import Muon, build_muon_optimizer
from .soap import SOAP


# ─────────────────────────────────────────────────────────────
#  Terminal helpers
# ─────────────────────────────────────────────────────────────

def _bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _save_atomic(state: dict, path: Path):
    """Atomically save a PyTorch checkpoint to avoid corrupt files during crashes/power-cuts."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def _save_json_atomic(data, path: Path):
    """Atomically save JSON file to prevent corrupted null-byte files on sudden shutdown."""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def _print_header(args, device, train_size: int, val_size: int,
                  n_batches: int, counts: np.ndarray):
    eff_bs = args.batch_size * args.grad_accum
    if args.optimizer == "muon":
        opt_desc = f"Muon (Dec 2024, NS-5 hidden lr={args.muon_lr} + Aux AdamW lr={args.lr})"
    elif args.optimizer == "soap":
        opt_desc = f"SOAP (ICLR 2025, eigenbasis precond freq={args.soap_freq})"
    else:
        opt_desc = "AdamW"

    if args.patience > 0:
        start_note = f", active from epoch {args.early_stop_start_epoch}" if args.early_stop_start_epoch > 0 else ""
        es_info = f"{args.patience} epochs on '{args.early_stopping_metric}' (min_delta={args.min_delta}{start_note})"
    else:
        es_info = "Disabled"

    print()
    print(_hr("═"))
    print("  FER-2013  ·  Training Run")
    print(_hr("═"))
    print(f"  Model         : {args.model}  (pretrained={args.pretrained}, amp={args.amp})")
    print(f"  Optimizer     : {opt_desc}")
    print(f"  Device        : {device}"
          + (f"  [{torch.cuda.get_device_name(0)}]" if device.type == "cuda" else ""))
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}"
          + (f" (effective {eff_bs} via {args.grad_accum}x accum)" if args.grad_accum > 1 else "")
          + f"   ({n_batches} batches/epoch)")
    print(f"  LR / decay    : {args.lr}  /  {args.weight_decay}")
    print(f"  Early stop    : {es_info}")
    print(f"  Auto-resume   : {args.resume} (checkpoint: latest.pt)")
    print(f"  Seed          : {args.seed}")
    print(f"  Train images  : {train_size:,}   |   Val images: {val_size:,}")
    print(f"  Output dir    : {args.output}")
    print(_hr())
    print("  Class distribution (train):")
    for i, name in enumerate(CLASS_NAMES):
        bar = "▮" * int(30 * counts[i] / counts.sum())
        print(f"    {name:<10}  {counts[i]:>5,}  {bar}")
    print(_hr("═"))
    print()


def _print_epoch_header(epoch: int, total_epochs: int, lr: float):
    pct = int(100 * (epoch - 1) / total_epochs)
    overall_bar = _bar(epoch - 1, total_epochs, 20)
    print(f"\n{'─'*72}")
    print(f"  Epoch {epoch:>2}/{total_epochs}  {overall_bar}  "
          f"{pct:>3}% done   LR={lr:.2e}")
    print(f"{'─'*72}")


def _print_batch_progress(batch_idx: int, n_batches: int,
                           loss: float, seen: int,
                           batch_start: float):
    elapsed   = time.time() - batch_start
    speed     = seen / max(elapsed, 1e-6)        # images/sec
    eta_secs  = (n_batches - batch_idx - 1) * (elapsed / max(batch_idx + 1, 1))
    bar       = _bar(batch_idx + 1, n_batches, 24)
    # Overwrite the same line
    line = (f"  Batch {batch_idx+1:>4}/{n_batches}  {bar}  "
            f"loss={loss:.4f}  {speed:>6.0f} img/s  "
            f"ETA {_fmt_time(eta_secs)}   ")
    print(f"\r{line}", end="", flush=True)


def _print_epoch_summary(epoch: int, total_epochs: int,
                          train_loss: float, val: dict,
                          epoch_time: float, best_acc: float,
                          best_f1: float, is_best: bool,
                          patience: int = 0, patience_counter: int = 0,
                          monitored: str = "accuracy",
                          start_epoch_gate: int = 0):
    print()   # newline after batch progress line
    tag = f"  *** NEW BEST ({monitored.upper()}) ***" if is_best else ""
    print(f"\n  {'─'*68}")
    print(f"  Epoch {epoch:>2}/{total_epochs} complete  ({_fmt_time(epoch_time)}){tag}")
    print(f"  {'─'*68}")
    print(f"  Train loss  : {train_loss:.4f}")
    print(f"  Val loss    : {val['loss']:.4f}")
    print(f"  Val acc     : {val['accuracy']*100:.2f}%   (best so far: {best_acc*100:.2f}%)")
    print(f"  Val macro-F1: {val['macro_f1']*100:.2f}%   (best so far: {best_f1*100:.2f}%)")
    if patience > 0:
        if start_epoch_gate > 0 and epoch < start_epoch_gate:
            print(f"  Early stop  : Inactive until epoch {start_epoch_gate} ({epoch}/{start_epoch_gate} epochs reached)")
        else:
            bar_len = min(patience, 20)
            filled = int(bar_len * patience_counter / max(patience, 1))
            pbar = "▮" * filled + "░" * (bar_len - filled)
            print(f"  Early stop  : [{pbar}] {patience_counter}/{patience} epochs without {monitored} gain")
    print(f"  {'─'*68}")


def _print_per_class(val_preds: list[int], val_true: list[int]):
    """Print per-class accuracy in a small table."""
    print("\n  Per-class accuracy (val):")
    for i, name in enumerate(CLASS_NAMES):
        mask = [t == i for t in val_true]
        if not any(mask):
            continue
        correct = sum(p == i for p, t in zip(val_preds, val_true) if t == i)
        total   = sum(mask)
        acc     = correct / total
        bar     = "▮" * int(20 * acc)
        print(f"    {name:<10}  {acc*100:>5.1f}%  {bar}")


def _print_final_summary(history: list[dict], best_acc: float, best_f1: float,
                          best_epoch: int, total_time: float,
                          out_dir: str, early_stopped: bool = False):
    print()
    print(_hr("═"))
    status_label = "TRAINING COMPLETE (EARLY STOPPED)" if early_stopped else "TRAINING COMPLETE"
    print(f"  {status_label}")
    print(_hr("═"))
    print(f"  Total time    : {_fmt_time(total_time)}")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val acc  : {best_acc*100:.2f}%")
    print(f"  Best val F1   : {best_f1*100:.2f}%")
    print(f"  Best model    : {out_dir}/best.pt")
    print(f"  Latest model  : {out_dir}/latest.pt")
    print(_hr("═"))
    print()


# ─────────────────────────────────────────────────────────────
#  Core evaluation (now returns predictions for per-class table)
# ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, use_amp: bool = False):
    model.eval()
    all_y, all_pred = [], []
    loss_fn  = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_y.extend(y.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss":     total_loss / max(total, 1),
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(all_y, all_pred, average="macro", zero_division=0),
        "_preds":   all_pred,
        "_true":    all_y,
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--seed",         type=int,   default=1)
    parser.add_argument("--model",
                        choices=["convnext_tiny", "regnet_y_800mf", "resnet18"],
                        default="convnext_tiny")
    parser.add_argument("--optimizer",    choices=["muon", "soap", "adamw"], default="muon",
                        help="Optimizer: 'muon' (Keller Jordan Dec 2024), 'soap' (ICLR 2025), or 'adamw'")
    parser.add_argument("--muon-lr",      type=float, default=0.02,
                        help="Learning rate for Muon 2D/4D hidden layer weights (default: 0.02)")
    parser.add_argument("--soap-freq",    type=int,   default=10,
                        help="Eigenbasis preconditioning frequency for SOAP (paper default: 10)")
    parser.add_argument("--beta1",        type=float, default=0.9,
                        help="beta1 for Adam/SOAP momentum")
    parser.add_argument("--beta2",        type=float, default=0.95,
                        help="beta2 for Adam/SOAP second moment")
    parser.add_argument("--grad-accum",   type=int,   default=1,
                        help="Gradient accumulation steps to scale effective batch size")
    parser.add_argument("--pretrained",   action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers",      type=int,   default=0)
    parser.add_argument("--amp",          action=argparse.BooleanOptionalAction, default=True,
                        help="Use mixed precision (fp16) for 3-4x speedup and lower VRAM")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience (epochs without gain, default: 15, 0 to disable)")
    parser.add_argument("--early-stop-start-epoch", type=int, default=0,
                        help="Do not trigger early stopping before reaching this epoch (default: 0)")
    parser.add_argument("--early-stopping-metric", choices=["accuracy", "macro_f1", "loss"], default="accuracy",
                        help="Metric to monitor for early stopping & best model (default: accuracy)")
    parser.add_argument("--min-delta",    type=float, default=1e-4,
                        help="Minimum change to qualify as an improvement (default: 1e-4)")
    parser.add_argument("--resume",       action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-resume from latest.pt in output dir if present (default: True)")
    parser.add_argument("--resume-from",  type=str,   default=None,
                        help="Explicit checkpoint file to resume from")
    parser.add_argument("--reset-patience", action=argparse.BooleanOptionalAction, default=False,
                        help="Reset patience counter to 0 on resume (default: False)")
    args = parser.parse_args()

    seed_everything(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Datasets ──────────────────────────────────────────────
    print("\n  Loading datasets...", end="", flush=True)
    t0 = time.time()
    train_ds = FER2013(args.csv, "train", train=True)
    val_ds   = FER2013(args.csv, "val",   train=False)
    print(f" done  ({time.time()-t0:.1f}s)")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    n_batches = len(train_dl)

    # ── Class weights ──────────────────────────────────────────
    counts       = np.bincount(train_ds.labels, minlength=len(CLASS_NAMES))
    class_weights = len(train_ds) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    # ── Model ─────────────────────────────────────────────────
    print("  Building model...", end="", flush=True)
    t0 = time.time()
    model = FERClassifier(args.model, pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" done  ({time.time()-t0:.1f}s)   {n_params/1e6:.2f}M trainable params")

    # ── Optimizer ─────────────────────────────────────────────
    if args.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
            nesterov=True,
            ns_steps=5,
            betas=(args.beta1, args.beta2),
        )
    elif args.optimizer == "soap":
        optimizer = SOAP(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            precondition_frequency=args.soap_freq,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == "cuda")

    # ── Checkpoint resume logic ───────────────────────────────
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume and (out / "latest.pt").is_file():
        resume_path = out / "latest.pt"

    start_epoch = 1
    best_acc = 0.0
    best_f1 = 0.0
    best_score = float("inf") if args.early_stopping_metric == "loss" else -1.0
    best_epoch = 1
    patience_counter = 0
    history = []

    if resume_path and resume_path.is_file():
        print(f"\n  Resuming training from checkpoint: {resume_path} ...", end="", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f" (warning: could not load optimizer state: {e})", end="")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f" (warning: could not load scheduler state: {e})", end="")
        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f" (warning: could not load scaler state: {e})", end="")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_f1 = ckpt.get("best_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 1)
        patience_counter = 0 if args.reset_patience else ckpt.get("patience_counter", 0)
        if "best_score" in ckpt:
            best_score = ckpt["best_score"]
        else:
            best_score = best_acc if args.early_stopping_metric == "accuracy" else best_f1
        history = ckpt.get("history", [])
        if not history and (out / "history.json").is_file():
            try:
                history = json.loads((out / "history.json").read_text())
            except Exception:
                pass
        print(f" done! Resuming at epoch {start_epoch}/{args.epochs}")
        p_str = f"{patience_counter}/{args.patience}" if args.patience > 0 else "Disabled"
        print(f"  Best {args.early_stopping_metric}: {best_score:.4f}  |  Early stop: {p_str}")

    # ── Print run header ──────────────────────────────────────
    _print_header(args, device, len(train_ds), len(val_ds),
                  n_batches, counts)

    # ── Training loop ─────────────────────────────────────────
    run_start = time.time()
    early_stopped = False

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else args.lr
        _print_epoch_header(epoch, args.epochs, current_lr)

        # — Train —
        model.train()
        running_loss = 0.0
        seen         = 0
        epoch_start  = time.time()
        batch_start  = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            raw_loss_val = (loss.item() * args.grad_accum) if args.grad_accum > 1 else loss.item()
            running_loss += raw_loss_val * y.size(0)
            seen         += y.size(0)

            # Print progress every batch
            _print_batch_progress(
                batch_idx, n_batches,
                running_loss / seen,
                seen, batch_start
            )

        scheduler.step()

        # — Validate —
        print(f"\r  Validating...{' '*60}", end="", flush=True)
        val = evaluate(model, val_dl, device, use_amp=args.amp)

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / max(seen, 1)

        # — Early stopping & Best checkpoint logic —
        if args.early_stopping_metric == "loss":
            current_metric_val = val["loss"]
            improved = (best_score - current_metric_val) > args.min_delta
        else:
            current_metric_val = val[args.early_stopping_metric]
            improved = (current_metric_val - best_score) > args.min_delta

        if improved:
            is_best = True
            best_score = current_metric_val
            best_epoch = epoch
            patience_counter = 0
        else:
            is_best = False
            patience_counter += 1

        best_acc = max(best_acc, val["accuracy"])
        best_f1  = max(best_f1, val["macro_f1"])

        ckpt_state = {
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "scaler":           scaler.state_dict() if scaler.is_enabled() else None,
            "architecture":     args.model,
            "pretrained":       args.pretrained,
            "seed":             args.seed,
            "epoch":            epoch,
            "class_names":      CLASS_NAMES,
            "val":              {k: v for k, v in val.items() if not k.startswith("_")},
            "best_score":       best_score,
            "best_acc":         best_acc,
            "best_f1":          best_f1,
            "best_epoch":       best_epoch,
            "patience_counter": patience_counter,
        }

        # Atomically save latest.pt every epoch (crash-proof)
        _save_atomic(ckpt_state, out / "latest.pt")

        # Atomically save best.pt whenever improvement is observed
        if is_best:
            _save_atomic(ckpt_state, out / "best.pt")

        # — Log & print summary —
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val["loss"], 6),
            "val_acc":    round(val["accuracy"], 6),
            "val_f1":     round(val["macro_f1"], 6),
            "epoch_secs": round(epoch_time, 1),
            "lr":         round(current_lr, 8),
            "patience":   patience_counter,
        }
        history.append(row)
        _save_json_atomic(history, out / "history.json")

        _print_epoch_summary(
            epoch, args.epochs, train_loss, val,
            epoch_time, best_acc, best_f1, is_best,
            patience=args.patience,
            patience_counter=patience_counter,
            monitored=args.early_stopping_metric,
            start_epoch_gate=args.early_stop_start_epoch,
        )

        # Print per-class table every 5 epochs and on the last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            _print_per_class(val["_preds"], val["_true"])

        # Check early stopping threshold (respecting early_stop_start_epoch)
        if args.patience > 0 and epoch >= args.early_stop_start_epoch and patience_counter >= args.patience:
            print(f"\n  [!] Early stopping triggered at epoch {epoch}: {args.patience} epochs without {args.early_stopping_metric} improvement (active after epoch {args.early_stop_start_epoch}).")
            early_stopped = True
            break

        # Remaining time estimate
        elapsed_total = time.time() - run_start
        epochs_done = epoch - start_epoch + 1
        avg_epoch_time = elapsed_total / max(epochs_done, 1)
        remaining = avg_epoch_time * (args.epochs - epoch)
        print(f"\n  Elapsed: {_fmt_time(elapsed_total)}   "
              f"Est. remaining: {_fmt_time(remaining)}")

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - run_start
    _print_final_summary(history, best_acc, best_f1, best_epoch, total_time, str(out), early_stopped=early_stopped)

    # ── Save run metadata ─────────────────────────────────────
    device_name = (torch.cuda.get_device_name(0)
                   if device.type == "cuda" else str(device))
    _save_json_atomic({
        "seed":                   args.seed,
        "model":                  args.model,
        "data_path":              str(Path(args.csv).resolve()),
        "validation_split_seed":  42,
        "input_size":             [224, 224],
        "normalization_mean_std": [0.5, 0.5],
        "pretrained":             args.pretrained,
        "amp":                    args.amp,
        "optimizer":              args.optimizer,
        "soap_freq":              args.soap_freq if args.optimizer == "soap" else None,
        "muon_lr":                args.muon_lr if args.optimizer == "muon" else None,
        "grad_accum":             args.grad_accum,
        "effective_batch_size":   args.batch_size * args.grad_accum,
        "device":                 str(device),
        "device_name":            device_name,
        "epochs":                 args.epochs,
        "batch_size":             args.batch_size,
        "lr":                     args.lr,
        "weight_decay":           args.weight_decay,
        "patience":               args.patience,
        "early_stop_start_epoch": args.early_stop_start_epoch,
        "early_stopping_metric":  args.early_stopping_metric,
        "min_delta":              args.min_delta,
        "early_stopped":          early_stopped,
        "best_epoch":             best_epoch,
        "best_val_acc":           round(best_acc, 6),
        "best_val_f1":            round(best_f1, 6),
        "total_seconds":          round(total_time, 1),
        "python":                 sys.version,
        "platform":               platform.platform(),
        "torch":                  torch.__version__,
        "torchvision":            torchvision.__version__,
        "class_names":            CLASS_NAMES,
        "class_counts":           counts.tolist(),
        "class_weights":          class_weights.tolist(),
    }, out / "run.json")


if __name__ == "__main__":
    main()
