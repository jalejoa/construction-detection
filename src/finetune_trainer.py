# finetune_trainer.py
# Fine-tuning U-Net (pretrained on buildings) to detect construction sites (binary) with BCE (+Dice).
#
# Expected dataset output:
#   image: (8, 256, 256) float32
#   mask : (1, 256, 256) float32 in {0,1}
#
# Saves:
#   data/models/construction_ft/<out_name>/{best.pth,last.pth,history.npy}

import time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import DATA_PROCESSED, DATA_MODELS
from patches import ConstructionJitterDataset
from model import UNet, DoubleConv, bce_dice_loss, iou_f1_from_logits, set_seed

def freeze_module(m: nn.Module, freeze: bool):
    for p in m.parameters():
        p.requires_grad = (not freeze)

def set_bn_eval(m: nn.Module):
    # keep BN fixed (no running stats update) for frozen parts
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            mod.eval()
@torch.no_grad()
def _pretty_print_metrics(prefix: str, m: dict):
    """
    Pretty, stable printing. Works even if some keys are missing.
    """
    keys_main = ["loss", "iou", "f1", "global_iou", "global_f1", "precision", "recall"]
    parts = []
    for k in keys_main:
        if k in m:
            v = m[k]
            if isinstance(v, float):
                parts.append(f"{k} {v:.4f}" if k == "loss" else f"{k} {v:.3f}")
            else:
                parts.append(f"{k} {v}")
    # Add TP/FP/FN if present
    for k in ["tp", "fp", "fn"]:
        if k in m:
            parts.append(f"{k.upper()} {m[k]}")
    print(prefix + " | " + " | ".join(parts))

def print_eval_report(metrics, title="Evaluation"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"Loss            : {metrics['loss']:.4f}")
    print(f"Mean IoU        : {metrics['iou']:.4f}")
    print(f"Mean F1         : {metrics['f1']:.4f}")

    if "global_iou" in metrics:
        print(f"Global IoU      : {metrics['global_iou']:.4f}")
    if "global_f1" in metrics:
        print(f"Global F1       : {metrics['global_f1']:.4f}")

    if "precision" in metrics:
        print(f"Precision       : {metrics['precision']:.4f}")
    if "recall" in metrics:
        print(f"Recall          : {metrics['recall']:.4f}")

    if "tp" in metrics:
        print("\nConfusion (pixel-level)")
        print(f"TP: {metrics['tp']:,}")
        print(f"FP: {metrics['fp']:,}")
        print(f"FN: {metrics['fn']:,}")

    print(f"{'='*60}\n")

# ---------------- Paths ----------------
def resolve_construction_paths(args):
    macro_csv  = Path(args.macro_csv)  if args.macro_csv  else (Path(DATA_PROCESSED) / "cropped_images" / "macro_catalog_filtered.csv")
    splits_csv = Path(args.splits_csv) if args.splits_csv else (Path(DATA_PROCESSED) / "site_splits.csv")
    if not macro_csv.exists():
        raise FileNotFoundError(f"macro_catalog.csv not found: {macro_csv}")
    if not splits_csv.exists():
        raise FileNotFoundError(f"site_splits.csv not found: {splits_csv}")
    return macro_csv, splits_csv

def resolve_pretrained_ckpt(args):
    ckpt = Path(args.pretrained_ckpt) if args.pretrained_ckpt else (
        Path(DATA_MODELS) / "building_pretrain" / args.pretrained_name / "best.pth"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")
    return ckpt


# ---------------- Trainer ----------------
class ConstructionTrainer:
    def __init__(self, args):
        self.args = args

        self.device = None
        self.device_type = None

        self.outdir = None
        self.best_path = None
        self.last_path = None
        self.hist_path = None

        self.macro_csv = None
        self.splits_csv = None
        self.ckpt_path = None

        self.dl_tr = None
        self.dl_va = None
        self.dl_te = None


        self.model = None
        self.scaler = GradScaler(enabled=False)
        self.optimizer = None

        self.history = {
            "epoch": [],
            "phase": [],
            "train_loss": [], "val_loss": [],
            "train_iou": [], "val_iou": [],
            "train_f1": [], "val_f1": [],
            "lr": [],
        }

        self.best_val_iou = -1.0
        self.bad_epochs = 0

    # --- high-level flow ---

    def fit(self):
        self.prepare_environment()
        self.prepare_paths()
        self.prepare_data()
        self.prepare_model()

        # phase 1
        self.run_phase(
            phase=1,
            epochs=self.args.epochs_phase1,
            min_epochs=self.args.min_epochs_phase1,
            patience=self.args.patience_phase1,
            lr=self.args.lr_phase1,
            start_epoch=1,
            reset_bad_epochs=False,
        )

        # phase 2
        start_epoch2 = (self.history["epoch"][-1] + 1) if self.history["epoch"] else 1
        self.run_phase(
            phase=2,
            epochs=self.args.epochs_phase2,
            min_epochs=self.args.min_epochs_phase2,
            patience=self.args.patience_phase2,
            lr=self.args.lr_phase2,
            start_epoch=start_epoch2,
            reset_bad_epochs=True,  # like your original logic
        )

        self.save_history()
        print("Done.")
    # --- evaluation API (what you wanted) ---
    def load_checkpoint(self, which="best"):
        if which == "best":
            path = self.best_path
        elif which == "last":
            path = self.last_path
        else:
            path = Path(which)

        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def evaluate(self, *, split="val", thr=0.5, detailed=True):
        if split == "val":
            dl = self.dl_va
        elif split == "test":
            dl = self.dl_te
        else:
            raise ValueError(f"Unsupported split: {split}")

        return self._eval_one_epoch(dl=dl, thr=thr, detailed=detailed)


    def sweep_thresholds(self, *, split="val", thrs=None):
        if thrs is None:
            thrs = np.linspace(0.05, 0.95, 19)

        results = []
        for thr in thrs:
            m = self.evaluate(split=split, thr=float(thr), detailed=True)
            results.append({
                "thr": float(thr),
                "loss": m["loss"],
                "iou": m["iou"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            })
        return results


    # --- setup steps ---
    def prepare_environment(self):
        set_seed(self.args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.cpu else "cpu")
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        print(f"Device: {self.device.type}")

        self.scaler = GradScaler(enabled=(self.device_type == "cuda"))

    def prepare_paths(self):
        self.outdir = Path(DATA_MODELS) / "construction_ft" / self.args.out_name
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.outdir / "best.pth"
        self.last_path = self.outdir / "last.pth"
        self.hist_path = self.outdir / "history.npy"

        print("Output folder:", self.outdir)
        print("Will save best to:", self.best_path)
        print("Will save last to:", self.last_path)

    def prepare_data(self):
        self.macro_csv, self.splits_csv = resolve_construction_paths(self.args)
        self.ckpt_path = resolve_pretrained_ckpt(self.args)

        print("macro_catalog:", self.macro_csv)
        print("site_splits  :", self.splits_csv)
        print("Pretrained ckpt:", self.ckpt_path)

        train_ds = ConstructionJitterDataset(
            macro_catalog_csv=self.macro_csv,
            processed_root=Path(DATA_PROCESSED),
            splits_csv=self.splits_csv,
            split="train",
            crop_size=self.args.crop_size,
            jitter_radius=self.args.jitter_radius,
            pos_fraction=self.args.pos_fraction,
            norm_stats_path=self.args.norm_stats_path,
        )

        val_ds = ConstructionJitterDataset(
            macro_catalog_csv=self.macro_csv,
            processed_root=Path(DATA_PROCESSED),
            splits_csv=self.splits_csv,
            split="val",
            crop_size=self.args.crop_size,
            jitter_radius=0,
            pos_fraction=0.5,
            norm_stats_path=self.args.norm_stats_path,
        )

        test_ds = ConstructionJitterDataset(
            macro_catalog_csv=self.macro_csv,
            processed_root=Path(DATA_PROCESSED),
            splits_csv=self.splits_csv,
            split="test",
            crop_size=self.args.crop_size,
            jitter_radius=0,
            pos_fraction=0.5,  # o 0.5 para determinístico; NO uses pos_fraction alto aquí
            norm_stats_path=self.args.norm_stats_path,
        )

        print(f"Samples | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        print("jitter(val):", val_ds.jitter_radius, "pos_frac(val):", val_ds.pos_fraction)
        print("jitter(test):", test_ds.jitter_radius, "pos_frac(test):", test_ds.pos_fraction)

        nw = self.args.num_workers
        self.dl_tr = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True,
            num_workers=nw, pin_memory=(self.device_type=="cuda"),
            persistent_workers=(nw > 0)
        )
        self.dl_va = DataLoader(
            val_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=nw, pin_memory=(self.device_type=="cuda"),
            persistent_workers=(nw > 0)
        )
        self.dl_te = DataLoader(
            test_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=nw, pin_memory=(self.device_type=="cuda"),
            persistent_workers=(nw > 0)
        )


    def prepare_model(self):
        self.model = UNet(in_channels=self.args.in_channels, base_c=self.args.base_c).to(self.device)

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        print("Loaded pretrained weights (strict=True).")

    # --- phase control ---
    def configure_phase(self, phase: int, lr: float, reset_bad_epochs: bool):
        # freeze/unfreeze
        if phase == 1:
            freeze_module(self.model.d1, True)
            freeze_module(self.model.d2, True)
            freeze_module(self.model.d3, True)
            freeze_module(self.model.d4, True)
            freeze_module(self.model.b,  True)
            # ensure BN in frozen parts stays fixed
            self._keep_frozen_bn_eval(phase=1)
            print("Phase 1: encoder+bottleneck frozen. Training decoder+head.")
        else:
            freeze_module(self.model, False)
            print("Phase 2: unfrozen all layers.")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.args.wd)
        
        #--- DEBUG
        print("d1 requires_grad any:", any(p.requires_grad for p in self.model.d1.parameters()))
        print("c1 requires_grad any:", any(p.requires_grad for p in self.model.c1.parameters()))

        if reset_bad_epochs:
            self.bad_epochs = 0

    def run_phase(self, *, phase: int, epochs: int, min_epochs: int, patience: int, lr: float,
                  start_epoch: int, reset_bad_epochs: bool):
        self.configure_phase(phase=phase, lr=lr, reset_bad_epochs=reset_bad_epochs)

        for epoch in range(start_epoch, start_epoch + epochs):
            t0 = time.time()

            tr = self._train_one_epoch(phase=phase, thr=self.args.threshold)
            va = self._eval_one_epoch(dl=self.dl_va, thr=self.args.threshold, detailed=False)

            elapsed = time.time() - t0
            print(f"[P{phase}] Epoch {epoch:03d} | {elapsed:.1f}s")
            _pretty_print_metrics("  train", tr)
            _pretty_print_metrics("  val  ", va)


            self._save_last(epoch=epoch, phase=phase)

            # early stop tracking (based on val IoU)
            if va["iou"] > self.best_val_iou + self.args.min_delta:
                self.best_val_iou = va["iou"]
                self.bad_epochs = 0
                self._save_best(epoch=epoch, phase=phase, val_iou=self.best_val_iou)
                print(f"  ↳ new best val IoU: {self.best_val_iou:.3f} (checkpoint saved)")
            else:
                self.bad_epochs += 1

            self._append_history(epoch=epoch, phase=phase, lr=lr, tr=tr, va=va)

            if epoch >= (start_epoch - 1 + min_epochs) and self.bad_epochs >= patience:
                print(f"[P{phase}] Early stopping: no improvement for {self.bad_epochs} epochs.")
                break

    # --- internal train/eval ---
    def _keep_frozen_bn_eval(self, phase: int):
        if phase == 1:
            set_bn_eval(self.model.d1)
            set_bn_eval(self.model.d2)
            set_bn_eval(self.model.d3)
            set_bn_eval(self.model.d4)
            set_bn_eval(self.model.b)

    def _train_one_epoch(self, *, phase: int, thr: float):
        self.model.train()
        self._keep_frozen_bn_eval(phase)

        sum_loss = sum_iou = sum_f1 = 0.0
        n = 0

        for img, msk, *_ in self.dl_tr:
            img = img.to(self.device, non_blocking=True)
            msk = msk.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, enabled=(self.device_type=="cuda")):
                logits = self.model(img)
                loss = bce_dice_loss(logits, msk)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = img.size(0)
            sum_loss += loss.item() * bs
            iou_b, f1_b = iou_f1_from_logits(logits.detach(), msk, thr=thr)
            sum_iou += iou_b * bs
            sum_f1  += f1_b  * bs
            n += bs

        denom = max(1, n)
        return {"loss": float(sum_loss/denom), "iou": float(sum_iou/denom), "f1": float(sum_f1/denom), "n": int(n)}

    @torch.no_grad()
    def _eval_one_epoch(self, *, dl, thr: float, detailed: bool):
        self.model.eval()

        sum_loss = 0.0
        sum_iou = 0.0
        sum_f1 = 0.0
        n = 0

        TP = FP = FN = 0
        inter_g = 0.0
        union_g = 0.0

        for img, msk, *_ in dl:
            img = img.to(self.device, non_blocking=True)
            msk = msk.to(self.device, non_blocking=True)

            with autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                logits = self.model(img)
                loss = bce_dice_loss(logits, msk)

            bs = img.size(0)
            sum_loss += loss.item() * bs

            iou_b, f1_b = iou_f1_from_logits(logits, msk, thr=thr)
            sum_iou += iou_b * bs
            sum_f1  += f1_b  * bs
            n += bs

            probs = torch.sigmoid(logits)
            pred = (probs >= thr).int()
            gt   = (msk >= 0.5).int()

            inter = (pred & gt).sum().item()
            uni   = (pred | gt).sum().item()

            inter_g += inter
            union_g += uni

            TP += inter
            FP += (pred & (1 - gt)).sum().item()
            FN += ((1 - pred) & gt).sum().item()

        denom = max(1, n)
        eps = 1e-8

        out = {
            "loss": float(sum_loss / denom),
            "iou":  float(sum_iou  / denom),   # mean-per-image IoU (macro)
            "f1":   float(sum_f1   / denom),   # mean-per-image F1 (macro)
            "global_iou": float((inter_g + eps) / (union_g + eps)) if union_g > 0 else 0.0,
            "global_f1":  float((2*TP + eps) / (2*TP + FP + FN + eps)),
            "n":    int(n),
        }

        if detailed:
            precision = TP / (TP + FP + eps)
            recall    = TP / (TP + FN + eps)
            out.update({
                "precision": float(precision),
                "recall": float(recall),
                "tp": int(TP), "fp": int(FP), "fn": int(FN),
            })

        return out


    # --- persistence / logging ---
    def _save_last(self, *, epoch: int, phase: int):
        torch.save(
            {"epoch": epoch, "phase": phase, "model": self.model.state_dict(), "opt": self.optimizer.state_dict()},
            self.last_path
        )

    def _save_best(self, *, epoch: int, phase: int, val_iou: float):
        torch.save(
            {"epoch": epoch, "phase": phase, "model": self.model.state_dict(),
             "opt": self.optimizer.state_dict(), "val_iou": float(val_iou)},
            self.best_path
        )
    def _append_history(self, *, epoch: int, phase: int, lr: float, tr: dict, va: dict):
            self.history["epoch"].append(epoch)
            self.history["phase"].append(phase)
            self.history["train_loss"].append(tr["loss"]); self.history["val_loss"].append(va["loss"])
            self.history["train_iou"].append(tr["iou"]);   self.history["val_iou"].append(va["iou"])
            self.history["train_f1"].append(tr["f1"]);     self.history["val_f1"].append(va["f1"])
            self.history["lr"].append(lr)

    def save_history(self):
            np.save(self.hist_path, self.history)
            print("Saved history to:", self.hist_path)


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Fine-tune buildings->construction (BCE)")

    # data
    ap.add_argument("--macro_csv", type=str, default=None)
    ap.add_argument("--splits_csv", type=str, default=None)

    # pretrained
    ap.add_argument("--pretrained_name", type=str, default="unet_8b_v5")
    ap.add_argument("--pretrained_ckpt", type=str, default=None)

    # output
    ap.add_argument("--out_name", type=str, default="cs_ft_v1_unetv5")

    # model
    ap.add_argument("--in_channels", type=int, default=8)
    ap.add_argument("--base_c", type=int, default=64)

    # sampling
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--jitter_radius", type=int, default=20)
    ap.add_argument("--pos_fraction", type=float, default=0.7)

    # runtime
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)

    # metrics/earlystop
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_delta", type=float, default=0.001)

    # phase 1
    ap.add_argument("--epochs_phase1", type=int, default=20)
    ap.add_argument("--min_epochs_phase1", type=int, default=10)
    ap.add_argument("--patience_phase1", type=int, default=6)
    ap.add_argument("--lr_phase1", type=float, default=1e-4)

    # phase 2
    ap.add_argument("--epochs_phase2", type=int, default=80)
    ap.add_argument("--min_epochs_phase2", type=int, default=20)
    ap.add_argument("--patience_phase2", type=int, default=12)
    ap.add_argument("--lr_phase2", type=float, default=1e-5)

    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--norm_stats_path", type=str, default=None, help="Path to normalization stats file used by ConstructionJitterDataset (optional).")


    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = ConstructionTrainer(args)
    trainer.fit()

    # Example usage after training:
    # trainer.load_checkpoint("best")
    # res = trainer.evaluate(split="val", thr=0.35, detailed=True)
    # print(res)
    #
    # sweep = trainer.sweep_thresholds(split="val", thrs=np.linspace(0.1, 0.9, 17))
    # print(sorted(sweep, key=lambda x: x["f1"], reverse=True)[:5])