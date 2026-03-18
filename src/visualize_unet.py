# visualize_unet.py
# Visualize UNet predictions (val/test) with GT + overlay (TP/FP/FN)
#
# Usage from notebook:
#   from visualize_unet import run_visualization
#   res = run_visualization(dataset="pretraining", split="val", out_name="unet_8b_v1",
#                           ckpt="best.pth", n_samples=8, threshold=0.5)
#
# Or CLI:
#   python -m src.visualize_unet --dataset pretraining --split val --out_name unet_8b_v1 --ckpt best.pth --n_samples 10

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DATA_DIR, DATA_MODELS
from model import UNet, set_seed
from pretrain_unet import TilesDataset, resolve_dataset_paths, load_ids, read_tif

# Prefer torch.amp.autocast (works with device_type argument)
from torch.amp import autocast


# -------------------- helpers --------------------

def _percentile_stretch(img: np.ndarray, p_low=2, p_high=98, eps=1e-8) -> np.ndarray:
    """Stretch per-channel to [0, 1] using percentiles. img: [H,W,3]."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        lo = np.percentile(img[..., c], p_low)
        hi = np.percentile(img[..., c], p_high)
        out[..., c] = (img[..., c] - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0)


def _make_rgb_from_multiband(arr_chw: np.ndarray, rgb_bands_1based: List[int]) -> np.ndarray:
    """
    arr_chw: [C,H,W] float32
    rgb_bands_1based: e.g., [6,4,2] for (R,G,B) using 1-based indexing
    returns: [H,W,3] float32 in [0,1] (stretched)
    """
    idx = [b - 1 for b in rgb_bands_1based]
    if max(idx) >= arr_chw.shape[0]:
        raise ValueError(f"rgb_bands {rgb_bands_1based} out of range for C={arr_chw.shape[0]}")
    rgb = np.stack([arr_chw[idx[0]], arr_chw[idx[1]], arr_chw[idx[2]]], axis=-1)  # [H,W,3]
    rgb = _percentile_stretch(rgb, 2, 98)
    return rgb


def _overlay_tpfpfn(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha=0.45) -> np.ndarray:
    """
    rgb: [H,W,3] float in [0,1]
    gt/pred: [H,W] {0,1}
    Overlay colors:
      TP = green, FP = red, FN = blue
    """
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    overlay = rgb.copy()

    # Colors (RGB)
    col_tp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    col_fp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    col_fn = np.array([0.0, 0.4, 1.0], dtype=np.float32)

    for mask, col in [(tp, col_tp), (fp, col_fp), (fn, col_fn)]:
        if mask.any():
            overlay[mask] = (1 - alpha) * overlay[mask] + alpha * col

    return np.clip(overlay, 0.0, 1.0)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_ckpt_path(out_name: str, ckpt: str) -> Path:
    """
    If ckpt is an existing path -> use it.
    Else -> interpret relative to data/models/building_pretrain/<out_name>/<ckpt>
    """
    c = Path(ckpt)
    if c.exists():
        return c

    base = DATA_MODELS / "building_pretrain" / out_name
    return base / ckpt


def _infer_in_channels(img_dir: Path, ids: List[str], bands: Optional[List[int]]) -> int:
    sample = read_tif(img_dir / f"{ids[0]}.tif").astype(np.float32)  # [C,H,W]
    if bands:
        idx = [b - 1 for b in bands]
        sample = sample[idx, ...]
    return int(sample.shape[0])


# -------------------- main API --------------------

def run_visualization(
    dataset: str = "pretraining",
    split: str = "val",
    out_name: str = "unet_8b_v1",
    ckpt: str = "best.pth",
    threshold: float = 0.5,
    n_samples: int = 8,
    seed: int = 42,
    cpu: bool = False,
    bands: Optional[List[int]] = None,
    base_c: int = 64,
    rgb_bands: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    make_montage: bool = False,
) -> Dict:
    """
    Generates PNG visualizations for a subset of tiles in a split.

    Returns dict with output folder and selected tile ids.
    """
    assert split in ["val", "test"], "split must be 'val' or 'test'"
    set_seed(seed)

    # Defaults
    if bands is None:
        bands = [1,2,3,4,5,6,7,8]
    if rgb_bands is None:
        # PlanetScope 8-band typical display: R=Red(6), G=Green(4), B=Blue(2)
        rgb_bands = [6, 4, 2]

    # --- Auto-load config from the training run folder (recommended) ---
    run_outdir = DATA_MODELS / "building_pretrain" / out_name
    cfg_path = run_outdir / "run_config.json"

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # If user didn't specify, take from config
        if bands is None:
            bands = cfg.get("bands", bands)
        if base_c is None:
            base_c = cfg.get("base_c", base_c)
        # Optional: if later you store rgb_bands/threshold in config
        # if rgb_bands is None:
        #     rgb_bands = cfg.get("rgb_bands", rgb_bands)
        # if threshold is None:
        #     threshold = cfg.get("threshold", threshold)

    # Fallbacks if still None
    if bands is None:
        bands = [1,2,3,4,5,6,7,8]
    if base_c is None:
        base_c = 64

    # Resolve dataset paths
    class _Args:
        pass
    args = _Args()
    args.dataset = dataset
    args.images = None
    args.masks = None
    args.splits = None
    args.stats = None

    paths = resolve_dataset_paths(args, DATA_DIR)
    img_dir: Path = paths["images"]
    msk_dir: Path = paths["masks"]
    splits_dir: Path = paths["splits"]
    stats_path: Path = paths["stats"]

    # Split ids
    split_file = splits_dir / f"{split}.txt"
    ids_all = load_ids(split_file)
    if len(ids_all) == 0:
        raise RuntimeError(f"No ids found in: {split_file}")

    # Choose samples
    n = min(int(n_samples), len(ids_all))
    ids = random.sample(ids_all, k=n)

    # Output folder
    if save_dir is None:
        save_dir_path = DATA_MODELS / "building_pretrain" / out_name / "viz" / split
    else:
        save_dir_path = Path(save_dir)
    _ensure_dir(save_dir_path)

    # Load stats (normalization)
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    mean, std = stats["mean"], stats["std"]

    # Device
    device = torch.device("cuda" if (torch.cuda.is_available() and not cpu) else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_on = (device_type == "cuda")

    # Infer channels
    in_ch = _infer_in_channels(img_dir, ids_all, bands)

    # Dataset (no augmentation)
    ds = TilesDataset(ids, img_dir, msk_dir, mean, std, augment=False, bands=bands)

    # Model + checkpoint
    model = UNet(in_channels=in_ch, base_c=base_c).to(device)
    ckpt_path = _resolve_ckpt_path(out_name, ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if "model" not in state:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    model.load_state_dict(state["model"])
    model.eval()

    # Run each tile -> save figure
    saved = []
    for i in range(n):
        x_t, y_t, tid = ds[i]  # x: [C,H,W] torch, y: [1,H,W] torch
        x = x_t.unsqueeze(0).to(device)  # [1,C,H,W]
        y = y_t.squeeze(0).numpy().astype(np.uint8)  # [H,W] 0/1

        with torch.no_grad():
            with autocast(device_type=device_type, enabled=amp_on):
                logits = model(x)  # [1,1,H,W]
            probs = torch.sigmoid(logits)[0, 0].float().cpu().numpy()  # [H,W]
        pred = (probs >= float(threshold)).astype(np.uint8)

        # For visualization RGB we should use RAW (not normalized) to look nice.
        # So we re-read the raw tif for tid:
        raw = read_tif(img_dir / f"{tid}.tif").astype(np.float32)  # [C,H,W]
        if bands:
            raw = raw[[b-1 for b in bands], ...]

        rgb = _make_rgb_from_multiband(raw, rgb_bands)

        overlay = _overlay_tpfpfn(rgb, gt=y, pred=pred, alpha=0.45)

        # Quick stats per tile (optional but useful)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        # Figure
        fig = plt.figure(figsize=(14, 4))

        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"{tid}\nRGB bands {rgb_bands}")
        ax1.axis("off")

        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(y, vmin=0, vmax=1)
        ax2.set_title("GT mask")
        ax2.axis("off")

        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(pred, vmin=0, vmax=1)
        ax3.set_title(f"Pred (thr={threshold:.2f})")
        ax3.axis("off")

        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(overlay)
        ax4.set_title(f"Overlay TP/FP/FN\nTP={tp:,} FP={fp:,} FN={fn:,}")
        ax4.axis("off")

        plt.tight_layout()

        out_png = save_dir_path / f"{tid}_viz.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved.append(str(out_png))

    # Montage (optional)
    montage_path = None
    if make_montage and n > 0:
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        for j in range(n):
            img = plt.imread(saved[j])
            ax = plt.subplot(rows, cols, j + 1)
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        montage_path = save_dir_path / f"montage_{split}_{out_name}.png"
        fig.savefig(montage_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "dataset": dataset,
        "split": split,
        "out_name": out_name,
        "ckpt": str(ckpt_path),
        "threshold": float(threshold),
        "n_samples": int(n),
        "selected_ids": ids,
        "save_dir": str(save_dir_path),
        "saved_pngs": saved,
        "montage": str(montage_path) if montage_path else None,
    }


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Visualize UNet building predictions (val/test).")
    ap.add_argument("--dataset", type=str, default="pretraining")
    ap.add_argument("--split", type=str, choices=["val", "test"], default="val")
    ap.add_argument("--out_name", type=str, default="unet_8b_v1")
    ap.add_argument("--ckpt", type=str, default="best.pth")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base_c", type=int, default=64)
    ap.add_argument("--bands", type=int, nargs="+", default=[1,2,3,4,5,6,7,8])
    ap.add_argument("--rgb_bands", type=int, nargs="+", default=[6,4,2], help="3 numbers (R G B), 1-based")
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--no_montage", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    res = run_visualization(
        dataset=a.dataset,
        split=a.split,
        out_name=a.out_name,
        ckpt=a.ckpt,
        threshold=a.threshold,
        n_samples=a.n_samples,
        seed=a.seed,
        cpu=a.cpu,
        bands=a.bands,
        base_c=a.base_c,
        rgb_bands=a.rgb_bands,
        save_dir=a.save_dir,
        make_montage=(not a.no_montage),
    )
    print(res)
