import json, argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_DIR, DATA_MODELS
from datetime import datetime
import rasterio
import random
import matplotlib.pyplot as plt

from model import UNet, DoubleConv, bce_dice_loss, iou_f1_from_logits, set_seed

def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read()  # [C,H,W] o [1,H,W]
    return arr

def load_ids(split_txt):
    return [ln.strip() for ln in Path(split_txt).read_text(encoding="utf-8").splitlines() if ln.strip()]

def resolve_dataset_paths(args, DATA_DIR):
    """
    Resolve dataset paths using:
    1) explicit CLI overrides if provided
    2) convention based on --dataset
    """

    dataset_dir = DATA_DIR / args.dataset

    # Convención base
    default_pre_dir = dataset_dir / "tiles"

    paths = {
        "images": Path(args.images) if args.images else default_pre_dir / "images",
        "masks":  Path(args.masks)  if args.masks  else default_pre_dir / "masks",
        "splits": Path(args.splits) if args.splits else default_pre_dir / "splits",
        "stats":  Path(args.stats)  if args.stats  else dataset_dir / "stats" / "norm_stats_building_8b.json",
    }

    # Sanity checks
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{k} path does not exist: {p}")

    return paths
# ------------- Dataset -------------
class TilesDataset(Dataset):
    def __init__(self, ids, img_dir, msk_dir, mean, std, augment=False, bands=None):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.msk_dir = Path(msk_dir)
        self.augment = augment
        self.bands = [b-1 for b in bands] if bands else None

        mean_arr = np.array(mean, dtype=np.float32)
        std_arr  = np.array(std,  dtype=np.float32)

        if self.bands is not None:
            mean_arr = mean_arr[self.bands]
            std_arr  = std_arr[self.bands]

        self.mean = mean_arr[:, None, None]
        self.std  = std_arr[:, None, None]


    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        tid = self.ids[idx]
        img = read_tif(self.img_dir / f"{tid}.tif").astype(np.float32)    # [C,H,W]
        if self.bands is not None:
            img = img[self.bands, ...]
        msk = read_tif(self.msk_dir / f"{tid}_mask.tif").astype(np.uint8) # [1,H,W] o [H,W]
        if msk.ndim == 3: msk = msk[0]
        msk = (msk > 0).astype(np.float32)[None, ...]  # [1,H,W]

        # augment mínimo
        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy(); msk = msk[:, :, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1,:].copy(); msk = msk[:, ::-1,:].copy()
            if random.random() < 0.3:
                img = np.rot90(img, k=1, axes=(1,2)).copy(); msk = np.rot90(msk, k=1, axes=(1,2)).copy()

        # normalización
        img = (img - self.mean) / (self.std + 1e-8)

        return torch.from_numpy(img), torch.from_numpy(msk), tid
    
def graphs(hist_file):
    hist = np.load(hist_file, allow_pickle=True).item()
    epochs = hist["epoch"]

    plt.figure(figsize=(12,4))

    # LOSS
    plt.subplot(1,3,1)
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    # IoU
    plt.subplot(1,3,2)
    plt.plot(epochs, hist["train_iou"], label="train")
    plt.plot(epochs, hist["val_iou"], label="val")
    plt.title("IoU")
    plt.legend()

    # F1
    plt.subplot(1,3,3)
    plt.plot(epochs, hist["train_f1"], label="train")
    plt.plot(epochs, hist["val_f1"], label="val")
    plt.title("F1-score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def _jsonable(obj):
    """Convert common non-JSON types (Path, numpy types) to JSON-serializable."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalars
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_run_config(args, outdir: Path, extra: dict | None = None):
    """
    Save the full run configuration to outdir/run_config.json
    This makes runs reproducible and allows auto-config in eval/visualize.
    """
    cfg = {k: _jsonable(v) for k, v in vars(args).items()}
    cfg["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        for k, v in extra.items():
            cfg[k] = _jsonable(v)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def load_run_config(outdir: Path) -> dict:
    p = outdir / "run_config.json"
    if not p.exists():
        raise FileNotFoundError(f"run_config.json not found in: {outdir}")
    return json.loads(p.read_text(encoding="utf-8"))

# ------------- Train -------------
def run(args):
    paths = resolve_dataset_paths(args, DATA_DIR)
    args.images = paths["images"]
    args.masks  = paths["masks"]
    args.splits = paths["splits"]
    args.stats  = paths["stats"]
    
    # Definir outdir basado en config
    args.outdir = str(DATA_MODELS / "building_pretrain" / args.out_name)


    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_f1": [],
        "val_f1": []
        }



    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device.type}")

    # AMP actualizado
    from torch.amp import GradScaler, autocast
    scaler = GradScaler(enabled=(device_type=="cuda"))

    # Splits & stats
    train_ids = load_ids(Path(args.splits, "train.txt"))
    val_ids   = load_ids(Path(args.splits, "val.txt"))
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    mean, std = stats["mean"], stats["std"]

    # Inferir canales con bandas opcionales
    sample_img = read_tif(Path(args.images, f"{train_ids[0]}.tif")).astype(np.float32)
    if args.bands:
        idx = [b-1 for b in args.bands]
        sample_img = sample_img[idx, ...]
    in_ch = sample_img.shape[0]
    print(f"In channels: {in_ch}")

    ds_tr = TilesDataset(train_ids, args.images, args.masks, mean, std, augment=True,  bands=args.bands)
    ds_va = TilesDataset(val_ids,   args.images, args.masks, mean, std, augment=False, bands=args.bands)
    nw = args.num_workers
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=nw, pin_memory=(device_type=="cuda"),
                    persistent_workers=(nw > 0))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                    num_workers=nw, pin_memory=(device_type=="cuda"),
                    persistent_workers=(nw > 0))

    model = UNet(in_channels=in_ch, base_c=args.base_c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# ---- Output paths (explicit logging) ----
    best_ckpt_path = outdir / "best.pth"
    last_ckpt_path = outdir / "last.pth"
    history_path   = outdir / "history.npy"
    config_path    = outdir / "run_config.json"

    print("\n=== Training outputs ===")
    print(f"Run directory : {outdir}")
    print(f"Best model    : {best_ckpt_path}")
    print(f"Last model    : {last_ckpt_path}")
    print(f"History file : {history_path}")
    print(f"Run config   : {config_path}")
    print("========================\n")
    


    # Save reproducible run config (so eval/visualize can auto-load bands/base_c/etc.)
    save_run_config(args, outdir)

    

    # Training loop
    best_val = -1.0
    bad_epochs = 0
    min_delta = 0.001  # prueba 0.001 o 0.002 para IoU

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0; tr_iou = 0.0; tr_f1 = 0.0; ntr = 0
        batch_times = []

        for img, msk, _ in dl_tr:
            tb = time.time()
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=(device_type=="cuda")):
                logits = model(img)
                loss = bce_dice_loss(logits, msk)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()*img.size(0)
            iou, f1 = iou_f1_from_logits(logits.detach(), msk)
            tr_iou += iou*img.size(0); tr_f1 += f1*img.size(0); ntr += img.size(0)

            batch_times.append(time.time() - tb)

        # Validation
        model.eval()
        va_loss = 0.0; va_iou = 0.0; va_f1 = 0.0; nva = 0
        with torch.no_grad():
            for img, msk, _ in dl_va:
                img = img.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)
                with autocast(device_type=device_type, enabled=(device_type=="cuda")):
                    logits = model(img)
                    loss = bce_dice_loss(logits, msk)
                va_loss += loss.item()*img.size(0)
                iou, f1 = iou_f1_from_logits(logits, msk)
                va_iou += iou*img.size(0); va_f1 += f1*img.size(0); nva += img.size(0)

        tr_loss /= max(1,ntr); tr_iou /= max(1,ntr); tr_f1 /= max(1,ntr)
        va_loss /= max(1,nva); va_iou /= max(1,nva); va_f1 /= max(1,nva)
        elapsed = time.time() - t0

        # Logs GPU
        gpu_log = ""
        if device_type == "cuda":
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / (1024**3)
            memr = torch.cuda.max_memory_allocated() / (1024**3)
            gpu_log = f" | GPU mem {mem:.2f}G (max {memr:.2f}G)"

        print(f"Epoch {epoch:03d} | {elapsed:.1f}s | "
              f"train loss {tr_loss:.4f} iou {tr_iou:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} iou {va_iou:.3f} f1 {va_f1:.3f}{gpu_log} | "
              f"~batch {np.mean(batch_times):.3f}s")

        # Save checkpoints
        torch.save({"epoch":epoch,"model":model.state_dict(),
                    "opt":optimizer.state_dict()}, outdir/"last.pth")
        if va_iou > best_val + min_delta:
            best_val = va_iou
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "val_iou": best_val,
                },
                outdir / "best.pth"
            )

            print(f"  ↳ new best val IoU: {best_val:.3f} (checkpoint saved)")
        else:
            bad_epochs += 1

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        # Early stopping
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {bad_epochs} epochs.")
            break

    np.save(outdir / "history.npy", history)
    graphs(outdir / "history.npy")



# ---------- EVALUATION ---------

def run_eval(args):
    """
    Evaluate a saved checkpoint on val or test split.
    Reports: IoU, F1 (from logits), Precision/Recall (from TP/FP/FN), plus counts.
    """
    from torch.amp import autocast

    set_seed(args.seed)

    # Resolve dataset paths from dataset name (or overrides)
    paths = resolve_dataset_paths(args, DATA_DIR)
    img_dir = paths["images"]
    msk_dir = paths["masks"]
    splits_dir = paths["splits"]
    stats_path = paths["stats"]

    # Output dir (where checkpoints live)
    outdir = DATA_MODELS / "building_pretrain" / args.out_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-load defaults from run_config.json if present (recommended)
    cfg_path = outdir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Only fill if user didn't explicitly set something meaningful
        if getattr(args, "bands", None) in (None, [],):
            args.bands = cfg.get("bands", args.bands)
        if getattr(args, "base_c", None) is None:
            args.base_c = cfg.get("base_c", args.base_c)
        # Optional: batch_size in eval (keep your explicit args.batch_size if you prefer)
        # if getattr(args, "batch_size", None) is None:
        #     args.batch_size = cfg.get("batch_size", args.batch_size)


    # Resolve checkpoint path:
    # - if args.ckpt is an existing path, use it
    # - else interpret as filename inside outdir
    ckpt_candidate = Path(args.ckpt)
    if ckpt_candidate.exists():
        ckpt_path = ckpt_candidate
    else:
        ckpt_path = outdir / args.ckpt

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Eval device: {device.type}")
    print(f"Split: {args.split} | threshold={args.threshold}")
    print(f"Checkpoint: {ckpt_path}")

    # Load split IDs
    split_file = splits_dir / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids = load_ids(split_file)
    print(f"Num tiles in {args.split}: {len(ids)}")

    # Load stats
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    mean, std = stats["mean"], stats["std"]

    # Infer in_channels robustly (use first tile)
    sample_img = read_tif(img_dir / f"{ids[0]}.tif").astype(np.float32)
    if args.bands:
        idx = [b - 1 for b in args.bands]
        sample_img = sample_img[idx, ...]
    in_ch = sample_img.shape[0]
    print(f"In channels: {in_ch}")

    # Dataset + DataLoader (no augmentation)
    ds = TilesDataset(ids, img_dir, msk_dir, mean, std, augment=False, bands=args.bands)

    nw = args.num_workers
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(nw > 0)
    )

    # Model + checkpoint
    model = UNet(in_channels=in_ch, base_c=args.base_c).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Metrics accumulators
    TP = 0
    FP = 0
    FN = 0

    # Also keep IoU/F1 like your earlier function (mean over batch, weighted by batch size)
    sum_iou = 0.0
    sum_f1 = 0.0
    n = 0

    thr = float(args.threshold)

    with torch.no_grad():
        for img, msk, _ in dl:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)

            with autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(img)

            # IoU/F1 from logits (uses thr=0.5 by default in your helper)
            # We want it consistent with chosen threshold -> call with thr=thr
            iou_b, f1_b = iou_f1_from_logits(logits, msk, thr=thr)
            bs = img.size(0)
            sum_iou += iou_b * bs
            sum_f1  += f1_b  * bs
            n += bs

            # Precision/Recall from TP/FP/FN (global pixel counts)
            probs = torch.sigmoid(logits)
            pred = (probs >= thr).int()
            gt   = (msk >= 0.5).int()

            TP += (pred & gt).sum().item()
            FP += (pred & (1 - gt)).sum().item()
            FN += ((1 - pred) & gt).sum().item()

    mean_iou = sum_iou / max(1, n)
    mean_f1  = sum_f1  / max(1, n)

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1_pr     = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"{args.split.upper()} IoU (mean): {mean_iou:.3f}")
    print(f"{args.split.upper()} F1  (mean): {mean_f1:.3f}")
    print(f"{args.split.upper()} Precision: {precision:.3f}")
    print(f"{args.split.upper()} Recall   : {recall:.3f}")
    print(f"{args.split.upper()} F1(PR)    : {f1_pr:.3f}")
    print(f"TP={TP:,} | FP={FP:,} | FN={FN:,}")

    return {
        "split": args.split,
        "threshold": thr,
        "iou": float(mean_iou),
        "f1": float(mean_f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(TP),
        "fp": int(FP),
        "fn": int(FN),
        "ckpt": str(ckpt_path),
    }


# ------------- CLI -------------
def parse_args():
    ap = argparse.ArgumentParser(description="U-Net pretraining + evaluation (buildings)")

    # Mode: train or eval
    ap.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Run mode: train or eval"
    )

    # Common / reproducibility
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=[1,2,3,4,5,6,7,8],
        help="1-based band indices to keep, e.g. --bands 1 2 3 4 5 6 7 8"
    )

    # DataLoader / perf
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    # Model
    ap.add_argument("--base_c", type=int, default=64)

    # Training hyperparams (only meaningful in mode=train)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    # Checkpoint/output (paths del proyecto, no del dataset)
    ap.add_argument(
        "--out_name",
        type=str,
        default="unet_8b_v1",
        help="Folder name under data/models/building_pretrain/"
    )

    # Evaluation params (only meaningful in mode=eval)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--ckpt", type=str, default="best.pth", help="Checkpoint filename or full path")
    ap.add_argument("--threshold", type=float, default=0.5)


    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run(args)        # tu training actual
    else:
        run_eval(args)   # nueva evaluación
