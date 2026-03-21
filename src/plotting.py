# plotting.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from config import DATA_PROCESSED


REQUIRED_COLS = ["thr", "iou", "f1", "precision", "recall"]

# --- FUNCIONES AUXILIARES ---

def _to_df(metrics: Union[pd.DataFrame, Sequence[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        df = metrics.copy()
    else:
        df = pd.DataFrame(list(metrics))

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("thr").reset_index(drop=True)
    return df

def _apply_plot_style(use_times: bool = True):
    """Aplica un estilo consistente para tesis (Times/Serif)."""
    if use_times:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
        })
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

def _save_figure(fig: plt.Figure, out_path: Union[str, Path], dpi: int = 300):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved figure: {out_path}")

def load_history(path):
    path = Path(path)
    h = np.load(path, allow_pickle=True)
    if isinstance(h, np.ndarray) and h.shape == () and hasattr(h, "item"):
        h = h.item()
    if not isinstance(h, dict):
        raise TypeError(f"history.npy no es dict. tipo={type(h)}")
    return h

def _best_epoch_from_val_iou(h):
    if "val_iou" not in h or len(h["val_iou"]) == 0:
        return None
    idx = int(np.nanargmax(np.asarray(h["val_iou"], dtype=float)))
    return h["epoch"][idx], float(h["val_iou"][idx])

def moving_avg(y, k=3):
    y = np.asarray(y, dtype=float)
    if k is None or k <= 1 or len(y) < k:
        return y
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    w = np.ones(k) / k
    return np.convolve(ypad, w, mode="valid")


# --- FUNCIONES DE PLOTEO ---

def plot_val_curves(history, title="", smooth_k=3, save_svg=None, show_loss=True):
    """
    MODIFICADA: Genera las curvas de validación en un solo ploteo (1x3).
    """
    _apply_plot_style()
    epochs = np.asarray(history["epoch"], dtype=int)
    
    # Best epoch por IoU original (sin suavizar)
    raw_val_iou = np.asarray(history["val_iou"], dtype=float)
    best_idx = int(np.nanargmax(raw_val_iou))
    best_epoch = int(history["epoch"][best_idx])
    best_val = float(raw_val_iou[best_idx])
    last_epoch = int(epochs[-1])

    # Configuración de subplots
    n_cols = 3 if show_loss else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 4), constrained_layout=True)
    
    # (a) Val IoU
    ax = axes[0]
    val_iou_s = moving_avg(history["val_iou"], k=smooth_k)
    ax.plot(epochs, val_iou_s, label="Val IoU")
    ax.axvline(best_epoch, linestyle="--", color="red", label=f"Best @ {best_epoch} ({best_val:.3f})")
    ax.axvline(last_epoch, linestyle=":", color="red", label=f"Stopped @ {last_epoch}")
    ax.set_title("(a) Validation IoU")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.grid(True); ax.legend()

    # (b) Val F1
    ax = axes[1]
    val_f1_s = moving_avg(history["val_f1"], k=smooth_k)
    ax.plot(epochs, val_f1_s, label="Val F1", color="orange")
    ax.axvline(best_epoch, linestyle="--", color="red")
    ax.axvline(last_epoch, linestyle=":", color="red")
    ax.set_title("(b) Validation F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.grid(True); ax.legend()

    # (c) Val Loss
    if show_loss:
        ax = axes[2]
        val_loss_s = moving_avg(history["val_loss"], k=smooth_k)
        ax.plot(epochs, val_loss_s, label="Val Loss", color="green")
        ax.axvline(best_epoch, linestyle="--", color="red")
        ax.axvline(last_epoch, linestyle=":", color="red")
        ax.set_title("(c) Validation Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True); ax.legend()

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_svg is not None:
        outdir = Path(save_svg); outdir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, outdir / "validation_curves_summary.svg")

    plt.show()

def plot_training_curves(history, title="", save_svg=None):
    """
    MODIFICADA: Genera las curvas de entrenamiento (Train vs Val) en un solo ploteo (1x3).
    """
    _apply_plot_style()
    epochs = np.asarray(history["epoch"], dtype=int)
    best = _best_epoch_from_val_iou(history)
    best_epoch = best[0] if best else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    metrics = [("loss", "Loss"), ("iou", "IoU"), ("f1", "F1")]
    letters = ["(a)", "(b)", "(c)"]

    for i, (m_key, m_name) in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, history[f"train_{m_key}"], label="Train")
        ax.plot(epochs, history[f"val_{m_key}"], label="Validation")
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="red", alpha=0.6, label="Best Val IoU")
        ax.set_title(f"{letters[i]} {m_name} vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(m_name)
        ax.grid(True)
        ax.legend()

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_svg is not None:
        outdir = Path(save_svg); outdir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, outdir / "training_curves_summary.svg")

    plt.show()

def plot_threshold_analysis(
    metrics: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    *,
    show_best_iou: bool = True,
    show_best_f1: bool = True,
    show_pr_eq: bool = True,
    title: str = "",
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    use_times: bool = True,
):
    """Mantenida igual que el original (ya era 1x3)."""
    _apply_plot_style(use_times=use_times)
    df = _to_df(metrics)

    best_iou_thr = float(df.loc[df["iou"].idxmax(), "thr"]) if show_best_iou else None
    best_f1_thr  = float(df.loc[df["f1"].idxmax(), "thr"]) if show_best_f1 else None
    pr_eq_thr = float(df.loc[(df["precision"] - df["recall"]).abs().idxmin(), "thr"]) if show_pr_eq else None

    def _vline(ax, x, label, linestyle="--"):
        ax.axvline(x, linestyle=linestyle, linewidth=1.2, color="red", label=label)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), constrained_layout=True)

    # (a)
    ax = axes[0]
    ax.plot(df["thr"], df["f1"], marker="x", label="F1")
    ax.plot(df["thr"], df["iou"], marker="x", label="IoU")
    if best_f1_thr: _vline(ax, best_f1_thr, f"Best F1 ({best_f1_thr:.2f})")
    if best_iou_thr: _vline(ax, best_iou_thr, f"Best IoU ({best_iou_thr:.2f})", linestyle=":")
    ax.set_title("(a) IoU and F1 vs Threshold"); ax.grid(True); ax.legend()

    # (b)
    ax = axes[1]
    ax.plot(df["thr"], df["precision"], label="Precision")
    ax.plot(df["thr"], df["recall"], label="Recall")
    if pr_eq_thr: _vline(ax, pr_eq_thr, f"P≈R ({pr_eq_thr:.2f})")
    ax.set_title("(b) Precision and Recall vs Threshold"); ax.grid(True); ax.legend()

    # (c)
    ax = axes[2]
    ax.plot(df["recall"], df["precision"], label="PR curve")
    if pr_eq_thr:
        row = df.loc[df["thr"].sub(pr_eq_thr).abs().idxmin()]
        ax.scatter([row["recall"]], [row["precision"]], color="red", label=f"P≈R ({pr_eq_thr:.2f})")
    ax.set_title("(c) Precision–Recall Curve"); ax.grid(True); ax.legend()

    if title: fig.suptitle(title)
    if out_path: _save_figure(fig, out_path, dpi=dpi)
    plt.show()
    return fig

def plot_compare(hist_a, hist_b, key="val_iou", label_a="Model A", label_b="Model B", title="", save_svg=None):
    """Mantenida igual."""
    _apply_plot_style()
    e1, y1 = np.asarray(hist_a["epoch"]), np.asarray(hist_a[key])
    e2, y2 = np.asarray(hist_b["epoch"]), np.asarray(hist_b[key])

    plt.figure(figsize=(7,4))
    plt.plot(e1, y1, label=label_a)
    plt.plot(e2, y2, label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel(key.replace("_", " ").upper())
    plt.title(title or f"{key} comparison")
    plt.grid(True); plt.legend()

    if save_svg is not None:
        outdir = Path(save_svg); outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / f"compare_{key}.svg", format="svg", bbox_inches="tight")
    plt.show()


def _extract_meta(rest):
    """
    Tries to recover a metadata dict from the extra outputs of the dataset.
    Works even if meta is nested or partially provided.
    Returns (meta_dict, order, year_month) where any can be None.
    """
    meta = None

    # Case A: dataset returns (img, msk, meta)
    if len(rest) == 1 and isinstance(rest[0], dict):
        meta = rest[0]

    # Case B: dataset returns (..., meta_dict) at the end
    if meta is None:
        for x in reversed(rest):
            if isinstance(x, dict):
                meta = x
                break

    order = None
    year_month = None

    # If we found a dict, try common keys
    if isinstance(meta, dict):
        for k in ["order", "site_order", "site_id"]:
            if k in meta:
                order = meta[k]
                break
        for k in ["year_month", "ym", "date", "month"]:
            if k in meta:
                year_month = meta[k]
                break

    # Case C: dataset returns order/year_month as separate scalars/strings
    # (we only try this if meta dict didn't provide them)
    if order is None or year_month is None:
        for x in rest:
            # simple heuristics
            if order is None and isinstance(x, (int, np.integer, str)):
                # don't overwrite if it's clearly year_month like "2023-05"
                if not (isinstance(x, str) and ("-" in x or "_" in x) and len(x) <= 10):
                    order = x
            if year_month is None and isinstance(x, str):
                if ("-" in x or "_" in x) and len(x) <= 10:
                    year_month = x

    return meta, order, year_month

@torch.no_grad()
def plot_one_from_loader(
    trainer,
    *,
    split="test",
    idx=0,
    thr=0.7,
    make_rgb_like=None,
    save_subdir=None,          # NUEVO
    dpi=600
):
    dl = trainer.dl_te if split == "test" else trainer.dl_va
    ds = dl.dataset

    img, msk, *rest = ds[idx]
    meta, order, year_month = _extract_meta(rest)

    device = trainer.device
    use_amp = (trainer.device_type == "cuda")

    img_b = img.unsqueeze(0).to(device)
    msk_b = msk.unsqueeze(0).to(device)

    trainer.model.eval()

    with torch.amp.autocast(device_type=trainer.device_type, enabled=use_amp):
        logits = trainer.model(img_b)

    prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
    pred = (prob >= thr).astype(np.uint8)
    gt   = (msk[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)

    rgb = make_rgb_like(img) if make_rgb_like else img[0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    ax[0].imshow(rgb)
    ax[0].set_title(f"RGB\norder={order}, month={year_month}")
    ax[0].axis("off")

    ax[1].imshow(rgb)
    ax[1].imshow(gt, alpha=0.3)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(rgb)
    ax[2].imshow(pred, alpha=0.3)
    ax[2].set_title(f"Prediction (thr={thr})")
    ax[2].axis("off")

    plt.tight_layout()

    # --- SAVE BLOCK ---
    if save_subdir is not None:
        outdir = Path(DATA_PROCESSED) / "plot_results" / save_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        filename = f"{split}_idx{idx}_order{order}_{year_month}.png"
        out_path = outdir / filename

        fig.savefig(out_path, format="png", bbox_inches="tight", dpi=dpi)
        print(f"Saved: {out_path}")

    plt.show()



@torch.no_grad()
def metrics_one_tile_from_logits(logits, targets, thr=0.7, eps=1e-8):
    """
    logits: (1,1,H,W)
    targets: (1,1,H,W) float {0,1}
    returns dict with TP/FP/FN/TN and IoU/F1/Prec/Rec
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).int()
    gt   = (targets >= 0.5).int()

    TP = int((pred & gt).sum().item())
    FP = int((pred & (1 - gt)).sum().item())
    FN = int(((1 - pred) & gt).sum().item())
    TN = int(((1 - pred) & (1 - gt)).sum().item())

    iou = (TP + eps) / (TP + FP + FN + eps)
    f1  = (2*TP + eps) / (2*TP + FP + FN + eps)
    prec = (TP + eps) / (TP + FP + eps)
    rec  = (TP + eps) / (TP + FN + eps)

    gt_px   = TP + FN
    pred_px = TP + FP

    return {
        "tp": TP, "fp": FP, "fn": FN, "tn": TN,
        "iou": float(iou), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "gt_px": int(gt_px), "pred_px": int(pred_px),
    }

@torch.no_grad()
def scan_split_table(trainer, *, split="test", thr=0.7, max_items=None):
    dl = trainer.dl_te if split == "test" else trainer.dl_va
    assert dl is not None, f"No dataloader for split={split}"
    ds = dl.dataset

    trainer.model.eval()
    device = trainer.device
    use_amp = (trainer.device_type == "cuda")

    rows = []
    n = len(ds) if max_items is None else min(len(ds), int(max_items))

    for idx in range(n):
        img, msk, *rest = ds[idx]
        meta, order, year_month = _extract_meta(rest)

        img_b = img.unsqueeze(0).to(device)
        msk_b = msk.unsqueeze(0).to(device)

        with torch.amp.autocast(device_type=trainer.device_type, enabled=use_amp):
            logits = trainer.model(img_b)

        m = metrics_one_tile_from_logits(logits, msk_b, thr=thr)
        m.update({
            "idx": int(idx),
            "order": order,
            "year_month": year_month,
        })
        rows.append(m)

    df = pd.DataFrame(rows)
    return df


def select_good_bad_ugly(df, *, k=6, unique_by="order"):
    df2 = df.copy()

    # Evitar repetir site/order
    def _pick_unique(sorted_df):
        picked = []
        seen = set()
        for _, r in sorted_df.iterrows():
            key = r.get(unique_by, None)
            if key in seen:
                continue
            seen.add(key)
            picked.append(r)
            if len(picked) >= k:
                break
        return pd.DataFrame(picked)

    # GOOD: IoU alto y fp_rate bajo
    df2["fp_rate"] = df2["fp"] / (df2["pred_px"] + 1e-8)
    good = _pick_unique(df2.sort_values(["iou","fp_rate"], ascending=[False, True]))

    # BAD: IoU bajo pero con GT positivo (para que sea un fallo “real”)
    bad_pool = df2[df2["gt_px"] > 0]
    bad = _pick_unique(bad_pool.sort_values(["iou","fn"], ascending=[True, False]))

    # UGLY: FP altísimo (falsas alarmas) o FN altísimo (se come la obra)
    ugly_fp = _pick_unique(df2.sort_values(["fp","iou"], ascending=[False, True]).head(200))
    ugly_fn = _pick_unique(df2.sort_values(["fn","iou"], ascending=[False, True]).head(200))

    # mezcla: prioriza FP, luego completa con FN si falta
    ugly = pd.concat([ugly_fp, ugly_fn], ignore_index=True)
    ugly = _pick_unique(ugly)

    return good, bad, ugly


# ---------------------------------------------------------------------------
# Funciones migradas desde notebook 04_finetuning
# ---------------------------------------------------------------------------

def make_rgb_like(img8):
    """(8,H,W) torch tensor → (H,W,3) numpy RGB normalizado [0-1]. Bandas 5,3,1."""
    x = img8[[5, 3, 1], :, :].float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x.permute(1, 2, 0).detach().cpu().numpy()


@torch.no_grad()
def show_prediction_with_meta(trainer, batch, thr=0.5, idx_in_batch=0):
    """4-panel (RGB / GT / Prob / Pred) para un sample de un batch ya cargado."""
    trainer.model.eval()

    img, msk, meta = batch
    img = img.to(trainer.device)
    msk = msk.to(trainer.device)

    with torch.no_grad():
        probs = torch.sigmoid(trainer.model(img))
        pred = (probs >= thr).float()

    i = idx_in_batch
    order_i = int(meta["order"][i].item())
    ym_i = meta["year_month"][i]

    x = img[i].cpu().numpy()
    y_gt = msk[i, 0].cpu().numpy()
    y_pr = pred[i, 0].cpu().numpy()
    y_pb = probs[i, 0].cpu().numpy()

    R, G, B = 5, 3, 1
    rgb = x[[R, G, B]].transpose(1, 2, 0)
    rgb = rgb / max(rgb.max(), 1e-6)
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(rgb); axes[0].set_title(f"RGB\norder={order_i}\n{ym_i}"); axes[0].axis("off")
    axes[1].imshow(y_gt, cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow(y_pb, cmap="viridis"); axes[2].set_title("Prob"); axes[2].axis("off")
    axes[3].imshow(y_pr, cmap="gray"); axes[3].set_title(f"Pred thr={thr:.2f}"); axes[3].axis("off")
    plt.tight_layout(); plt.show()


@torch.no_grad()
def scan_split_table_batched(trainer, thr=0.7, max_batches=None):
    """
    Versión en batch de scan_split_table(): más rápida, usa AMP.
    Devuelve DataFrame con columnas: order, year_month, tp, fp, fn,
    iou, f1, precision, recall, gt_px, pred_px.
    """
    trainer.model.eval()
    dl = trainer.dl_va

    rows = []
    device = trainer.device
    use_amp = (trainer.device_type == "cuda")

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        img, msk = batch[0], batch[1]
        meta = batch[2] if len(batch) > 2 and isinstance(batch[2], dict) else None
        if meta is None:
            raise ValueError("No encontré meta dict en el batch. Ajusta el índice (batch[2]).")

        img = img.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=trainer.device_type, enabled=use_amp):
            logits = trainer.model(img)

        probs = torch.sigmoid(logits)
        pred = (probs >= thr).to(torch.int32)
        gt   = (msk >= 0.5).to(torch.int32)

        dims = (2, 3)
        TP = (pred & gt).sum(dim=dims).cpu().numpy()
        FP = (pred & (1 - gt)).sum(dim=dims).cpu().numpy()
        FN = ((1 - pred) & gt).sum(dim=dims).cpu().numpy()

        inter = TP.astype(np.float64)
        union = (pred.sum(dim=dims).cpu().numpy() + gt.sum(dim=dims).cpu().numpy() - TP).astype(np.float64)

        eps = 1e-8
        iou = (inter + eps) / (union + eps)
        f1  = (2*inter + eps) / (2*inter + FP + FN + eps)
        precision = (inter + eps) / (inter + FP + eps)
        recall    = (inter + eps) / (inter + FN + eps)

        gt_px   = gt.sum(dim=dims).cpu().numpy()
        pred_px = pred.sum(dim=dims).cpu().numpy()

        orders = meta["order"].detach().cpu().numpy()
        yms    = meta["year_month"]

        for i in range(len(orders)):
            rows.append({
                "order": int(orders[i]),
                "year_month": str(yms[i]),
                "tp": int(TP[i].item()),
                "fp": int(FP[i].item()),
                "fn": int(FN[i].item()),
                "iou": float(iou[i].item()),
                "f1": float(f1[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "gt_px": int(gt_px[i].item()),
                "pred_px": int(pred_px[i].item()),
            })

    return pd.DataFrame(rows)


def pick_sets(df, *, n=6):
    """
    Estratifica un DataFrame de comparación dual-modelo (columnas _A/_B, d_iou)
    en 5 tiers: good, bad, ugly, A_wins, B_wins.
    Complementa select_good_bad_ugly() (single-model).
    """
    p50_lo = df["iou_A"].quantile(0.45)
    p50_hi = df["iou_A"].quantile(0.55)

    df_nonempty = df[df["gt_px_A"] > 0].copy()
    df_nonempty["iou_best"] = df_nonempty[["iou_A", "iou_B"]].max(axis=1)

    good = df_nonempty.sort_values("iou_best", ascending=False).head(n)

    mid = df_nonempty[
        (df_nonempty["iou_best"] >= p50_lo) & (df_nonempty["iou_best"] <= p50_hi)
    ].copy()
    mid["err"] = mid["fp_A"] + mid["fn_A"] + mid["fp_B"] + mid["fn_B"]
    bad = mid.sort_values("err", ascending=False).head(n)

    ugly = df_nonempty.sort_values("iou_best", ascending=True).head(n)

    winA = df_nonempty.sort_values("d_iou", ascending=False).head(n)
    winB = df_nonempty.sort_values("d_iou", ascending=True).head(n)

    return {"good": good, "bad": bad, "ugly": ugly, "A_wins": winA, "B_wins": winB}


@torch.no_grad()
def find_sample_in_val(trainer, order, year_month):
    """Busca linealmente en dl_va el sample (order, year_month). Devuelve (img, msk, meta)."""
    for batch in trainer.dl_va:
        img, msk = batch[0], batch[1]
        meta = batch[2]
        orders = meta["order"].cpu().numpy()
        yms = meta["year_month"]
        for i in range(len(orders)):
            if int(orders[i]) == int(order) and str(yms[i]) == str(year_month):
                return img[i], msk[i], {"order": int(orders[i]), "year_month": str(yms[i])}
    raise ValueError(f"No encontré (order={order}, year_month={year_month}) en val.")


@torch.no_grad()
def plot_compare_one(order, year_month, *, trainerA, trainerB, thr=0.7,
                     label_A="Model A", label_B="Model B"):
    """Compara las predicciones de dos modelos sobre un tile específico (4 paneles)."""
    img, msk, meta = find_sample_in_val(trainerA, order, year_month)
    device = trainerA.device
    use_amp = (trainerA.device_type == "cuda")

    img_b = img.unsqueeze(0).to(device)

    trainerA.model.eval(); trainerB.model.eval()

    with torch.amp.autocast(device_type=trainerA.device_type, enabled=use_amp):
        logA = trainerA.model(img_b)
        logB = trainerB.model(img_b)

    probA = torch.sigmoid(logA)[0, 0].detach().cpu().numpy()
    probB = torch.sigmoid(logB)[0, 0].detach().cpu().numpy()
    gt    = (msk[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)

    predA = (probA >= thr).astype(np.uint8)
    predB = (probB >= thr).astype(np.uint8)

    rgb = make_rgb_like(img)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(rgb); ax[0].set_title(f"Tile RGB\norder={meta['order']} Date: {meta['year_month']}"); ax[0].axis("off")
    ax[1].imshow(rgb); ax[1].imshow(gt, alpha=0.3); ax[1].set_title("GT (overlay)"); ax[1].axis("off")
    ax[2].imshow(rgb); ax[2].imshow(predA, alpha=0.3); ax[2].set_title(f"{label_A} pred (thr={thr})"); ax[2].axis("off")
    ax[3].imshow(rgb); ax[3].imshow(predB, alpha=0.3); ax[3].set_title(f"{label_B} pred (thr={thr})"); ax[3].axis("off")
    plt.tight_layout()
    plt.show()


def quick_delta_summary(df):
    """Devuelve DataFrame de estadísticas (mean/median/p10/p90) de d_iou, d_f1, d_fp, d_fn."""
    out = {}
    for col in ["d_iou", "d_f1", "d_fp", "d_fn"]:
        out[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "p10": float(df[col].quantile(0.10)),
            "p90": float(df[col].quantile(0.90)),
        }
    return pd.DataFrame(out).T
