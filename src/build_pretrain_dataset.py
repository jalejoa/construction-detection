"""
build_pretrain_dataset.py
=========================
Full pipeline to prepare the pretraining dataset for the U-Net building detector.

Pipeline stages:
  1. build_mosaic   — Reprojects and mosaics Planet quads into one GeoTIFF + grid JSON
  2. rasterize_mask — Burns building footprints (GPKG) into a binary mask aligned to the grid
  3. tile_dataset   — Sliding-window tiling (default 512×512), keeps tiles with ≥ pos_min positives
  4. make_splits    — Random 70/15/15 train/val/test split by tile ID
  5. compute_norm   — Per-band mean/std from train tiles → JSON

Expected input layout (defaults):
    data/raw/planet_quads/   ← Planet .tif quads (8-band + optional alpha)
    data/raw/alkis/gebaude.gpkg

Output layout:
    data/pretraining/
        mosaic.tif
        mosaic.grid.json
        building_mask.tif
        tiles/
            images/
            masks/
            splits/
                train.txt  val.txt  test.txt
        stats/
            norm_stats_building_8b.json

Usage (CLI):
    cd src
    python build_pretrain_dataset.py
    python build_pretrain_dataset.py --quads_dir ../data/raw/planet_quads --epsg 25832

Usage (Python):
    from build_pretrain_dataset import run_pipeline
    run_pipeline(quads_dir=..., footprints_path=..., out_dir=...)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

from config import DATA_RAW, DATA_DIR


# ---------------------------------------------------------------------------
# Stage 1 — Mosaic
# ---------------------------------------------------------------------------

def build_mosaic(
    quads_dir: Path,
    out_tif: Path,
    out_grid_json: Path,
    *,
    epsg: int = 25832,
    res: float = 3.0,
    bands: List[int],
) -> dict:
    """
    Reprojects and mosaics all .tif quads in quads_dir into a single GeoTIFF.
    Also saves a grid JSON used by subsequent stages.

    Parameters
    ----------
    quads_dir : Path
        Directory containing Planet .tif quad files.
    out_tif : Path
        Output mosaic GeoTIFF path.
    out_grid_json : Path
        Output grid JSON path.
    epsg : int
        Target CRS as EPSG code.
    res : float
        Target pixel resolution in metres.
    bands : list[int]
        1-based band indices to include (e.g. [1,2,3,4,5,6,7,8] skips alpha).

    Returns
    -------
    dict
        Grid dictionary (epsg, transform, width, height, bounds, res).
    """
    paths = sorted(quads_dir.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in: {quads_dir}")

    print(f"[mosaic] Found {len(paths)} quads in {quads_dir}")
    band_idx = [b - 1 for b in bands]  # 0-based

    dst_crs = rasterio.crs.CRS.from_epsg(epsg)

    # Compute global bounds in target CRS
    all_bounds = []
    for p in paths:
        with rasterio.open(p) as src:
            left, bottom, right, top = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
            all_bounds.append((left, bottom, right, top))

    g_left   = min(b[0] for b in all_bounds)
    g_bottom = min(b[1] for b in all_bounds)
    g_right  = max(b[2] for b in all_bounds)
    g_top    = max(b[3] for b in all_bounds)

    width  = int(np.ceil((g_right  - g_left)   / res))
    height = int(np.ceil((g_top    - g_bottom)  / res))
    transform = from_origin(g_left, g_top, res, res)

    n_bands = len(band_idx)
    with rasterio.open(paths[0]) as ref:
        dtype = ref.dtypes[0]

    dest = np.zeros((n_bands, height, width), dtype=dtype)

    for p in paths:
        with rasterio.open(p) as src:
            tmp = np.zeros((n_bands, height, width), dtype=dtype)
            for out_b, src_b in enumerate(band_idx):
                reproject(
                    source=rasterio.band(src, src_b + 1),
                    destination=tmp[out_b],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.bilinear,
                )
            mask = tmp != 0
            dest = np.where(mask, tmp, dest)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": n_bands,
        "dtype": dtype,
        "crs": dst_crs,
        "transform": transform,
        "nodata": 0,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
    }

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(dest)

    grid = {
        "epsg": epsg,
        "transform": list(transform),
        "width": width,
        "height": height,
        "bounds": [g_left, g_bottom, g_right, g_top],
        "res": res,
        "template_raster": str(out_tif),
    }
    out_grid_json.parent.mkdir(parents=True, exist_ok=True)
    out_grid_json.write_text(json.dumps(grid, indent=2), encoding="utf-8")

    print(f"[mosaic] Written: {out_tif}  ({width}×{height} px, {n_bands} bands)")
    print(f"[mosaic] Grid JSON: {out_grid_json}")
    return grid


# ---------------------------------------------------------------------------
# Stage 2 — Rasterize mask
# ---------------------------------------------------------------------------

def rasterize_mask(
    footprints_path: Path,
    grid: dict,
    out_mask: Path,
    *,
    layer: Optional[str] = None,
) -> Path:
    """
    Rasterizes building footprints to a binary uint8 mask aligned to grid.

    Parameters
    ----------
    footprints_path : Path
        Vector file with building polygons (GPKG, SHP, GeoJSON).
    grid : dict
        Grid dictionary as returned by build_mosaic.
    out_mask : Path
        Output mask GeoTIFF path.
    layer : str | None
        Layer name if the GPKG has multiple layers.

    Returns
    -------
    Path
        Path to the written mask.
    """
    dst_crs   = rasterio.crs.CRS.from_epsg(int(grid["epsg"]))
    dst_tr    = Affine(*grid["transform"])
    dst_w     = int(grid["width"])
    dst_h     = int(grid["height"])

    gdf = gpd.read_file(footprints_path, layer=layer) if layer else gpd.read_file(footprints_path)
    print(f"[mask] {len(gdf)} building footprints loaded")

    if gdf.crs is None:
        raise ValueError("Footprints file has no CRS defined.")
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)

    shapes = (
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    )

    mask_arr = rasterize(
        shapes=shapes,
        out_shape=(dst_h, dst_w),
        transform=dst_tr,
        all_touched=False,
        fill=0,
        dtype="uint8",
    )

    profile = {
        "driver": "GTiff",
        "height": dst_h,
        "width":  dst_w,
        "count":  1,
        "dtype":  "uint8",
        "crs":    dst_crs,
        "transform": dst_tr,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
    }

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(mask_arr, 1)

    pos_pct = 100.0 * mask_arr.sum() / mask_arr.size
    print(f"[mask] Written: {out_mask}  ({pos_pct:.2f}% positive pixels)")
    return out_mask


# ---------------------------------------------------------------------------
# Stage 3 — Tile dataset
# ---------------------------------------------------------------------------

def _iter_windows(width: int, height: int, ts: int, stride: int):
    max_top  = height - ts
    max_left = width  - ts
    if max_top < 0 or max_left < 0:
        return
    for top in range(0, max_top + 1, stride):
        for left in range(0, max_left + 1, stride):
            yield Window(left, top, ts, ts)


def tile_dataset(
    img_path: Path,
    msk_path: Path,
    out_images_dir: Path,
    out_masks_dir: Path,
    *,
    tile_size: int = 512,
    stride: int = 512,
    pos_min: float = 0.01,
    id_prefix: str = "tile",
) -> List[str]:
    """
    Slides a window over the mosaic+mask pair and writes tiles to disk.
    Only tiles with >= pos_min fraction of positive pixels are kept.

    Returns
    -------
    list[str]
        Tile IDs that were written.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    tile_ids: List[str] = []
    total = kept = 0

    with rasterio.open(img_path) as si, rasterio.open(msk_path) as sm:
        if (si.width, si.height) != (sm.width, sm.height):
            raise ValueError("Image and mask dimensions do not match.")
        if si.crs != sm.crs:
            raise ValueError("Image and mask CRS do not match.")

        img_dtype = si.dtypes[0]

        for win in _iter_windows(si.width, si.height, tile_size, stride):
            total += 1

            m = sm.read(1, window=win)
            m_bin = (m > 0).astype(np.uint8)
            if float(m_bin.sum()) / (tile_size * tile_size) < pos_min:
                continue

            I = si.read(window=win)

            top  = int(win.row_off)
            left = int(win.col_off)
            tid  = f"{id_prefix}_{top}_{left}"

            # Image tile
            tr_img = rasterio.windows.transform(win, si.transform)
            prof_img = {
                "driver": "GTiff",
                "height": tile_size,
                "width":  tile_size,
                "count":  si.count,
                "dtype":  img_dtype,
                "crs":    si.crs,
                "transform": tr_img,
                "tiled": True,
                "blockxsize": min(256, tile_size),
                "blockysize": min(256, tile_size),
                "compress": "DEFLATE",
                "predictor": 2,
                "nodata": 0,
            }
            with rasterio.open(out_images_dir / f"{tid}.tif", "w", **prof_img) as dst:
                dst.write(I)

            # Mask tile
            tr_msk = rasterio.windows.transform(win, sm.transform)
            prof_msk = {
                "driver": "GTiff",
                "height": tile_size,
                "width":  tile_size,
                "count":  1,
                "dtype":  "uint8",
                "crs":    sm.crs,
                "transform": tr_msk,
                "tiled": True,
                "blockxsize": min(256, tile_size),
                "blockysize": min(256, tile_size),
                "compress": "DEFLATE",
                "predictor": 2,
            }
            with rasterio.open(out_masks_dir / f"{tid}_mask.tif", "w", **prof_msk) as dst:
                dst.write(m_bin, 1)

            tile_ids.append(tid)
            kept += 1

    print(f"[tiles] Kept {kept} / {total} windows (pos_min={pos_min})")
    return tile_ids


# ---------------------------------------------------------------------------
# Stage 4 — Make splits
# ---------------------------------------------------------------------------

def make_splits(
    tile_ids: List[str],
    out_splits_dir: Path,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Randomly splits tile IDs into train/val/test and writes .txt files.

    Returns
    -------
    dict
        {"train": [...], "val": [...], "test": [...]}
    """
    ids = list(tile_ids)
    random.seed(seed)
    random.shuffle(ids)

    n     = len(ids)
    n_tr  = int(train_frac * n)
    n_va  = int(val_frac   * n)

    splits = {
        "train": ids[:n_tr],
        "val":   ids[n_tr : n_tr + n_va],
        "test":  ids[n_tr + n_va :],
    }

    out_splits_dir.mkdir(parents=True, exist_ok=True)
    for split, split_ids in splits.items():
        (out_splits_dir / f"{split}.txt").write_text(
            "\n".join(split_ids), encoding="utf-8"
        )

    print(f"[splits] {' | '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
    return splits


# ---------------------------------------------------------------------------
# Stage 5 — Compute normalization stats
# ---------------------------------------------------------------------------

def compute_norm(
    images_dir: Path,
    train_ids: List[str],
    out_json: Path,
) -> dict:
    """
    Computes per-band mean and std from train tiles using two-pass algorithm.

    Returns
    -------
    dict
        {"mean": [...], "std": [...]}
    """
    means = []
    sqs   = []
    n_bands = None

    for tid in train_ids:
        with rasterio.open(images_dir / f"{tid}.tif") as src:
            arr = src.read().astype(np.float32)  # [C, H, W]
        if n_bands is None:
            n_bands = arr.shape[0]
        x = arr.reshape(n_bands, -1)
        means.append(x.mean(axis=1))
        sqs.append((x ** 2).mean(axis=1))

    m = np.mean(np.stack(means), axis=0)
    v = np.mean(np.stack(sqs),   axis=0) - m ** 2
    s = np.sqrt(np.maximum(v, 1e-8))

    stats = {"mean": m.tolist(), "std": s.tolist()}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[norm] Stats for {n_bands} bands saved: {out_json}")
    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    quads_dir: Path,
    footprints_path: Path,
    out_dir: Path,
    *,
    epsg: int = 25832,
    res: float = 3.0,
    bands: List[int] = None,
    tile_size: int = 512,
    stride: int = 512,
    pos_min: float = 0.01,
    id_prefix: str = "tile",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    stats_name: str = "norm_stats_building_8b.json",
) -> None:
    if bands is None:
        bands = list(range(1, 9))  # bands 1-8, skip alpha (band 9)

    tiles_dir  = out_dir / "tiles"
    images_dir = tiles_dir / "images"
    masks_dir  = tiles_dir / "masks"
    splits_dir = tiles_dir / "splits"
    stats_dir  = out_dir / "stats"

    # 1. Mosaic
    grid = build_mosaic(
        quads_dir=quads_dir,
        out_tif=out_dir / "mosaic.tif",
        out_grid_json=out_dir / "mosaic.grid.json",
        epsg=epsg,
        res=res,
        bands=bands,
    )

    # 2. Rasterize mask
    rasterize_mask(
        footprints_path=footprints_path,
        grid=grid,
        out_mask=out_dir / "building_mask.tif",
    )

    # 3. Tile
    tile_ids = tile_dataset(
        img_path=out_dir / "mosaic.tif",
        msk_path=out_dir / "building_mask.tif",
        out_images_dir=images_dir,
        out_masks_dir=masks_dir,
        tile_size=tile_size,
        stride=stride,
        pos_min=pos_min,
        id_prefix=id_prefix,
    )

    # 4. Splits
    splits = make_splits(
        tile_ids=tile_ids,
        out_splits_dir=splits_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    # 5. Norm stats
    compute_norm(
        images_dir=images_dir,
        train_ids=splits["train"],
        out_json=stats_dir / stats_name,
    )

    print(f"\n[done] Pretraining dataset ready at: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build pretraining dataset for U-Net building detector."
    )
    ap.add_argument(
        "--quads_dir",
        type=Path,
        default=DATA_RAW / "planet_quads",
        help="Directory with Planet .tif quads (default: data/raw/planet_quads/)",
    )
    ap.add_argument(
        "--footprints",
        type=Path,
        default=DATA_RAW / "alkis" / "gebaude.gpkg",
        help="Building footprints vector file (default: data/raw/alkis/gebaude.gpkg)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DATA_DIR / "pretraining",
        help="Output root directory (default: data/pretraining/)",
    )
    ap.add_argument("--epsg",      type=int,   default=25832)
    ap.add_argument("--res",       type=float, default=3.0,   help="Pixel resolution in metres")
    ap.add_argument("--bands",     type=int,   nargs="+",     default=list(range(1, 9)),
                    help="1-based band indices to use (default: 1 2 3 4 5 6 7 8)")
    ap.add_argument("--tile_size", type=int,   default=512)
    ap.add_argument("--stride",    type=int,   default=512)
    ap.add_argument("--pos_min",   type=float, default=0.01,
                    help="Min positive pixel fraction to keep a tile")
    ap.add_argument("--id_prefix", type=str,   default="tile")
    ap.add_argument("--train_frac",type=float, default=0.70)
    ap.add_argument("--val_frac",  type=float, default=0.15)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--stats_name",type=str,   default="norm_stats_building_8b.json")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        quads_dir=args.quads_dir,
        footprints_path=args.footprints,
        out_dir=args.out_dir,
        epsg=args.epsg,
        res=args.res,
        bands=args.bands,
        tile_size=args.tile_size,
        stride=args.stride,
        pos_min=args.pos_min,
        id_prefix=args.id_prefix,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        stats_name=args.stats_name,
    )
