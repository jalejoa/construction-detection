from __future__ import annotations
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
import geopandas as gpd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from rasterio.features import rasterize
import json


def build_macro_index(
    point_quad: gpd.GeoDataFrame,
    quad_catalog: gpd.GeoDataFrame,
    id_col: str = "order",
) -> gpd.GeoDataFrame:
    """
    Build an index of macro patches to generate.
    One row per construction site and month, with the matching quad path.
    """
    # Keep only needed columns on each side
    pq = point_quad[[id_col, "mosaic_id", "quad_id", "year_month", "geometry"]].copy()
    qc = quad_catalog[["mosaic_id", "quad_id", "year_month", "path"]].copy()

    # Merge to attach quad path to each site-month
    merged = pq.merge(
        qc,
        on=["mosaic_id", "quad_id", "year_month"],
        how="inner"
    )

    # Rename path column for clarity
    merged = merged.rename(columns={"path": "quad_path"})

    # Ensure GeoDataFrame with point geometry
    macro_index = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=point_quad.crs
    )

    return macro_index



def generate_macro_patch_for_site(
    row,
    macro_size_m: float,
    images_dir: str | Path,
    overwrite: bool = False,
) -> dict:
    """
    Generate a macro patch (image only) for a single construction site.
    The patch is a square of macro_size_m x macro_size_m (in meters),
    fully contained within the quad and guaranteed to include the site.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    quad_path = Path(row["quad_path"])
    point_geom = row["geometry"]

    # --- FIX: if geometry is MultiPoint → take first Point ---
    if point_geom.geom_type == "MultiPoint":
        point_geom = point_geom.geoms[0]

    order = row["order"]
    year_month = str(row["year_month"])
    ym_str = ym_dash_to_underscore(year_month)
    quad_id = row["quad_id"]

    # Folder per site
    site_dir = Path(images_dir) / str(order)
    site_dir.mkdir(parents=True, exist_ok=True)

    # Filename: cs_{order}_{YYYY_MM}_{quad_id}.tif
    img_name = f"cs_{order}_{ym_str}.tif"
    img_path = site_dir / img_name

    ###ACTIVATE THIS TO DEBUG CORRUPTED IMAGES
    #print(f"[MacroPatch] order={order} | ym={ym_str} | quad={quad_id}")

    if img_path.exists() and not overwrite:
        return {
            "order": order,
            "year_month": year_month,
            "quad_path": str(quad_path),
            "image_path": str(img_path),
            "status": "skipped",
        }

    with rasterio.open(quad_path) as src:
        # Reproject site point to the quad CRS (quads are in EPSG:3857)
        pt = gpd.GeoSeries([point_geom], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        cx, cy = pt.x, pt.y

        left, bottom, right, top = src.bounds
        size = macro_size_m
        half = size / 2.0

        # --- X direction ---
        xmin = cx - half
        xmax = cx + half

        if xmin < left:
            xmin = left
            xmax = left + size
        if xmax > right:
            xmax = right
            xmin = right - size

        # --- Y direction ---
        ymin = cy - half
        ymax = cy + half

        if ymin < bottom:
            ymin = bottom
            ymax = bottom + size
        if ymax > top:
            ymax = top
            ymin = top - size

        # Safety check
        if (right - left) < size or (top - bottom) < size:
            return {
                "order": order,
                "year_month": year_month,
                "quad_path": str(quad_path),
                "image_path": str(img_path),
                "status": "failed_quad_too_small",
            }

        window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
        window = window.round_offsets().round_lengths()

        patch = src.read(window=window)
        transform = rasterio.windows.transform(window, src.transform)

        meta = src.meta.copy()
        meta.update({
            "height": window.height,
            "width": window.width,
            "transform": transform,
        })

        with rasterio.open(img_path, "w", **meta) as dst:
            dst.write(patch)

    return {
        "order": order,
        "year_month": year_month,
        "quad_path": str(quad_path),
        "image_path": str(img_path),
        "status": "created",
    }



def generate_all_macro_patches(
    macro_index: gpd.GeoDataFrame,
    macro_size_m: float,
    images_dir: str | Path,
    catalog_csv_path: str | Path,
    catalog_gpkg_path: str | Path,
    overwrite: bool = False,
):
    """
    Generate macro patches for all rows in macro_index and build a macro catalog.
    Returns a GeoDataFrame with one row per macro patch.
    """
    images_dir = Path(images_dir)
    catalog_csv_path = Path(catalog_csv_path)
    catalog_gpkg_path = Path(catalog_gpkg_path)

    results = []

    for _, row in tqdm(
        macro_index.iterrows(),
        total=len(macro_index),
        desc="Generating macro patches"
    ):
        
        res = generate_macro_patch_for_site(
            row=row,
            macro_size_m=macro_size_m,
            images_dir=images_dir,
            overwrite=overwrite,
        )
        # attach geometry for later GeoDataFrame construction
        res["geometry"] = row["geometry"]
        results.append(res)

    df = pd.DataFrame(results)

    # Save plain catalog (without geometry) as CSV
    df_no_geom = df.drop(columns=["geometry"])
    catalog_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_no_geom.to_csv(catalog_csv_path, index=False)

    # Build GeoDataFrame for spatial use and save as GPKG
    macro_gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs=macro_index.crs
    )

    catalog_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    macro_gdf.to_file(
        catalog_gpkg_path,
        layer="macro_catalog",
        driver="GPKG"
    )

    return macro_gdf


def _ym_to_int(ym: str) -> int:
    # expects "YYYY_MM"
    y, m = ym.split("_")
    return int(y) * 100 + int(m)


def select_polygons_for_month(site_polys: gpd.GeoDataFrame, target_ym: str) -> gpd.GeoDataFrame:
    """
    Given all polygons for one site (with label_date = YYYY_MM),
    return the polygon set that should apply to target_ym:
    the most recent label_date <= target_ym.
    """
    if site_polys.empty:
        return site_polys

    target = _ym_to_int(target_ym)

    # keep only valid label_date rows
    tmp = site_polys.dropna(subset=["label_date"]).copy()
    tmp["label_int"] = tmp["label_date"].astype(str).apply(_ym_to_int)

    eligible = tmp[tmp["label_int"] <= target]
    if eligible.empty:
        # no previous label exists for that month -> return empty (all background)
        return tmp.iloc[0:0]

    best_label = eligible["label_int"].max()
    return eligible[eligible["label_int"] == best_label].drop(columns=["label_int"])


def rasterize_mask_for_patch(
    patch_path: Path | str,
    site_mask_gpkg: Path | str,
    target_ym: str,
    out_mask_path: Path | str,
    layer: str = "mask",
    burn_value: int = 1,
    background: int = 0,
    overwrite: bool = False,
) -> dict:
    """
    Rasterize construction polygons for one site into a pixel-aligned mask for one patch.

    - patch_path: cs_{order}_{YYYY_MM}.tif
    - site_mask_gpkg: mask_{order}.gpkg (Polygon, EPSG:3857, field label_date=YYYY_MM)
    - target_ym: "YYYY_MM" from the patch name
    """
    patch_path = Path(patch_path)
    site_mask_gpkg = Path(site_mask_gpkg)
    out_mask_path = Path(out_mask_path)

    if out_mask_path.exists() and not overwrite:
        return {"status": "skipped_exists", "mask_path": str(out_mask_path)}

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    # Read patch grid
    with rasterio.open(patch_path) as src:
        out_meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # Read polygons (single site)
    polys = gpd.read_file(site_mask_gpkg, layer=layer)

    # Ensure CRS matches patch (should already be EPSG:3857)
    if polys.crs is None:
        raise ValueError(f"Mask file has no CRS: {site_mask_gpkg}")
    if polys.crs != crs:
        polys = polys.to_crs(crs)

    polys_for_month = select_polygons_for_month(polys, target_ym)

    # If no polygons apply -> all background mask
    if polys_for_month.empty:
        mask = np.full((height, width), background, dtype=np.uint8)
        used_label = None
    else:
        shapes = [(geom, burn_value) for geom in polys_for_month.geometry if geom is not None]
        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=background,
            dtype=np.uint8,
            all_touched=False,
        )
        # all polygons in the selected set share the same label_date by design
        used_label = str(polys_for_month.iloc[0]["label_date"])

    # Write mask
    out_meta.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 255,  # optional; keeps 0/1 valid
        }
    )

    with rasterio.open(out_mask_path, "w", **out_meta) as dst:
        dst.write(mask, 1)

    return {
        "status": "ok",
        "patch_path": str(patch_path),
        "mask_path": str(out_mask_path),
        "target_ym": target_ym,
        "used_label_date": used_label,
        "n_polygons": int(len(polys_for_month)),
    }


def ym_dash_to_underscore(ym: str) -> str:
    # Converts "YYYY-MM" -> "YYYY_MM"
    return str(ym).replace("-", "_")


def generate_masks_from_macro_catalog(
    macro_catalog_csv: Path | str,
    processed_root: Path | str,
    labels_subdir: str = "cropped_images/macro/labels",
    masks_layer: str = "mask",
    overwrite: bool = False,
    save_log_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Batch rasterization based on macro_catalog.csv.

    Expects columns: order, year_month, image_path
    - year_month in the CSV can be "YYYY-MM" or "YYYY_MM"
    - image_path points to cs_{order}_{YYYY_MM}.tif
    """
    macro_catalog_csv = Path(macro_catalog_csv)
    processed_root = Path(processed_root)

    df = pd.read_csv(macro_catalog_csv)

    required = {"order", "year_month", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"macro_catalog is missing columns: {missing}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rasterizing masks"):
        order = int(row["order"])
        ym = ym_dash_to_underscore(row["year_month"])
        patch_path = Path(row["image_path"])

        site_mask_gpkg = patch_path.parent / "mask" / f"mask_{order}.gpkg"

        out_dir = processed_root / labels_subdir / str(order)
        out_mask_path = out_dir / f"mask_{order}_{ym}.tif"

        try:
            res = rasterize_mask_for_patch(
                patch_path=patch_path,
                site_mask_gpkg=site_mask_gpkg,
                target_ym=ym,
                out_mask_path=out_mask_path,
                layer=masks_layer,
                overwrite=overwrite,
            )
        except Exception as e:
            res = {
                "status": "error",
                "patch_path": str(patch_path),
                "mask_path": str(out_mask_path),
                "order": order,
                "target_ym": ym,
                "error": repr(e),
            }

        res["order"] = order
        res["year_month"] = ym
        results.append(res)

    out = pd.DataFrame(results)

    if save_log_path is not None:
        save_log_path = Path(save_log_path)
        save_log_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_log_path, index=False)

    return out

import numpy as np
from dataclasses import dataclass

def compute_train_percentiles(
    macro_catalog_csv: Path | str,
    site_splits_csv: Path | str,
    bands_to_use: list[int],
    p_low: float = 2.0,
    p_high: float = 98.0,
    max_images: int | None = None,   # optional: sample subset for speed
    seed: int = 42,
) -> dict:
    macro = pd.read_csv(macro_catalog_csv)
    splits = pd.read_csv(site_splits_csv)

    df = macro.merge(splits[["order", "split"]], on="order", how="inner")
    df = df[df["split"] == "train"].copy()

    if max_images is not None and len(df) > max_images:
        df = df.sample(n=max_images, random_state=seed)

    # collect per-band pixel samples
    samples = {b: [] for b in bands_to_use}

    for p in df["image_path"].tolist():
        with rasterio.open(p) as src:
            arr = src.read()  # (bands, H, W)

        # keep selected bands
        arr = arr[bands_to_use, :, :].astype(np.float32)

        for i, b in enumerate(bands_to_use):
            samples[b].append(arr[i].ravel())

    stats = {}
    for b in bands_to_use:
        x = np.concatenate(samples[b], axis=0)
        p2 = float(np.percentile(x, p_low))
        p98 = float(np.percentile(x, p_high))
        stats[str(b)] = {"p2": p2, "p98": p98}

    return stats


def save_percentiles_json(stats: dict, out_path: Path | str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


@dataclass(frozen=True)
class CropWindow:
    x0: int
    y0: int
    w: int
    h: int
    kind: str  # "pos" or "neg"
    tries: int
    pos_pixels: int




def _clamp_center(c: int, half: int, max_size: int) -> int:
    # Center must allow a full crop inside [0, max_size)
    return int(np.clip(c, half, max_size - half))

def sample_deterministic_window(
    mask: np.ndarray,
    crop_size: int = 256,
) -> CropWindow:
    """
    Deterministic crop for val/test:
    - If positives exist, center the crop at the centroid of positive pixels.
    - Else, use a center crop.
    """
    H, W = mask.shape
    half = crop_size // 2

    ys, xs = np.where(mask > 0)

    if len(xs) > 0:
        cx = int(xs.mean())
        cy = int(ys.mean())
        kind = "pos_centroid"
    else:
        cx = W // 2
        cy = H // 2
        kind = "center"

    cx = _clamp_center(cx, half, W)
    cy = _clamp_center(cy, half, H)

    x0 = int(cx - half)
    y0 = int(cy - half)

    pos_pixels = int(mask[y0:y0 + crop_size, x0:x0 + crop_size].sum())

    return CropWindow(
        x0=x0,
        y0=y0,
        w=crop_size,
        h=crop_size,
        kind=kind,
        tries=1,
        pos_pixels=pos_pixels,
    )

def sample_jitter_window(
    mask: np.ndarray,
    crop_size: int = 256,
    jitter_radius: int = 20,
    pos_fraction: float = 0.7,
    max_tries: int = 50,
    rng: np.random.Generator | None = None,
) -> CropWindow:
    """
    Sample a 2D crop window (crop_size x crop_size) from a mask.

    - Positive crop: centered near a positive pixel with random jitter.
    - Negative crop: random window with zero positive pixels (or close to zero).
    """
    if rng is None:
        rng = np.random.default_rng()

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    H, W = mask.shape
    if crop_size > H or crop_size > W:
        raise ValueError(f"crop_size={crop_size} is larger than mask shape {mask.shape}")

    half = crop_size // 2
    want_pos = rng.random() < pos_fraction

    pos_coords = np.argwhere(mask > 0)
    has_pos = len(pos_coords) > 0

    # If there is no construction, force negative sampling
    if want_pos and not has_pos:
        want_pos = False

    # Precompute valid center bounds
    cx_min, cx_max = half, W - half
    cy_min, cy_max = half, H - half

    if want_pos:
        # Try jittered centers around a positive pixel
        for t in range(1, max_tries + 1):
            py, px = pos_coords[rng.integers(0, len(pos_coords))]
            dx = int(rng.integers(-jitter_radius, jitter_radius + 1))
            dy = int(rng.integers(-jitter_radius, jitter_radius + 1))

            cx = _clamp_center(px + dx, half, W)
            cy = _clamp_center(py + dy, half, H)

            x0 = cx - half
            y0 = cy - half

            crop = mask[y0:y0 + crop_size, x0:x0 + crop_size]
            pos_pixels = int((crop > 0).sum())

            if pos_pixels > 0:
                return CropWindow(x0, y0, crop_size, crop_size, "pos", t, pos_pixels)

        # Fallback: if jitter fails, use a strict center on a positive pixel
        py, px = pos_coords[rng.integers(0, len(pos_coords))]
        cx = _clamp_center(px, half, W)
        cy = _clamp_center(py, half, H)
        x0, y0 = cx - half, cy - half
        crop = mask[y0:y0 + crop_size, x0:x0 + crop_size]
        return CropWindow(x0, y0, crop_size, crop_size, "pos", max_tries, int((crop > 0).sum()))

    else:
        # Negative sampling: random windows with no positives
        for t in range(1, max_tries + 1):
            cx = int(rng.integers(cx_min, cx_max + 1))
            cy = int(rng.integers(cy_min, cy_max + 1))
            x0 = cx - half
            y0 = cy - half

            crop = mask[y0:y0 + crop_size, x0:x0 + crop_size]
            pos_pixels = int((crop > 0).sum())

            if pos_pixels == 0:
                return CropWindow(x0, y0, crop_size, crop_size, "neg", t, 0)

        # Fallback: return the least-positive crop found (rare in dense sites)
        # This avoids crashing training when background-free patches exist.
        best = None
        best_pos = None

        for t in range(1, 11):
            cx = int(rng.integers(cx_min, cx_max + 1))
            cy = int(rng.integers(cy_min, cy_max + 1))
            x0 = cx - half
            y0 = cy - half
            crop = mask[y0:y0 + crop_size, x0:x0 + crop_size]
            pos_pixels = int((crop > 0).sum())

            if best_pos is None or pos_pixels < best_pos:
                best_pos = pos_pixels
                best = CropWindow(x0, y0, crop_size, crop_size, "neg", max_tries + t, pos_pixels)

        return best



def read_crop_image_and_mask(
    image_path: Path | str,
    mask_path: Path | str,
    win: CropWindow,
    img_dtype: np.dtype = np.float32,
    mask_dtype: np.dtype = np.int64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a multi-band image patch and a single-band mask, then crop both using the same window.

    Returns:
      img_crop: (C, H, W) float32
      mask_crop: (H, W) int64
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    with rasterio.open(image_path) as src_img:
        img = src_img.read()  # (C, H, W)

    with rasterio.open(mask_path) as src_m:
        mask = src_m.read(1)  # (H, W)

    x0, y0, w, h = win.x0, win.y0, win.w, win.h

    img_crop = img[:, y0:y0 + h, x0:x0 + w].astype(img_dtype, copy=False)
    mask_crop = mask[y0:y0 + h, x0:x0 + w].astype(mask_dtype, copy=False)

    return img_crop, mask_crop


def filter_macro_catalog(
    macro_catalog_csv: Path | str,
    flagged_csv: Path | str,
    out_csv: Path | str | None = None,
) -> pd.DataFrame:
    """
    Remove flagged (order, year_month) rows from macro_catalog.
    """
    macro = pd.read_csv(macro_catalog_csv)
    flagged = pd.read_csv(flagged_csv)

    if flagged.empty:
        print("[filter_macro_catalog] flagged_macro.csv is empty — no patches excluded.")
        if out_csv is not None:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            macro.to_csv(out_csv, index=False)
        return macro

    # Keep only required columns
    flagged = flagged[["order", "year_month"]].copy()

    # Normalize types
    macro["order"] = macro["order"].astype(int)
    flagged["order"] = flagged["order"].astype(int)
    macro["year_month"] = macro["year_month"].astype(str)
    flagged["year_month"] = flagged["year_month"].astype(str)

    # Anti-join
    macro = macro.merge(
        flagged.assign(_flagged=True),
        on=["order", "year_month"],
        how="left",
    )
    filtered = macro[macro["_flagged"].isna()].drop(columns=["_flagged"]).reset_index(drop=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_csv, index=False)

    return filtered



class ConstructionJitterDataset(Dataset):
    def __init__(
        self,
        macro_catalog_csv: Path | str,
        processed_root: Path | str,
        splits_csv: Path | str | None = None,
        split: str | None = None,  # "train" | "val" | "test"
        crop_size: int = 256,
        jitter_radius: int = 20,
        pos_fraction: float = 0.7,
        bands_to_use: list[int] | None = None,
        rng_seed: int = 42,
        norm_stats_path: Path | str | None = None,
        return_meta: bool = True,
    ):
        """
        Dataset that generates 256x256 crops on-the-fly using jitter sampling.

        If splits_csv + split are provided, filtering is applied by site ('order').
        return_meta: if True, __getitem__ returns (img, mask, meta); if False, returns (img, mask).
        """
        self.df = pd.read_csv(macro_catalog_csv)
        self.processed_root = Path(processed_root)

        self.crop_size = crop_size
        self.jitter_radius = jitter_radius
        self.pos_fraction = pos_fraction
        self.split = split

        self.bands_to_use = bands_to_use if bands_to_use is not None else list(range(8))
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed
        self.return_meta = return_meta

        required = {"order", "year_month", "image_path"}
        missing = required - set(self.df.columns)
        

        ### ----- Load images statistics

        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else None
        self.do_minmax = self.norm_stats_path is not None

        if self.do_minmax:
            with self.norm_stats_path.open() as f:
                stats = json.load(f)

            # store p2/p98 aligned to bands_to_use order
            p2 = []
            p98 = []
            for b in self.bands_to_use:
                p2.append(stats[str(b)]["p2"])
                p98.append(stats[str(b)]["p98"])

            self.p2 = np.array(p2, dtype=np.float32)   # shape (C,)
            self.p98 = np.array(p98, dtype=np.float32) # shape (C,)

        
        if missing:
            raise ValueError(f"macro_catalog missing columns: {missing}")

        # Optional: apply site-level split
        if splits_csv is not None or split is not None:
            if splits_csv is None or split is None:
                raise ValueError("Provide both splits_csv and split, or neither.")
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be one of: 'train', 'val', 'test'")

            splits_df = pd.read_csv(splits_csv)
            if not {"order", "split"}.issubset(splits_df.columns):
                raise ValueError("splits_csv must have columns: order, split")

            allowed_orders = set(splits_df.loc[splits_df["split"] == split, "order"].astype(int).tolist())
            self.df["order"] = self.df["order"].astype(int)
            self.df = self.df[self.df["order"].isin(allowed_orders)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("Dataset is empty after filtering. Check split files/paths.")

    def __len__(self):
        return len(self.df)

    def _build_mask_path(self, order: int, year_month: str) -> Path:
        ym = ym_dash_to_underscore(year_month)
        return (
            self.processed_root
            / "cropped_images/macro/labels"
            / str(order)
            / f"mask_{order}_{ym}.tif"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        order = int(row["order"])
        year_month = row["year_month"]
        image_path = Path(row["image_path"])
        mask_path = self._build_mask_path(order, year_month)

        # Read mask (cheap)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Sample window

        if self.split == "train" or self.split is None:
            # stochastic sampling for training
            rng = self.rng
            jitter_radius = self.jitter_radius
            pos_fraction  = self.pos_fraction
        else:
            # deterministic sampling for val/test (stable metrics)
            import zlib
            key = f"{order}_{year_month}".encode("utf-8")
            seed = (zlib.crc32(key) + int(self.rng_seed)) % (2**32)
            rng = np.random.default_rng(seed)

            jitter_radius = 0            # <-- clave
            pos_fraction  = 0.7          # o 0.0 si quieres distribución natural

        if self.split == "train":
            win = sample_jitter_window(
                mask=mask,
                crop_size=self.crop_size,
                jitter_radius=self.jitter_radius,
                pos_fraction=self.pos_fraction,
                rng=self.rng,
            )
        else:
            win = sample_deterministic_window(
                mask=mask,
                crop_size=self.crop_size,
            )

        # Read crops
        img_crop, mask_crop = read_crop_image_and_mask(
            image_path=image_path,
            mask_path=mask_path,
            win=win,
            img_dtype=np.float32,
            mask_dtype=np.int64,
        )
        # Image normalization
        img_crop = img_crop[self.bands_to_use, :, :].astype("float32")  # (C,H,W)

        if self.do_minmax:
            # clip per band
            p2 = self.p2[:, None, None]
            p98 = self.p98[:, None, None]
            img_crop = np.clip(img_crop, p2, p98)

            # min-max per band
            denom = (p98 - p2)
            denom = np.where(denom == 0, 1.0, denom)  # avoid division by zero
            img_crop = (img_crop - p2) / denom

        # To torch
        img_tensor = torch.from_numpy(img_crop)  # float32, (C, H, W)

        if self.return_meta:
            # Training mode: mask as (1, H, W) float32 for BCEWithLogitsLoss
            mask_crop = mask_crop.astype("float32")
            mask_crop = np.expand_dims(mask_crop, axis=0)  # (1, H, W)
            mask_tensor = torch.from_numpy(mask_crop)
            meta = {
                "order": int(row["order"]),
                "year_month": row["year_month"],
            }
            return img_tensor, mask_tensor, meta
        else:
            # Exploration mode: mask as (H, W) long for direct inspection
            mask_tensor = torch.from_numpy(mask_crop)  # int64, (H, W)
            return img_tensor, mask_tensor


def make_site_splits(
    macro_catalog_csv: Path | str,
    out_splits_csv: Path | str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create site-level splits (by 'order') and try to match target fractions
    in terms of total rows (months) while keeping all rows from a site in the same split.

    Outputs a DataFrame: order, split
    """
    df = pd.read_csv(macro_catalog_csv)

    if "order" not in df.columns:
        raise ValueError("macro_catalog.csv must contain an 'order' column")

    # Count how many rows (months) each site contributes
    site_counts = (
        df.groupby("order")
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )

    total_rows = int(site_counts["n_rows"].sum())
    targets = {
        "train": int(round(total_rows * train_frac)),
        "val": int(round(total_rows * val_frac)),
        "test": total_rows,  # leftover assigned implicitly
    }

    rng = np.random.default_rng(seed)

    # Shuffle sites (but keep the n_rows distribution available)
    site_counts = site_counts.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    splits = []
    acc = {"train": 0, "val": 0, "test": 0}

    for _, r in site_counts.iterrows():
        site_id = int(r["order"])
        n = int(r["n_rows"])

        # Decide which split benefits most (greedy to match targets)
        # Priority: fill train, then val, else test
        if acc["train"] + n <= targets["train"]:
            s = "train"
        elif acc["val"] + n <= targets["val"]:
            s = "val"
        else:
            s = "test"

        acc[s] += n
        splits.append((site_id, s))

    splits_df = pd.DataFrame(splits, columns=["order", "split"]).sort_values("order")

    out_splits_csv = Path(out_splits_csv)
    out_splits_csv.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(out_splits_csv, index=False)

    return splits_df
