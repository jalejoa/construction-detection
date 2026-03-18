# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial ML pipeline for detecting construction sites using Planet satellite imagery (8-band multispectral). The workflow proceeds in stages: explore construction site points → query Planet Basemaps API → download image quads → crop macro patches → generate segmentation masks → pretrain a U-Net on buildings → fine-tune on construction sites.

## API Key Setup

The pipeline reads `PLANET_API_KEY` from the environment. Copy `.env.example` to `.env` and set the key there — `config.py` loads it automatically via `python-dotenv`. `config.py` also defines all data path constants (`DATA_DIR`, `DATA_RAW`, `DATA_PROCESSED`, `DATA_QUADS`, `DATA_MODELS`) and the shared `BASEMAPS_API_URL`.

## Running Scripts

All scripts in `src/` import from `config.py` using relative imports, so run them from the `src/` directory or ensure `src/` is on `PYTHONPATH`:

```bash
# Pretrain U-Net on buildings
cd src
python pretrain_unet.py --mode train --dataset <dataset_name> --out_name unet_8b_v1 --epochs 60

# Evaluate pretrained model
python pretrain_unet.py --mode eval --dataset <dataset_name> --out_name unet_8b_v1 --split val

# Fine-tune on construction sites (OOP trainer with evaluation/threshold-sweep API)
python finetune_trainer.py --pretrained_name unet_8b_v1 --out_name cs_ft_v1
```

Notebooks in `notebooks/` document the workflow interactively (01→02→03→04).

## Architecture

### Data Pipeline (stages in order)

1. **`explore_points.py`** — Loads a shapefile of construction site points, extracts date range and AOI bounding box, saves to GPKG and JSON.

2. **`mosaics.py`** — Queries Planet Basemaps API for mosaics covering the AOI and date range. Expands each construction point into one row per month (`expand_points_by_month`), then queries which image quads cover those points (`get_quads_for_points`, `assign_quads_to_points`). Produces a `point_quad` table linking each `(order, year_month)` to a `(mosaic_id, quad_id)`.

3. **`download_quads.py`** — Downloads the actual `.tif` quad files from Planet. Saves to `data/quads/<YYYY_MM>/<mosaic_id>/<quad_id>.tif`.

4. **`patches.py`** — Crops macro patches (large square windows in meters) from quads, centered on each construction site. Also handles:
   - Rasterizing polygon labels (GeoPackage masks) into pixel-aligned mask TIFFs
   - `ConstructionJitterDataset`: PyTorch Dataset with jitter-based positive/negative crop sampling during training, deterministic centroid-based sampling for val/test
   - `make_site_splits`: greedy site-level train/val/test split (splits entire sites, not individual images)
   - Band normalization using p2/p98 percentile stats loaded from JSON

#### ConstructionJitterDataset — return modes

`__getitem__` has two modes controlled by `return_meta` (default `True`):

| `return_meta` | Returns | mask shape | mask dtype | Use case |
|---|---|---|---|---|
| `True` (default) | `(img, mask, meta)` | `(1, H, W)` | `float32` | Training / fine-tuning (BCEWithLogitsLoss) |
| `False` | `(img, mask)` | `(H, W)` | `int64` | Exploration / notebook 02 inspection |

Always pass `return_meta=False` in notebook 02 exploration cells. Notebook 04 and `finetune_trainer.py` use the default (`True`).

### Model Architecture

`UNet`, `DoubleConv`, `bce_dice_loss`, `iou_f1_from_logits`, and `set_seed` all live in `src/model.py` and are imported by `pretrain_unet.py`, `finetune_trainer.py`, and `visualize_unet.py`. The U-Net is a standard 4-level encoder-decoder with skip connections, operating on 8-channel input and producing a 1-channel binary segmentation output. Loss is BCE + Dice (50/50).

### Fine-tuning Strategy

`finetune_trainer.py` uses a **2-phase approach** via `ConstructionTrainer`:
- **Phase 1**: Encoder + bottleneck frozen, only decoder + head trained (higher LR ~1e-4)
- **Phase 2**: All layers unfrozen (lower LR ~1e-5)

Post-training: `trainer.evaluate(split="test", thr=0.35)` and `trainer.sweep_thresholds(split="val")`.

### Key Data Identifiers

- `order`: unique integer ID per construction site (replaces `fid_1` in newer code)
- `year_month`: string `"YYYY-MM"` or `"YYYY_MM"` (inconsistent; functions convert between them)
- Patch files: `data/processed/cropped_images/macro/<order>/cs_<order>_<YYYY_MM>.tif`
- Mask files: `data/processed/cropped_images/macro/labels/<order>/mask_<order>_<YYYY_MM>.tif`
- Label polygons (per site): `<site_dir>/mask/mask_<order>.gpkg` with a `label_date` field (`YYYY_MM`) used for temporal label matching

### Model Output Paths

- Pretrained building models: `data/models/building_pretrain/<out_name>/`
- Fine-tuned construction models: `data/models/construction_ft/<out_name>/`
- Each run saves: `best.pth`, `last.pth`, `history.npy`, `run_config.json`
