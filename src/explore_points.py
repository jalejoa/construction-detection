# explore_points.py

import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def load_points(shp_path: str):
    """Load point shapefile."""
    return gpd.read_file(shp_path)


def get_date_range(gdf, start_col="Start", end_col="End"):
    """Return min start date and max end date."""
    gdf = gdf.copy()
    gdf[start_col] = pd.to_datetime(gdf[start_col])
    gdf[end_col] = pd.to_datetime(gdf[end_col])
    return gdf[start_col].min(), gdf[end_col].max() 


def get_bbox(gdf):
    """Return bbox [minx, miny, maxx, maxy]."""
    return list(map(float, gdf.total_bounds))


def save_aoi_gpkg(gdf, out_gpkg_path):
    """Save points to GPKG."""
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame(
    {"id": ["AOI"], "geometry": [bbox_geom]},
    crs="EPSG:4326"
    )
    bbox_gdf.to_file(out_gpkg_path, layer="aoi_bbox", driver="GPKG")
    print (f"Saved AOI points to {out_gpkg_path}")


def save_metadata_json(min_date, max_date, bbox, out_json_path):
    """Save metadata JSON."""
    metadata = {
        "min_start_date": str(min_date.date()),
        "max_end_date": str(max_date.date()),
        "aoi_bbox": bbox,
    }
    with open(out_json_path, "w") as f:
        json.dump(metadata, f, indent=4)


def process_points(
    shp_path,
    out_gpkg_path,
    out_json_path,
    start_col="Start",
    end_col="End",
):
    """Run full process and return results."""
    gdf = load_points(shp_path)
    min_date, max_date = get_date_range(gdf, start_col, end_col)
    bbox = get_bbox(gdf)

    save_aoi_gpkg(gdf, out_gpkg_path)
    save_metadata_json(min_date, max_date, bbox, out_json_path)

    return min_date, max_date, bbox