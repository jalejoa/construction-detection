import pandas as pd
import os
import time
import geopandas as gpd
from shapely.geometry import box
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASEMAPS_API_URL = "https://api.planet.com/basemaps/v1"


def paginated_get(session, url, item_key, **kwargs):
    while True:
        rv = session.get(url, **kwargs)
        rv.raise_for_status()
        page = rv.json()
        
        for item in page[item_key]:
            yield item
            
        if '_next' in page['_links']:
            url = page['_links']['_next']
        else:
            break

def get_mosaics_for_aoi(
    bbox,
    start_date,
    end_date,
    api_key: str,
    name_prefix: str = "ps_",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Planet mosaics, filter by prefix and time overlap,
    and return a reference table with:
    mosaic_id, name, time_start, time_end, year_month.
    """
    session = requests.Session()
    session.auth = (api_key, '')
    retries = Retry(total=10, backoff_factor=1, status_forcelist=[429, 502, 503])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    url = f"{BASEMAPS_API_URL}/mosaics"

    mosaics = list(paginated_get(session, url, item_key="mosaics"))

    if not mosaics:
        df_empty = pd.DataFrame(columns=["mosaic_id", "name", "time_start", "time_end", "year_month"])
        if save_path is not None:
            df_empty.to_csv(save_path, index=False)
        return df_empty

    records = []
    for m in mosaics:
        records.append({
            "mosaic_id": m.get("id"),
            "name": m.get("name"),
            "time_start": m.get("first_acquired"),
            "time_end":   m.get("last_acquired"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df

    df = df[df["name"].str.startswith(name_prefix)].copy()

    df["time_start"] = pd.to_datetime(df["time_start"]).dt.tz_localize(None)
    df["time_end"]   = pd.to_datetime(df["time_end"]).dt.tz_localize(None)

    mask = (df["time_end"] >= start_date) & (df["time_start"] <= end_date)

    df = df[mask].copy()

    df["year_month"] = df["time_start"].dt.strftime("%Y-%m")

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def expand_points_by_month(
    points_gdf: gpd.GeoDataFrame,
    mosaics_df: pd.DataFrame,
    id_col: str = "fid_1",
    date_start_col: str = "Start",
    date_end_col: str = "End",
    ongoing_col: str = "Ongoing",
    month_margin: int = 0,
) -> gpd.GeoDataFrame:
    """
    Expand each construction point into one row per month between Start and End,
    handling ongoing sites and attaching the corresponding mosaic_id by year_month.
    """
    df = points_gdf[[id_col, date_start_col, date_end_col, ongoing_col, "geometry"]].copy()

    now = pd.Timestamp.today().normalize()
    current_month_end = now + pd.offsets.MonthEnd(0)

    # Handle ongoing sites: set End to end of current month
    ongoing_mask = df[ongoing_col].fillna(False)
    df.loc[ongoing_mask, date_end_col] = current_month_end

    # Drop rows without valid dates
    df = df.dropna(subset=[date_start_col, date_end_col]).copy()

    # Ensure End >= Start
    valid_range = df[date_end_col] >= df[date_start_col]
    df = df[valid_range].copy()

    def months_for_row(row):
        start = row[date_start_col]
        end = row[date_end_col]

        if month_margin > 0:
            start = start - pd.DateOffset(months=month_margin)
            end = end + pd.DateOffset(months=month_margin)

        periods = pd.period_range(start, end, freq="M")
        if len(periods) == 0:
            return []
        return [p.strftime("%Y-%m") for p in periods]

    df["year_month"] = df.apply(months_for_row, axis=1)
    df = df.explode("year_month").dropna(subset=["year_month"])

    mosaics_small = mosaics_df[["year_month", "mosaic_id"]].drop_duplicates()
    merged = df.merge(mosaics_small, on="year_month", how="inner")

    result = gpd.GeoDataFrame(merged, geometry="geometry", crs=points_gdf.crs)
    return result





def get_quads_for_points(
    points_by_month: gpd.GeoDataFrame,
    api_key: str,
    buffer_deg: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    For each mosaic_id present in points_by_month, request only the quads
    that intersect the bounding box of the points that use that mosaic.
    Returns a GeoDataFrame with mosaic_id, quad_id, geometry (EPSG:4326).
    """
    session = requests.Session()
    session.auth = (api_key, '')
    retries = Retry(total=10, backoff_factor=1, status_forcelist=[429, 502, 503])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    mosaics = points_by_month["mosaic_id"].dropna().unique()

    records = []

    for mosaic_id in mosaics:
        subset = points_by_month[points_by_month["mosaic_id"] == mosaic_id]
        if subset.empty:
            continue

        minx, miny, maxx, maxy = subset.total_bounds

        minx = float(minx) - buffer_deg
        miny = float(miny) - buffer_deg
        maxx = float(maxx) + buffer_deg
        maxy = float(maxy) + buffer_deg

        bbox_str = f"{minx},{miny},{maxx},{maxy}"

        url = f"{BASEMAPS_API_URL}/mosaics/{mosaic_id}/quads"
        params = {"bbox": bbox_str}

        quads_iter = paginated_get(session, url, item_key="items", params=params)

        for q in quads_iter:
            qid = q.get("id")
            bbox = q.get("bbox", None)
            if not bbox:
                continue

            qminx, qminy, qmaxx, qmaxy = bbox
            geom = box(qminx, qminy, qmaxx, qmaxy)

            records.append({
                "mosaic_id": mosaic_id,
                "quad_id": qid,
                "geometry": geom,
            })

    if not records:
        return gpd.GeoDataFrame(columns=["mosaic_id", "quad_id", "geometry"], crs="EPSG:4326")

    quads_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    quads_gdf = quads_gdf.drop_duplicates(subset=["mosaic_id", "quad_id"])

    return quads_gdf

def assign_quads_to_points(
    points_by_month: gpd.GeoDataFrame,
    quads_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Spatially match points to quads using mosaic_id groups.
    Returns a clean GeoDataFrame with:
    fid_1, Start, End, Ongoing, year_month, mosaic_id, quad_id, geometry.
    """
    results = []

    mosaics = points_by_month["mosaic_id"].dropna().unique()

    for mid in mosaics:
        pts = points_by_month[points_by_month["mosaic_id"] == mid]
        quads = quads_gdf[quads_gdf["mosaic_id"] == mid]

        if pts.empty or quads.empty:
            continue

        joined = gpd.sjoin(
            pts,
            quads[["mosaic_id", "quad_id", "geometry"]],
            how="left",
            predicate="within"
        )

        results.append(joined)

    if not results:
        return gpd.GeoDataFrame(
            columns=["order", "Start", "End", "Ongoing", "year_month", "mosaic_id", "quad_id", "geometry"],
            crs=points_by_month.crs
        )

    out = pd.concat(results, ignore_index=True)

    if "mosaic_id_left" in out.columns and "mosaic_id_right" in out.columns:
        out["mosaic_id"] = out["mosaic_id_left"]
        out = out.drop(columns=["mosaic_id_left", "mosaic_id_right"], errors="ignore")

    out = out.drop(columns=["index_right"], errors="ignore")

    cols = ["order", "Start", "End", "Ongoing", "year_month", "mosaic_id", "quad_id", "geometry"]
    cols_existing = [c for c in cols if c in out.columns]
    out = out[cols_existing]

    return gpd.GeoDataFrame(out, geometry="geometry", crs=points_by_month.crs)

def extract_quads_to_download(point_quad: pd.DataFrame) -> pd.DataFrame:
    df = point_quad[["mosaic_id", "quad_id"]].drop_duplicates().reset_index(drop=True)
    return df

def build_quad_index(point_quad: pd.DataFrame) -> pd.DataFrame:
    out = (
        point_quad.groupby(["mosaic_id", "quad_id"])["order"]
        .agg(n_sites="nunique", sites=lambda x: sorted(set(x)))
        .reset_index()
    )
    return out


