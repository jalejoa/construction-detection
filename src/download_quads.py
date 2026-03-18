from pathlib import Path
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import BASEMAPS_API_URL


def create_session(api_key: str) -> requests.Session:
    """
    Create a requests.Session with Planet API authentication and retry logic.
    """
    session = requests.Session()
    session.auth = (api_key, "")

    retries = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[429, 502, 503],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def ensure_dir(path: str | Path) -> None:
    """
    Ensure the parent directory of the given file path exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)



def build_quad_output_path(row, base_dir: str | Path) -> Path:
    """
    Build the output path for a quad using year_month, mosaic_id and quad_id.
    """
    base_dir = Path(base_dir)

    year_month = str(row["year_month"])  # e.g. "2020-06"
    ym_folder = year_month.replace("-", "_")

    mosaic_id = row["mosaic_id"]
    quad_id = row["quad_id"]

    return base_dir / ym_folder / mosaic_id / f"{quad_id}.tif"


def fetch_quad_metadata(
    session: requests.Session,
    mosaic_id: str,
    quad_id: str,
    timeout: int = 60,
) -> dict:
    """
    Fetch metadata for a given quad inside a mosaic.
    """
    url = f"{BASEMAPS_API_URL}/mosaics/{mosaic_id}/quads/{quad_id}"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_quad_file(
    session: requests.Session,
    download_url: str,
    output_path: str | Path,
    overwrite: bool = False,
    timeout: int = 300,
) -> dict:
    """
    Download a quad from its download_url into output_path.
    Returns a dict with status information.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "status": "skipped"}

    ensure_dir(output_path)

    try:
        with session.get(download_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return {"path": str(output_path), "status": "downloaded"}

    except Exception as e:
        return {"path": str(output_path), "status": f"failed: {e}"}


def download_quads_for_list(
    quads_df: pd.DataFrame,
    api_key: str,
    base_dir: str | Path,
    overwrite: bool = False,
    max_quads: int | None = None,
) -> pd.DataFrame:
    session = create_session(api_key)
    base_dir = Path(base_dir)

    if max_quads is not None:
        quads_df = quads_df.head(max_quads)

    logs = []
    total = len(quads_df)

    for i, row in quads_df.iterrows():
        mosaic_id = row["mosaic_id"]
        quad_id = row["quad_id"]

        output_path = build_quad_output_path(row, base_dir)

        try:
            meta = fetch_quad_metadata(session, mosaic_id, quad_id)
            download_url = meta["_links"]["download"]

            result = download_quad_file(
                session=session,
                download_url=download_url,
                output_path=output_path,
                overwrite=overwrite,
            )

        except Exception as e:
            result = {"path": str(output_path), "status": f"failed: {e}"}

        logs.append({
            "mosaic_id": mosaic_id,
            "quad_id": quad_id,
            "year_month": row.get("year_month"),
            "path": str(output_path),
            "status": result["status"],
        })

        if (len(logs) % 10) == 0:
            print(f"{len(logs)}/{total} processed...")

        if (len(logs) % 50) == 0:
            pd.DataFrame(logs).to_csv(f"{base_dir}/download_log_partial.csv", index=False)


    return pd.DataFrame(logs)