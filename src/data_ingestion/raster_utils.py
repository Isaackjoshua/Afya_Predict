"""Optional raster helpers used by the live paths of the satellite adapters.

Imported lazily so the platform runs without `rasterio`/`geopandas`, which are
heavy wheels that many low-spec deployments cannot build (shortcoming #9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.core.logging import get_logger
from src.core.types import RegionConfig

log = get_logger("ingest.raster")


def load_boundaries(region: RegionConfig):
    """Load the council boundary GeoDataFrame declared by the region config."""
    import geopandas as gpd

    if not region.geojson_path:
        raise FileNotFoundError(f"region {region.name} declares no boundaries.geojson_path")
    path = Path(region.geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"boundary file not found: {path}")
    frame = gpd.read_file(path)
    name_col = next(
        (c for c in ("name", "NAME", "district", "council", "shapeName") if c in frame.columns),
        None,
    )
    if name_col is None:
        raise ValueError(f"{path} has no recognisable district-name column")
    return frame.rename(columns={name_col: "district"}).to_crs(region.crs)


def download_raster(url: str, dest_dir: Path, timeout: int = 120) -> Path:
    """Download `url` into `dest_dir`, reusing an existing file when present."""
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rstrip("/").split("/")[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return dest


def zonal_means(raster_path: Path, boundaries, nodata: Optional[float] = None) -> Dict[str, float]:
    """Mean raster value inside each boundary polygon, keyed by district name."""
    import rasterio
    from rasterio.mask import mask as rio_mask

    out: Dict[str, float] = {}
    with rasterio.open(raster_path) as src:
        fill = nodata if nodata is not None else src.nodata
        for _, row in boundaries.iterrows():
            try:
                clipped, _ = rio_mask(src, [row.geometry.__geo_interface__], crop=True, filled=True)
            except ValueError:  # polygon outside raster extent
                continue
            data = clipped.astype("float64")
            if fill is not None:
                data[data == fill] = np.nan
            data[data < -1e30] = np.nan
            if np.all(np.isnan(data)):
                continue
            out[str(row["district"])] = float(np.nanmean(data))
    return out


def point_samples(raster_path: Path, region: RegionConfig) -> Dict[str, float]:
    """Sample a raster at each district centroid (fallback when polygons absent)."""
    import rasterio

    out: Dict[str, float] = {}
    with rasterio.open(raster_path) as src:
        coords = [(d.lon, d.lat) for d in region.districts]
        for district, (value,) in zip(region.districts, src.sample(coords)):
            if src.nodata is not None and value == src.nodata:
                continue
            out[district.name] = float(value)
    return out
