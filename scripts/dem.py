"""Download a DEM for a study-area box and save it as a GeoTIFF."""
import logging
from pathlib import Path
from typing import Tuple, Union

import py3dep
from affine import Affine
from rasterio.enums import Resampling
from shapely.geometry import box

logger = logging.getLogger(__name__)


def download_dem(
    x: Tuple[float, float],
    y: Tuple[float, float],
    crs: str,
    resolution: int,
    out_path: Union[str, Path],
    buffer: float = 0.0,
) -> Path:
    """
    Download a 3DEP DEM covering a box (plus an optional buffer) and write it
    to a GeoTIFF. If the file already exists the download is skipped.

    Parameters
    ----------
    x, y : (min, max) tuples
        Box bounds in the coordinates of ``crs``.
    crs : str
        CRS of the box and of the output DEM, e.g. ``'EPSG:32613'``.
    resolution : int
        DEM cell size in meters (3DEP supports 10, 30, ...).
    out_path : str | Path
        Output GeoTIFF path. Parent folders are created.
    buffer : float
        Meters added on every side. Use at least the largest Sx search
        distance so cells at the box edge see their full upwind terrain.

    Returns
    -------
    pathlib.Path
        The GeoTIFF path.
    """
    out_path = Path(out_path)
    if out_path.exists():
        logger.info('DEM exists, skipping download: %s', out_path)
        return out_path

    aoi = box(x[0], y[0], x[1], y[1]).buffer(buffer, join_style='mitre')
    dem = py3dep.get_dem(geometry=aoi, resolution=resolution, crs=crs)
    dem = dem.rio.write_crs(dem.rio.crs)

    # py3dep returns the DEM in EPSG:5070 at a slightly different cell size,
    # so resample onto an exact grid in the requested CRS at the requested
    # resolution (snapped to multiples of the resolution) covering the AOI.
    minx, miny, maxx, maxy = aoi.bounds
    minx, miny = (resolution * (v // resolution) for v in (minx, miny))
    maxx, maxy = (resolution * -(-v // resolution) for v in (maxx, maxy))
    dem = dem.rio.reproject(
        crs,
        resampling=Resampling.bilinear,
        transform=Affine(resolution, 0, minx, 0, -resolution, maxy),
        shape=(int((maxy - miny) / resolution), int((maxx - minx) / resolution)),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dem.rio.to_raster(out_path)
    logger.info('Wrote DEM: %s', out_path)
    return out_path
