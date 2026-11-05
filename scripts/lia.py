"""Local incidence angle (LIA) for a study-area box, from NISAR line-of-sight vectors and a DEM.

The line-of-sight (LOS) unit vectors come from the ``radarGrid`` metadata of one NISAR GSLC
granule (same group that ``nisar_pytools`` reads). LIA depends on the track geometry, not on the
acquisition date, so a single granule per track/frame is enough.
"""
import logging
from pathlib import Path
from typing import Tuple, Union

import earthaccess
import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
from pyproj import CRS, Transformer
from rasterio.enums import Resampling

from scripts.dem import download_dem
from scripts.download import _search_nisar

logger = logging.getLogger(__name__)

RADAR_GRID: str = 'science/LSAR/GSLC/metadata/radarGrid'
"""HDF5 group holding the LOS vectors, same group nisar_pytools uses for GSLC products."""


def derive_los_z(los_x: np.ndarray, los_y: np.ndarray) -> np.ndarray:
    """
    Up component of a unit LOS vector, ``sqrt(1 - x^2 - y^2)``. It is taken as positive
    (target-to-sensor points up). Rounding can push ``x^2 + y^2`` slightly above 1, so the
    argument is clipped at 0.
    """
    return np.sqrt(np.clip(1.0 - los_x**2 - los_y**2, 0.0, None))


def lia_from_los(
    dem: xr.DataArray,
    los_x: np.ndarray,
    los_y: np.ndarray,
    heights: np.ndarray,
    x_rg: np.ndarray,
    y_rg: np.ndarray,
    radar_epsg: int,
) -> xr.DataArray:
    """
    Local incidence angle (degrees) on the grid of ``dem``.

    ``nisar_pytools`` does no reprojection: it samples the radarGrid at the DEM's x/y values as
    given, so a DEM in another CRS would be sampled at the wrong place. Here the DEM is first
    reprojected to the radarGrid CRS (``radar_epsg``) if needed, the angle is computed there, and
    the result is resampled back to the original DEM grid.

    Parameters
    ----------
    dem : xarray.DataArray
        2D elevation (m, above the ellipsoid ideally) with ``x``/``y`` coordinates and a
        projected CRS in metres.
    los_x, los_y : numpy.ndarray
        East and north LOS components, shape ``(n_heights, n_y, n_x)``.
    heights, x_rg, y_rg : numpy.ndarray
        radarGrid axes.
    radar_epsg : int
        EPSG of the radarGrid x/y coordinates.
    """
    if dem.rio.crs is None:
        raise ValueError('dem must have a CRS')
    los_z = derive_los_z(los_x, los_y)

    dem_crs = dem.rio.crs
    same_crs = CRS.from_user_input(dem_crs) == CRS.from_epsg(radar_epsg)
    dem_work = dem if same_crs else dem.rio.reproject(f'EPSG:{radar_epsg}', resampling=Resampling.bilinear)

    lia = local_incidence_angle(dem_work, los_x, los_y, los_z, heights, x_rg, y_rg, epsg=radar_epsg)
    if not same_crs:
        lia = lia.rio.reproject_match(dem, resampling=Resampling.bilinear)
    return lia


def _read_los(
    url: str, bounds_utm: Tuple[float, float, float, float], box_epsg: int, margin: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Stream the LOS vectors of one granule, cropped to the box plus ``margin`` (m).

    Returns ``los_x, los_y, heights, x_rg, y_rg, epsg``.
    """
    fs = earthaccess.get_fsspec_https_session()
    with h5py.File(fs.open(url, cache_type='background', block_size=16 * 1024 * 1024), 'r') as hf:
        g = hf[RADAR_GRID]
        for name in ('losUnitVectorX', 'losUnitVectorY', 'heightAboveEllipsoid', 'xCoordinates', 'yCoordinates'):
            if name not in g:
                raise KeyError(f'{name} not in {RADAR_GRID}; found {sorted(g)}')
        epsg = int(g['projection'][()])
        x = g['xCoordinates'][:]
        y = g['yCoordinates'][:]

        # Box corners in the radarGrid CRS (they may differ from the box CRS).
        tf = Transformer.from_crs(box_epsg, epsg, always_xy=True)
        minx, miny, maxx, maxy = bounds_utm
        cx, cy = tf.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        xlo, xhi = min(cx) - margin, max(cx) + margin
        ylo, yhi = min(cy) - margin, max(cy) + margin

        xi = np.where((x >= xlo) & (x <= xhi))[0]
        yi = np.where((y >= ylo) & (y <= yhi))[0]
        if xi.size == 0 or yi.size == 0:
            raise ValueError('The box does not intersect this granule\'s radarGrid.')
        xs = slice(xi[0], xi[-1] + 1)
        ys = slice(yi[0], yi[-1] + 1)

        los_x = g['losUnitVectorX'][:, ys, xs].astype(np.float64)
        los_y = g['losUnitVectorY'][:, ys, xs].astype(np.float64)
        heights = g['heightAboveEllipsoid'][:]
    return los_x, los_y, heights, x[xs], y[ys], epsg


def compute_lia(
    x: Tuple[float, float],
    y: Tuple[float, float],
    crs: str,
    track: int,
    frame: int,
    start_date: str,
    end_date: str,
    resolution: int = 10,
    margin: float = 5000.0,
    out_dir: Union[str, Path] = 'data/lia',
    dem_path: Union[str, Path, None] = None,
) -> Path:
    """
    Compute the local incidence angle for a box and write it as a GeoTIFF.

    Steps: download a DEM for the box (skipped if present), find one NISAR GSLC granule for the
    track/frame, stream its LOS vectors cropped to the box, and compute LIA on the DEM grid.
    Needs working Earthdata credentials (``earthaccess.login``). If the output exists it is
    returned without recomputing.

    Parameters
    ----------
    x, y : (min, max) tuples
        Box bounds in ``crs``.
    crs : str
        Projected CRS of the box and the output, e.g. ``'EPSG:32613'``.
    track, frame : int
        NISAR relative orbit and frame. The output name contains both.
    start_date, end_date : str
        Search window for the granule; any acquisition gives the same geometry, the first hit is used.
    resolution : int
        DEM and output cell size in metres.
    margin : float
        Metres of radarGrid kept around the box so interpolation has neighbours (the radarGrid is coarse).
    out_dir : str | Path
        Output folder, created if needed.
    dem_path : str | Path, optional
        DEM GeoTIFF to use or create. Default ``data/dem/dem_{resolution}m_track{T}_frame{F}.tif``.

    Returns
    -------
    pathlib.Path
        ``{out_dir}/lia_track{track:03d}_frame{frame:03d}_{resolution}m.tif``: float32 degrees,
        NaN where the LOS grid does not cover a pixel.
    """
    out_path = Path(out_dir) / f'lia_track{track:03d}_frame{frame:03d}_{resolution}m.tif'
    if out_path.exists():
        logger.info('LIA exists, skipping: %s', out_path)
        return out_path

    if dem_path is None:
        dem_path = Path('data/dem') / f'dem_{resolution}m_track{track:03d}_frame{frame:03d}.tif'
    dem_path = download_dem(x, y, crs, resolution, dem_path)
    dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze(drop=True)

    earthaccess.login()
    results = _search_nisar('GSLC', track, frame, start_date, end_date)
    if not results:
        raise ValueError(f'No GSLC granule for track {track}, frame {frame}, {start_date} to {end_date}.')
    logger.info('Using granule %s', results[0].properties['sceneName'])

    box_epsg = CRS.from_user_input(crs).to_epsg()
    los_x, los_y, heights, x_rg, y_rg, radar_epsg = _read_los(
        results[0].properties['url'], (x[0], y[0], x[1], y[1]), box_epsg, margin
    )
    lia = lia_from_los(dem, los_x, los_y, heights, x_rg, y_rg, radar_epsg)

    frac_nan = float(np.isnan(lia.values).mean())
    if frac_nan > 0:
        logger.warning('%.1f%% of LIA pixels are NaN (outside the LOS grid or its height range).', 100 * frac_nan)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lia.rio.write_nodata(np.nan).rio.to_raster(out_path)
    logger.info('Wrote LIA: %s', out_path)
    return out_path
