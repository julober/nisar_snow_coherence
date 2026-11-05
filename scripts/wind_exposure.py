"""
Maximum upwind slope (Sx) wind exposure index from Winstral et al. (2002),
J. Hydrometeorology 3, 524-538.

For each cell we look upwind along rays inside a wedge centered on the
prevailing wind direction. On each ray, Sx is the steepest upward angle from
the cell of interest to any cell within ``dmax`` (paper Eq. 1). The final Sx
is the mean over all rays in the wedge (paper Eq. 2).

Sign convention (same as the paper): positive Sx = sheltered by upwind
terrain, negative Sx = exposed (the cell is higher than everything upwind).
"""
import numpy as np
import xarray as xr
from numba import njit, prange


@njit(parallel=True, cache=True)
def _sx_kernel(dem, dx, azimuths_rad, dmax):
    """Mean over rays of the max upward slope angle (degrees); NaN if no data."""
    nrow, ncol = dem.shape
    nstep = int(dmax // dx)
    out = np.full((nrow, ncol), np.nan)
    for r in prange(nrow):
        for c in range(ncol):
            z0 = dem[r, c]
            if np.isnan(z0):
                continue
            total = 0.0
            nray = 0
            for a in azimuths_rad:
                # Rows increase southward, so north (cos) is a negative row step.
                dcol = np.sin(a)
                drow = -np.cos(a)
                best = -np.inf
                for k in range(1, nstep + 1):
                    rr = r + int(np.round(drow * k))
                    cc = c + int(np.round(dcol * k))
                    if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
                        break  # ran off the DEM
                    z = dem[rr, cc]
                    if np.isnan(z):
                        continue
                    # True distance to the sampled cell (paper Eq. 1 denominator).
                    dist = dx * np.sqrt(float((rr - r) ** 2 + (cc - c) ** 2))
                    ang = np.arctan((z - z0) / dist)
                    if ang > best:
                        best = ang
                if best > -np.inf:
                    total += best
                    nray += 1
            if nray > 0:
                out[r, c] = np.degrees(total / nray)
    return out


def sx(dem, dx, azimuth, dmax, half_width=30.0, step=5.0):
    """
    Maximum upwind slope index for one search distance.

    Parameters
    ----------
    dem : 2D numpy array
        Elevation (m), rows running north to south (standard raster order).
        NaN cells are skipped.
    dx : float
        Square cell size (m).
    azimuth : float
        Prevailing direction the wind blows FROM (deg clockwise from north).
    dmax : float
        Maximum upwind search distance (m).
    half_width : float
        Half the wedge angle (deg); the paper uses 30 (60 deg total).
    step : float
        Angular spacing between rays (deg); the paper uses 5.

    Returns
    -------
    2D numpy array
        Sx in degrees, same shape as ``dem``.
    """
    n = int(round(2 * half_width / step)) + 1
    azimuths = np.deg2rad(azimuth + np.linspace(-half_width, half_width, n))
    return _sx_kernel(np.ascontiguousarray(dem, dtype=np.float64), float(dx), azimuths, float(dmax))


def compute_sx(dem: xr.DataArray, azimuth: float, dmaxs=(100, 300, 500), **kwargs) -> xr.Dataset:
    """
    Sx for several search distances from a projected-CRS DEM DataArray.

    Parameters
    ----------
    dem : xarray.DataArray
        2D (y, x) DEM in a projected CRS with equal x and y cell size and y
        decreasing (north up), e.g. opened with ``rioxarray``.
    azimuth : float
        Prevailing wind direction (deg FROM, clockwise from north).
    dmaxs : iterable of float
        Search distances (m); one ``sx_<dmax>`` variable is made for each.
    **kwargs
        Passed to :func:`sx` (``half_width``, ``step``).

    Returns
    -------
    xarray.Dataset
        Variables ``sx_<dmax>`` in degrees on the DEM grid, with the DEM CRS.
    """
    dx = abs(float(dem.x[1] - dem.x[0]))
    dy = abs(float(dem.y[1] - dem.y[0]))
    if not np.isclose(dx, dy):
        raise ValueError(f'DEM cells must be square (dx={dx}, dy={dy})')
    if dem.y[0] < dem.y[-1]:
        raise ValueError('DEM y must decrease (north-up); flip it first')

    ds = xr.Dataset(coords={'y': dem.y, 'x': dem.x})
    for d in dmaxs:
        ds[f'sx_{int(d)}'] = (
            ('y', 'x'),
            sx(dem.values, dx, azimuth, d, **kwargs).astype('float32'),
            {
                'long_name': f'Winstral max upwind slope index, dmax={d} m',
                'units': 'degrees',
                'wind_from_azimuth': float(azimuth),
            },
        )
    return ds.rio.write_crs(dem.rio.crs)
