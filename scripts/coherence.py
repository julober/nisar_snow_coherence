import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def thermal_coherence(snr: xr.DataArray, starts, ends) -> xr.DataArray:
    """
    Thermal-noise coherence (gamma_thermal) for each interferometric pair,
    ``1 / sqrt((1 + 1/SNR_1) * (1 + 1/SNR_2))``, where SNR_1 and SNR_2 are the
    (linear) SNR of the two acquisitions of the pair.

    Parameters
    ----------
    snr : xarray.DataArray
        Linear SNR with a ``date`` dimension (e.g. ``build_snr_timeseries``
        output, ``db=False``), in the same local time as ``starts``/``ends``.
    starts, ends : array-like of datetime64
        Start (reference) and end (secondary) times of each pair, e.g. the
        ``start``/``end`` coordinates of a ``build_cube`` result. Each is
        matched to the nearest ``snr`` date.

    Returns
    -------
    xarray.DataArray
        ``(pair, y, x)`` array; ``pair`` strings are ``'YYYYMMDD_YYYYMMDD'``
        built from ``starts``/``ends``, matching ``build_cube``.
    """
    rho, pairs = [], []
    for start, end in zip(starts, ends):
        snr1 = snr.sel(date=start, method='nearest')
        snr2 = snr.sel(date=end, method='nearest')
        rho.append((1.0 / np.sqrt((1 + 1 / snr1) * (1 + 1 / snr2))).reset_coords(drop=True))
        pairs.append(f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}")
    return xr.concat(rho, dim='pair').assign_coords(pair=pairs)


def spatial_coherence(
    baseline: xr.DataArray, theta: xr.DataArray, slant_range: xr.DataArray, res: float, wvl: float
) -> xr.DataArray:
    """
    Spatial-baseline coherence (gamma_spatial),
    ``1 - 2 * |B_perp| * res * cos(theta)**2 / (wvl * R)``.

    Parameters
    ----------
    baseline : xarray.DataArray
        Perpendicular baseline (m).
    theta : xarray.DataArray
        Incidence angle, passed straight to ``np.cos``, so it must be in
        radians (NISAR's ``incidence_angle`` layer is stored in degrees;
        convert with ``np.deg2rad`` first if that is what is passed in).
    slant_range : xarray.DataArray
        Slant range (m).
    res : float
        Ground resolution (m) of the quantity being corrected.
    wvl : float
        Radar wavelength (m).

    Returns
    -------
    xarray.DataArray
        Same shape as the inputs.
    """
    return 1 - ((2 * np.abs(baseline) * res * (np.cos(theta) ** 2)) / (wvl * slant_range))
