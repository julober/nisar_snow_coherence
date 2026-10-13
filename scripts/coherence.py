import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

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


def expected_coherence(gamma, looks: float) -> np.ndarray:
    """
    Expected sample coherence magnitude E{|gamma_hat|} for true coherence
    ``gamma`` estimated from ``looks`` independent looks, Eq. (62) of Bamler
    and Hartl (1998):
    ``G(L) G(3/2) / G(L + 1/2) * 3F2(3/2, L, L; L + 1/2, 1; |gamma|^2) * (1 - |gamma|^2)^L``.

    Parameters
    ----------
    gamma : array-like
        True coherence in [0, 1).
    looks : float
        (Effective) number of looks, L.

    Returns
    -------
    numpy.ndarray
        Expected sample coherence, same shape as ``gamma``. It is >= ``gamma``
        (the estimator overestimates low coherence) and tends to ``gamma`` as
        ``looks`` grows.
    """
    g2 = np.atleast_1d(np.asarray(gamma, dtype=float)).ravel() ** 2
    # sum the 3F2 series in log space, since terms are huge before the (1 - g2)^L factor tames them
    k = np.arange(20000)[:, None]
    log_ratio = np.log(1.5 + k) + 2 * np.log(looks + k) - np.log(looks + 0.5 + k) - 2 * np.log(1 + k) + np.log(np.maximum(g2, 1e-300))
    log_terms = np.vstack([np.zeros((1, g2.size)), np.cumsum(log_ratio[:-1], axis=0)])
    log_f = logsumexp(log_terms, axis=0)
    log_pref = gammaln(looks) + gammaln(1.5) - gammaln(looks + 0.5)
    out = np.exp(log_pref + log_f + looks * np.log1p(-g2))
    return out.reshape(np.shape(gamma)) if np.ndim(gamma) else out


def _debias(coherence, looks: float, n: int = 200):
    """Numpy inverse of ``expected_coherence`` by table interpolation."""
    true = np.append(np.linspace(0, 0.98, n), 1.0)  # series is slow near 1, where E{|g|} = g = 1 exactly
    return np.interp(coherence, np.append(expected_coherence(true[:-1], looks), 1.0), true)


def debias_coherence(coherence: xr.DataArray, looks: float, n: int = 200) -> xr.DataArray:
    """
    True coherence from sample coherence, by inverting ``expected_coherence``
    (Eq. 9 of Hoppinen et al., 2026). ``expected_coherence`` is tabulated on
    ``n`` true-coherence values in [0, 0.98] (plus the exact point 1 -> 1) and interpolated backwards.

    Parameters
    ----------
    coherence : xarray.DataArray
        Sample coherence magnitude. Values below the zero-coherence bias
        ``expected_coherence(0, looks)`` map to 0.
    looks : float
        (Effective) number of looks of ``coherence``.
    n : int, default=200
        Number of table entries.

    Returns
    -------
    xarray.DataArray
        True coherence, same shape as ``coherence``.
    """
    return xr.apply_ufunc(
        _debias, coherence, kwargs={'looks': looks, 'n': n},
        dask='parallelized', output_dtypes=[float], keep_attrs=True,
    )


def effective_looks_factor(coh_20: float, coh_80: float, n_20: float, n_80: float) -> float:
    """
    Oversampling factor ``f`` (effective looks ``L = N / f``) for the 20 m and
    80 m grids, found over zero-coherence open water where the debiased 20 m
    and 80 m coherence must agree (Hoppinen et al., 2026, Sec. 3.6).

    Parameters
    ----------
    coh_20, coh_80 : float
        Mean sample coherence over the water pixels at 20 m (block-averaged
        onto the 80 m grid) and at 80 m.
    n_20, n_80 : float
        Nominal looks at each resolution (multilook rows * columns).

    Returns
    -------
    float
        ``f``, such that ``debias(coh_20, n_20 / f) == debias(coh_80, n_80 / f)``.
    """
    def diff(f):
        return _debias(coh_20, n_20 / f) - _debias(coh_80, n_80 / f)

    # debiased values clip to 0 once f is large, so take the first crossing
    fs = np.linspace(1, 3, 101)
    i = np.argmax([diff(f) <= 0 for f in fs])
    return brentq(diff, fs[i - 1], fs[i])
