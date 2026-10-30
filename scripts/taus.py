## ADAPTED FROM https://github.com/ZachHoppinen/uavsar-coherence/
"""
Temporal coherence decay model, gamma(t) = gamma_inf + (1 - gamma_inf) * exp(-t / tau), fit per pixel.

Changes from the original: gamma_inf is fit by default (pass a number to fix it), the bounds of gamma_inf and tau
are separate, a pixel is fit on its valid (non-NaN) pairs only, and the R^2 of the fit is returned (the parameter
errors are optional, see ``metrics``).
"""
import numpy as np
from scipy.optimize import curve_fit

# Starting value for gamma_inf when it is fit (tau's starting value is a function argument).
GAMMA_INF_GUESS = 0.3


def decorrelation_temporal_model(t, gamma_inf, tau):
    """Coherence after a time difference ``t`` (same units as ``tau``) for asymptotic coherence ``gamma_inf``."""
    return gamma_inf + (1 - gamma_inf) * np.exp(-t / tau)


def r_squared(y, y_hat):
    """Coefficient of determination of a fit, 1 - SS_res / SS_tot. NaN if ``y`` is constant (SS_tot is 0)."""
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - np.sum((y - y_hat) ** 2) / ss_tot)


def fit_coh_decay_model(cohs, days, tau_guess=6, tau_bounds=(0, 100), xtol=1e-6, ftol=1e-6, gamma_inf=None):
    """
    Fit the decay model to one pixel's coherence time series.

    cohs : 1D array of coherence values
    days : 1D array of time differences, same length as ``cohs``
    tau_guess : starting value of tau
    tau_bounds : (lower, upper) bounds of tau, in the units of ``days``
    xtol, ftol : curve_fit tolerances
    gamma_inf : if given, held fixed and only tau is fit; if None, both are fit (gamma_inf bounded to [0, 1])

    Returns ``(gamma_inf, tau, r2, gamma_inf_err, tau_err)``: r2 is the R^2 of the whole fit and the errors are the
    1-sigma parameter errors; for a fixed gamma_inf, its error is NaN.
    Raises (RuntimeError, ValueError) if the fit fails, as curve_fit does.
    """
    lo, hi = tau_bounds
    if gamma_inf is not None:
        params, pcov = curve_fit(lambda t, tau: decorrelation_temporal_model(t, gamma_inf, tau),
                                 days, cohs, p0=(tau_guess,), bounds=(lo, hi), ftol=ftol, xtol=xtol)
        r2 = r_squared(cohs, decorrelation_temporal_model(days, gamma_inf, params[0]))
        return float(gamma_inf), float(params[0]), r2, np.nan, float(np.sqrt(pcov[0, 0]))
    params, pcov = curve_fit(decorrelation_temporal_model, days, cohs, p0=(GAMMA_INF_GUESS, tau_guess),
                             bounds=((0, lo), (1, hi)), ftol=ftol, xtol=xtol)
    err = np.sqrt(np.diag(pcov))
    r2 = r_squared(cohs, decorrelation_temporal_model(days, *params))
    return float(params[0]), float(params[1]), r2, float(err[0]), float(err[1])


# Optional per-pixel maps, requested by name (the parameter 1-sigma errors); R^2 is always returned.
METRICS = ('tau_err', 'gamma_inf_err')


def get_taus_spatial(coh_arr, days, tau_guess=6, tau_bounds=(0, 100), xtol=1e-6, ftol=1e-6, gamma_inf=None,
                     min_pairs=10, metrics=()):
    """
    Fit the decay model at every pixel of a block.

    coh_arr : 3D array (n_pairs, ny, nx)
    days : 1D array (n_pairs,) of time differences
    min_pairs : a pixel with fewer valid (non-NaN) pairs than this is not fit
        (with gamma_inf free it needs at least 3 to be determined at all)
    metrics : extra maps to return, any of ``METRICS`` ('tau_err', 'gamma_inf_err'); none by default

    Returns a dict of (ny, nx) arrays: ``tau``, ``gamma_inf``, ``r2`` (R^2 of the fit) and any requested metrics
    (float32, NaN where not fit), ``n_pairs`` (int16, valid pairs per pixel) and ``success`` (bool).
    """
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f'unknown metrics {sorted(unknown)}; choose from {METRICS}')
    days = np.asarray(days, dtype='float64')
    n, ny, nx = coh_arr.shape
    if days.shape != (n,):
        raise ValueError(f'days has shape {days.shape} but coh_arr has {n} pairs')
    min_pairs = max(min_pairs, 2 if gamma_inf is not None else 3)

    out = {k: np.full((ny, nx), np.nan, dtype='float32') for k in ('tau', 'gamma_inf', 'r2', *metrics)}
    valid = np.isfinite(coh_arr)
    out['n_pairs'] = valid.sum(axis=0).astype('int16')
    out['success'] = np.zeros((ny, nx), dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            ok = valid[:, iy, ix]
            if out['n_pairs'][iy, ix] < min_pairs:
                continue
            try:
                g, tau, r2, g_err, tau_err = fit_coh_decay_model(
                    coh_arr[ok, iy, ix], days[ok], tau_guess, tau_bounds, xtol, ftol, gamma_inf)
            except (RuntimeError, ValueError):  # no convergence, or bad input for this pixel: leave it as not fit
                continue
            out['tau'][iy, ix], out['gamma_inf'][iy, ix], out['r2'][iy, ix] = tau, g, r2
            if 'tau_err' in metrics:
                out['tau_err'][iy, ix] = tau_err
            if 'gamma_inf_err' in metrics:
                out['gamma_inf_err'][iy, ix] = g_err
            out['success'][iy, ix] = True
    return out
