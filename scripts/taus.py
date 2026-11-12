## ADAPTED FROM https://github.com/ZachHoppinen/uavsar-coherence/
"""
Temporal coherence decay model, gamma(t) = gamma_inf + (gamma_0 - gamma_inf) * exp(-t / tau), fit per pixel, and a
snow version of it that subtracts kappa * swe * cos(inc / 2).

Changes from the original: gamma_0 is tunable (fixed at 1 by default; pass ``gamma_0=None`` to fit it), gamma_inf is
fit by default (pass a number to fix it), the bounds of gamma_inf and tau are separate, a pixel is fit on its valid
(non-NaN) pairs only, and the goodness of fit (NSE, KGE and Pearson r) is returned (the parameter errors are
optional, see ``METRICS``).
"""
import numpy as np
from scipy.optimize import curve_fit

# Starting values for the parameters that are fit (tau's starting value is a function argument).
GAMMA_0_GUESS = 0.9
GAMMA_INF_GUESS = 0.3
KAPPA_GUESS = 0.1


def decorrelation_temporal_model(t, gamma_0, gamma_inf, tau):
    """Coherence after a time difference ``t`` (same units as ``tau``), decaying from ``gamma_0`` to ``gamma_inf``."""
    return gamma_inf + (gamma_0 - gamma_inf) * np.exp(-t / tau)


def decorrelation_temporal_model_snow(t, swe, inc, gamma_0, gamma_inf, tau, k, s0, p):
    """
    The decay model with a snow term.

    swe : snow water equivalent of the pair (``kappa`` carries its units)
    inc : local incidence angle in DEGREES (converted to radians here)
    """
    return (decorrelation_temporal_model(t, gamma_0, gamma_inf, tau)
            * gamma_snow(swe, inc, k, s0, p, gamma_inf))

def gamma_snow(swe, theta, k=100, s0=5, p=5, gamma_inf=0.15):
    a = np.cos(np.deg2rad(theta) / 2.0) 
    w = 1.0 - np.exp(-swe / s0) 
    # w = 1
    return (1 - gamma_inf) * np.exp(-swe / k) * (a ** (p * w)) + gamma_inf

# ---------------------------------------------------------------------------------------------------------------
# Goodness of fit. ``y`` is the observation and ``y_hat`` the model; all return a float, NaN if undefined.
# ---------------------------------------------------------------------------------------------------------------

def nse(y, y_hat):
    """Nash-Sutcliffe efficiency, 1 - SS_res / SS_tot (1 = perfect, 0 = no better than the mean of ``y``).
    NaN if ``y`` is constant (SS_tot is 0)."""
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - np.sum((y - y_hat) ** 2) / ss_tot)


def pearson_r(y, y_hat):
    """Pearson correlation between ``y`` and ``y_hat``. NaN if either is constant."""
    dy, dh = y - np.mean(y), y_hat - np.mean(y_hat)
    denom = np.sqrt(np.sum(dy ** 2) * np.sum(dh ** 2))
    if denom == 0:
        return np.nan
    return float(np.sum(dy * dh) / denom)


def kge(y, y_hat):
    """
    Kling-Gupta efficiency (Gupta et al., 2009), 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2), with r the
    Pearson correlation, alpha = std(y_hat) / std(y) and beta = mean(y_hat) / mean(y). 1 is perfect.
    NaN if ``y`` is constant or has a zero mean, or ``y_hat`` is constant (r is undefined).
    """
    r = pearson_r(y, y_hat)
    mean_y, std_y = np.mean(y), np.std(y)
    if np.isnan(r) or std_y == 0 or mean_y == 0:
        return np.nan
    alpha = np.std(y_hat) / std_y
    beta = np.mean(y_hat) / mean_y
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def fit_metrics(y, y_hat):
    """``{'nse', 'kge', 'pearson_r'}`` of a fit."""
    return {'nse': nse(y, y_hat), 'kge': kge(y, y_hat), 'pearson_r': pearson_r(y, y_hat)}


# ---------------------------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------------------------

def _fit(model, t, y, fixed, guesses, bounds, xtol, ftol):
    """
    Fit ``model(t, **params)`` to ``y`` with ``curve_fit``, varying only the parameters named in ``guesses``.

    model : function (t, **params), for example a partial of the decay model with swe and inc already bound
    fixed : {name: value} of parameters held constant
    guesses : {name: start value} of the parameters to fit (order = order of the fit)
    bounds : {name: (lo, hi)} for every fitted parameter

    Returns ``(values, errors, y_hat)``: dicts over all parameters (fixed ones have NaN error), and the fitted curve.
    Raises (RuntimeError, ValueError) if the fit fails, as curve_fit does.
    """
    free = list(guesses)

    def f(t, *p):
        return model(t, **fixed, **dict(zip(free, p)))

    popt, pcov = curve_fit(f, t, y, p0=[guesses[k] for k in free],
                           bounds=([bounds[k][0] for k in free], [bounds[k][1] for k in free]),
                           ftol=ftol, xtol=xtol)
    perr = np.sqrt(np.diag(pcov))
    values = {**{k: float(v) for k, v in fixed.items()}, **dict(zip(free, map(float, popt)))}
    errors = {**{k: np.nan for k in fixed}, **dict(zip(free, map(float, perr)))}
    return values, errors, f(t, *popt)


def _free_and_fixed(gamma_0, gamma_inf, tau_guess, tau_bounds):
    """Split gamma_0 / gamma_inf into fitted (None) and fixed (a number); tau is always fitted."""
    guesses, bounds, fixed = {}, {}, {}
    for name, value, guess in (('gamma_0', gamma_0, GAMMA_0_GUESS), ('gamma_inf', gamma_inf, GAMMA_INF_GUESS)):
        if value is None:
            guesses[name], bounds[name] = guess, (0, 1)
        else:
            fixed[name] = float(value)
    guesses['tau'], bounds['tau'] = tau_guess, tuple(tau_bounds)
    return fixed, guesses, bounds


def _result(values, errors, y, y_hat):
    """Flat dict: the parameters, the goodness of fit and ``<param>_err`` for every parameter."""
    return {**values, **fit_metrics(y, y_hat), **{f'{k}_err': v for k, v in errors.items()}}


def fit_coh_decay_model(cohs, days, tau_guess=6, tau_bounds=(0, 100), xtol=1e-6, ftol=1e-6, gamma_0=1.0,
                        gamma_inf=None):
    """
    Fit the decay model to one pixel's coherence time series.

    cohs : 1D array of coherence values
    days : 1D array of time differences, same length as ``cohs``
    tau_guess : starting value of tau
    tau_bounds : (lower, upper) bounds of tau, in the units of ``days``
    xtol, ftol : curve_fit tolerances
    gamma_0 : coherence at zero time difference; held fixed at this value (default 1), or fit if None
    gamma_inf : asymptotic coherence; held fixed if given, fit if None. A fitted gamma_0 / gamma_inf is bounded to [0, 1]

    Returns a dict with ``gamma_0``, ``gamma_inf``, ``tau`` (fitted or fixed), the goodness of the whole fit
    (``nse``, ``kge``, ``pearson_r``) and the 1-sigma errors ``gamma_0_err``, ``gamma_inf_err``, ``tau_err``
    (NaN for a fixed parameter).
    Raises (RuntimeError, ValueError) if the fit fails, as curve_fit does.
    """
    fixed, guesses, bounds = _free_and_fixed(gamma_0, gamma_inf, tau_guess, tau_bounds)
    values, errors, y_hat = _fit(decorrelation_temporal_model, days, cohs, fixed, guesses, bounds, xtol, ftol)
    return _result(values, errors, cohs, y_hat)


def fit_coh_decay_model_snow(cohs, days, swe, inc, tau_guess=6, tau_bounds=(0, 100), xtol=1e-6, ftol=1e-6,
                             gamma_0=1.0, gamma_inf=None, kappa_guess=KAPPA_GUESS, kappa_bounds=(0, np.inf)):
    """
    Fit the snow decay model to one pixel's coherence time series; arguments as ``fit_coh_decay_model`` plus

    swe : 1D array, snow water equivalent of each pair (same length as ``cohs``)
    inc : local incidence angle of this pixel in degrees (a number, or 1D like ``swe``)
    kappa_guess, kappa_bounds : start value and (lower, upper) bounds of the snow sensitivity kappa (>= 0 by default)

    The result also has ``kappa`` and ``kappa_err``.
    """
    fixed, guesses, bounds = _free_and_fixed(gamma_0, gamma_inf, tau_guess, tau_bounds)
    guesses['kappa'], bounds['kappa'] = kappa_guess, tuple(kappa_bounds)

    def model(t, **params):
        return decorrelation_temporal_model_snow(t, swe, inc, **params)

    values, errors, y_hat = _fit(model, days, cohs, fixed, guesses, bounds, xtol, ftol)
    return _result(values, errors, cohs, y_hat)


# Optional per-pixel maps, requested by name (the parameter 1-sigma errors). ``kappa_err`` needs the snow model.
METRICS = ('tau_err', 'gamma_inf_err', 'gamma_0_err', 'kappa_err')

# Maps that are always returned (``kappa`` is added with the snow model).
BASE_MAPS = ('tau', 'gamma_inf', 'gamma_0', 'nse', 'kge', 'pearson_r')


def get_taus_spatial(coh_arr, days, tau_guess=6, tau_bounds=(0, 100), xtol=1e-6, ftol=1e-6, gamma_0=1.0,
                     gamma_inf=None, min_pairs=10, metrics=(), swe_arr=None, inc=None,
                     kappa_guess=KAPPA_GUESS, kappa_bounds=(0, np.inf)):
    """
    Fit the decay model at every pixel of a block.

    coh_arr : 3D array (n_pairs, ny, nx)
    days : 1D array (n_pairs,) of time differences
    gamma_0, gamma_inf : fixed if a number, fit per pixel if None (see ``fit_coh_decay_model``)
    min_pairs : a pixel with fewer valid pairs than this is not fit (it is raised to the number of fitted
        parameters + 1, the least that determines them)
    metrics : extra maps to return, any of ``METRICS``; none by default
    swe_arr, inc : give both to fit the snow model instead: ``swe_arr`` is (n_pairs, ny, nx) and ``inc`` is the
        (ny, nx) local incidence angle in degrees. A pair is used only where coherence, swe and inc are all finite.
    kappa_guess, kappa_bounds : start value and bounds of kappa (snow model only)

    Returns a dict of (ny, nx) arrays: ``tau``, ``gamma_inf``, ``gamma_0``, ``nse``, ``kge``, ``pearson_r``, ``kappa``
    (snow model only) and any requested metrics (float32, NaN where not fit; fixed parameters are filled in where a
    pixel was fit), ``n_pairs`` (int16, valid pairs per pixel) and ``success`` (bool).
    """
    snow = swe_arr is not None
    if snow != (inc is not None):
        raise ValueError('swe_arr and inc must be given together (snow model) or both left out')
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f'unknown metrics {sorted(unknown)}; choose from {METRICS}')
    if 'kappa_err' in metrics and not snow:
        raise ValueError("'kappa_err' needs the snow model (swe_arr and inc)")
    days = np.asarray(days, dtype='float64')
    n, ny, nx = coh_arr.shape
    if days.shape != (n,):
        raise ValueError(f'days has shape {days.shape} but coh_arr has {n} pairs')

    valid = np.isfinite(coh_arr)
    if snow:
        if swe_arr.shape != coh_arr.shape:
            raise ValueError(f'swe_arr has shape {swe_arr.shape} but coh_arr has {coh_arr.shape}')
        inc = np.asarray(inc, dtype='float64')
        if inc.shape != (ny, nx):
            raise ValueError(f'inc has shape {inc.shape} but the block is {(ny, nx)}')
        valid = valid & np.isfinite(swe_arr) & np.isfinite(inc)[None]

    n_free = 1 + (gamma_0 is None) + (gamma_inf is None) + snow
    min_pairs = max(min_pairs, n_free + 1)

    maps = (*BASE_MAPS, *(('kappa',) if snow else ()))
    out = {k: np.full((ny, nx), np.nan, dtype='float32') for k in (*maps, *metrics)}
    out['n_pairs'] = valid.sum(axis=0).astype('int16')
    out['success'] = np.zeros((ny, nx), dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            ok = valid[:, iy, ix]
            if out['n_pairs'][iy, ix] < min_pairs:
                continue
            try:
                if snow:
                    res = fit_coh_decay_model_snow(
                        coh_arr[ok, iy, ix], days[ok], swe_arr[ok, iy, ix], inc[iy, ix], tau_guess, tau_bounds,
                        xtol, ftol, gamma_0, gamma_inf, kappa_guess, kappa_bounds)
                else:
                    res = fit_coh_decay_model(coh_arr[ok, iy, ix], days[ok], tau_guess, tau_bounds, xtol, ftol,
                                              gamma_0, gamma_inf)
            except (RuntimeError, ValueError):  # no convergence, or bad input for this pixel: leave it as not fit
                continue
            for k in (*maps, *metrics):
                out[k][iy, ix] = res[k]
            out['success'][iy, ix] = True
    return out
