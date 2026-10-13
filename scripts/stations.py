import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

SNOTEL_UTC_OFFSET_HOURS: int = -8
"""SNOTEL timestamps are in fixed Pacific Standard Time (UTC-8, no DST)."""

IN_TO_MM: float = 25.4

MESOWEST_VARS: dict = {'wind_speed': 'wind_speed_set_1', 'air_temp': 'air_temp_set_1'}
"""Maps output variable names to MesoWest CSV columns (m/s and degC)."""

SNOTEL_VARS: dict = {'swe': 'WTEQ', 'air_temp': 'TOBS'}
"""Maps output variable names to SNOTEL CSV column prefixes."""


def read_mesowest(
    files: List[Union[str, Path]], names: Dict[str, str], utc_offset_hours: float
) -> xr.Dataset:
    """
    Read MesoWest CSVs (UTC timestamps) into an hourly-mean
    ``(station_id, datetime)`` dataset of wind speed and air temperature.

    Parameters
    ----------
    files : list of str | Path
        MesoWest station CSVs (units row directly under the header) with
        ``wind_speed_set_1`` (m/s) and ``air_temp_set_1`` (degC) columns.
    names : dict
        Maps ``Station_ID`` to a friendly name (stored as ``station_name``).
    utc_offset_hours : float
        Fixed offset from UTC (no DST) applied to the timestamps, e.g. ``-7``
        for MST. Required; use the same value as ``build_cube``.

    Returns
    -------
    xarray.Dataset
        ``wind_speed`` (m/s) and ``air_temp`` (degC), tz-naive ``datetime`` in
        local standard time, NaN where a station has no observation.
    """
    out = []
    for f in files:
        ds = pd.read_csv(f, comment='#')
        ds = ds.drop(ds.index[0])  # units row

        t = pd.DatetimeIndex(pd.to_datetime(ds['Date_Time'], utc=True))
        df = pd.DataFrame({k: ds[c].astype(float).values for k, c in MESOWEST_VARS.items()}, index=t)
        df = df[~df.index.duplicated(keep='first')].sort_index().resample('1h').mean()
        df.index = (df.index + pd.Timedelta(hours=utc_offset_hours)).tz_localize(None).rename('datetime')

        stid = ds['Station_ID'].iloc[0]
        da = xr.Dataset({k: ('datetime', df[k].values) for k in df}, coords={'datetime': df.index})
        da = da.expand_dims(station_id=[stid])
        da = da.assign_coords(station_name=('station_id', [names.get(stid, stid)]))
        out.append(da)

    wind = xr.concat(out, dim='station_id', join='outer')
    wind['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    wind['air_temp'].attrs = {'units': 'degC', 'long_name': 'Air temperature'}
    wind.attrs = {'utc_offset_hours': utc_offset_hours}
    return wind


def read_snotel(
    files: List[Union[str, Path]], names: Dict[str, str], utc_offset_hours: float
) -> xr.Dataset:
    """
    Read SNOTEL CSVs (``{station_id}_STAND_WATERYEAR_*.csv``, timestamps in
    fixed UTC-8) into an hourly-mean ``(station_id, datetime)`` dataset.

    Parameters
    ----------
    files : list of str | Path
        SNOTEL station CSVs; the station id is the filename prefix before ``_``.
        Snow water equivalent (``WTEQ``, in or mm) and air temperature
        (``TOBS``, degC or degF) columns are read.
    names : dict
        Maps station id to a friendly name (stored as ``station_name``).
    utc_offset_hours : float
        Fixed offset from UTC (no DST) the timestamps are shifted to, e.g.
        ``-7`` for MST. Required; use the same value as ``build_cube``.

    Returns
    -------
    xarray.Dataset
        ``swe`` (mm, start of day), ``swe_accum`` (``swe.diff('datetime')``)
        and ``air_temp`` (degC), tz-naive ``datetime`` in local standard time.
    """
    out = []
    for f in files:
        stid = Path(f).name.split('_')[0]
        ds = pd.read_csv(f, header=1).replace(-99.9, np.nan)

        cols = {}
        for k, prefix in SNOTEL_VARS.items():
            col = next(c for c in ds.columns if c.startswith(prefix))
            vals = ds[col].astype(float).values
            if '(in)' in col:
                vals = vals * IN_TO_MM
            if '(degF)' in col:
                vals = (vals - 32) * 5 / 9
            cols[k] = vals

        t = pd.DatetimeIndex(pd.to_datetime(ds['Date'] + ' ' + ds['Time']))
        t = t + pd.Timedelta(hours=utc_offset_hours - SNOTEL_UTC_OFFSET_HOURS)
        df = pd.DataFrame(cols, index=t.rename('datetime'))
        df = df[~df.index.duplicated(keep='first')].sort_index().resample('1h').mean()

        da = xr.Dataset({k: ('datetime', df[k].values) for k in df}, coords={'datetime': df.index})
        da = da.expand_dims(station_id=[stid])
        da = da.assign_coords(station_name=('station_id', [names.get(stid, stid)]))
        out.append(da)

    swe = xr.concat(out, dim='station_id', join='outer')
    swe['swe_accum'] = swe['swe'].diff(dim='datetime')
    swe['swe'].attrs = {'units': 'mm', 'long_name': 'Snow water equivalent (start of day)'}
    swe['swe_accum'].attrs = {'units': 'mm', 'long_name': 'Snow water equivalent accumulation'}
    swe['air_temp'].attrs = {'units': 'degC', 'long_name': 'Air temperature'}
    swe.attrs = {'utc_offset_hours': utc_offset_hours}
    return swe


def event_wind_by_pair(
    swe_accum: xr.DataArray, wind: xr.DataArray, starts, ends, threshold: float,
    hours_after: float = 24, freq: str = '6h',
) -> xr.DataArray:
    """
    Mean wind speed during snow-accumulation events, per interferometric pair.

    Station-mean ``swe_accum`` is summed to ``freq`` bins; bins above
    ``threshold`` are events. Each event covers the bin start plus
    ``hours_after`` hours, clipped to the pair's end. The result is the
    station-mean wind (averaged to ``freq``) over the union of event windows
    inside each pair, 0 where a pair has no events.

    Parameters
    ----------
    swe_accum : xarray.DataArray
        SWE accumulation (mm) with ``station_id`` and ``datetime`` dims
        (``read_snotel(...)['swe_accum']``).
    wind : xarray.DataArray
        Wind speed (m/s) with ``station_id`` and ``datetime`` dims
        (``read_mesowest``). Must use the same ``utc_offset_hours`` as
        ``swe_accum``, ``starts`` and ``ends``.
    starts, ends : array-like of datetime64
        Start/end of each pair (e.g. a ``build_cube`` result's ``start``/``end``).
    threshold : float
        Accumulation (mm per ``freq`` bin) above which a bin is an event.
    hours_after : float, default=24
        Hours after each event bin that count as part of the event.
    freq : str, default='6h'
        Time bin used for events and wind.

    Returns
    -------
    xarray.DataArray
        Named ``'snow_wind_mps'`` on dim ``pair`` (``'YYYYMMDD_YYYYMMDD'``).
    """
    swe_bins = swe_accum.mean('station_id').resample(datetime=freq).sum()
    wind_bins = wind.mean('station_id').resample(datetime=freq).mean()

    event_times = swe_bins['datetime'].values[(swe_bins > threshold).values]
    times = wind_bins['datetime'].values
    after = np.timedelta64(int(hours_after * 3600), 's')

    vals, pairs = [], []
    for start, end in zip(starts, ends):
        start, end = np.datetime64(start), np.datetime64(end)
        mask = np.zeros(times.shape, dtype=bool)
        for t in event_times[(event_times >= start) & (event_times <= end)]:
            mask |= (times >= t) & (times <= min(t + after, end))
        w = wind_bins.values[mask]
        vals.append(float(np.nanmean(w)) if np.isfinite(w).any() else 0.0)
        pairs.append(f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}")

    return xr.DataArray(vals, dims='pair', coords={'pair': pairs}, name='snow_wind_mps')


def pair_metrics(
    swe_accum: xr.DataArray, wind_speed: xr.DataArray, air_temp: xr.DataArray, starts, ends, threshold: float, event_freq: str,
) -> xr.Dataset:
    """
    Snow accumulation and wind metrics for each interferometric pair, using the
    station mean of ``swe_accum`` and ``wind_speed`` over the pair.

    Parameters
    ----------
    swe_accum : xarray.DataArray
        SWE accumulation (mm) with ``station_id`` and ``datetime`` dims.
    wind_speed : xarray.DataArray
        Wind speed (m/s) with ``station_id`` and ``datetime`` dims. Must use the
        same ``utc_offset_hours`` as ``swe_accum``, ``starts`` and ``ends``.
    starts, ends : array-like of datetime64
        Start/end of each pair (e.g. a ``build_cube`` result's ``start``/``end``).
    threshold, event_freq
        Event definition passed to ``event_wind_by_pair`` (``freq=event_freq``).

    Returns
    -------
    xarray.Dataset
        On dim ``pair`` (``'YYYYMMDD_YYYYMMDD'``): ``swe_tot`` (sum of positive
        accumulation, mm), ``swe_max_6h`` / ``swe_max_12h`` / ``swe_max_1d``
        (largest accumulation in a 6-hour / 12-hour / 1-day bin, mm),
        ``wind_mean`` (m/s), ``wind_event`` (mean wind during accumulation
        events, m/s), ``swe_x_wind`` (``swe_tot * wind_mean``) and
        ``swe_max_6h_x_wind_event`` (``swe_max_6h * wind_event``).
    """
    swe_mean = swe_accum.mean(dim='station_id')
    wind_mean = wind_speed.mean(dim='station_id')
    air_temp_mean = air_temp.mean(dim='station_id')

    metrics = {k: [] for k in ('swe_tot', 'swe_loss', 'swe_max_6h', 'swe_max_12h', 'swe_max_1d', 'wind_mean', 'temp_mean', 'degree_days')}
    pairs = []
    for start, end in zip(starts, ends):
        start, end = np.datetime64(start), np.datetime64(end)
        swe_pair = swe_mean.sel(datetime=slice(start, end))
        wind_pair = wind_mean.sel(datetime=slice(start, end))
        temp_pair = air_temp_mean.sel(datetime=slice(start, end))

        metrics['swe_tot'].append(float(np.nansum(swe_pair)))#.where(swe_pair > 0).values)))
        metrics['swe_loss'].append(float(np.nansum(swe_pair.where(swe_pair < 0).values)))
        for f in ('6h', '12h', '1d'):
            metrics[f'swe_max_{f}'].append(float(np.nanmax(swe_pair.resample(datetime=f).sum().values)))
        metrics['wind_mean'].append(float(np.nanmean(wind_pair.values)))
        metrics['temp_mean'].append(float(np.nanmean(temp_pair.values)))
        pairs.append(f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}")
        metrics['degree_days'].append(float(np.nansum(temp_pair.where(temp_pair > 0).values)))

    out = xr.Dataset({k: ('pair', v) for k, v in metrics.items()}, coords={'pair': pairs})
    out['swe_x_wind'] = out['swe_tot'] * out['wind_mean']
    for f in ('6h', '12h', '1d'):
        out[f'wind_event_{f}'] = event_wind_by_pair(swe_accum, wind_speed, starts, ends, threshold, freq=f)
        out[f'swe_max_{f}_x_wind_event'] = out[f'swe_max_{f}'] * out[f'wind_event_{f}']
    return out
