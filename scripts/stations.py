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


def read_mesowest(
    files: List[Union[str, Path]], names: Dict[str, str], utc_offset_hours: float
) -> xr.DataArray:
    """
    Read MesoWest CSVs (UTC timestamps, ``wind_speed_set_1`` in m/s) into an
    hourly-mean ``(station_id, datetime)`` array.

    Parameters
    ----------
    files : list of str | Path
        MesoWest station CSVs (units row directly under the header).
    names : dict
        Maps ``Station_ID`` to a friendly name (stored as ``station_name``).
    utc_offset_hours : float
        Fixed offset from UTC (no DST) applied to the timestamps, e.g. ``-7``
        for MST. Required; use the same value as ``build_cube``.

    Returns
    -------
    xarray.DataArray
        Named ``'wind_speed'`` (m/s), tz-naive ``datetime`` in local standard
        time, NaN where a station has no observation.
    """
    out = []
    for f in files:
        ds = pd.read_csv(f, comment='#')
        ds = ds.drop(ds.index[0])  # units row

        t = pd.to_datetime(ds['Date_Time'], utc=True)
        wind = pd.Series(ds['wind_speed_set_1'].astype(float).values, index=pd.DatetimeIndex(t))
        wind = wind[~wind.index.duplicated(keep='first')].sort_index().resample('1h').mean()
        wind.index = (wind.index + pd.Timedelta(hours=utc_offset_hours)).tz_localize(None).rename('datetime')

        stid = ds['Station_ID'].iloc[0]
        da = xr.DataArray(wind.values, coords={'datetime': wind.index}, dims='datetime')
        da = da.expand_dims(station_id=[stid])
        da = da.assign_coords(station_name=('station_id', [names.get(stid, stid)]))
        out.append(da)

    wind = xr.concat(out, dim='station_id', join='outer')
    wind.name = 'wind_speed'
    wind.attrs = {'units': 'm/s', 'long_name': 'Wind speed', 'utc_offset_hours': utc_offset_hours}
    return wind


def read_snotel(
    files: List[Union[str, Path]], names: Dict[str, str], utc_offset_hours: float,
    variable_prefix: str = 'WTEQ',
) -> xr.Dataset:
    """
    Read SNOTEL CSVs (``{station_id}_STAND_WATERYEAR_*.csv``, timestamps in
    fixed UTC-8) into an hourly-mean ``(station_id, datetime)`` dataset.

    Parameters
    ----------
    files : list of str | Path
        SNOTEL station CSVs; the station id is the filename prefix before ``_``.
    names : dict
        Maps station id to a friendly name (stored as ``station_name``).
    utc_offset_hours : float
        Fixed offset from UTC (no DST) the timestamps are shifted to, e.g.
        ``-7`` for MST. Required; use the same value as ``build_cube``.
    variable_prefix : str, default='WTEQ'
        Column prefix to read (snow water equivalent). Values in inches are
        converted to mm.

    Returns
    -------
    xarray.Dataset
        ``swe`` (mm, start of day) and ``swe_accum`` (``swe.diff('datetime')``),
        tz-naive ``datetime`` in local standard time.
    """
    out = []
    for f in files:
        stid = Path(f).name.split('_')[0]
        ds = pd.read_csv(f, header=1).replace(-99.9, np.nan)

        col = next(c for c in ds.columns if c.startswith(variable_prefix))
        vals = ds[col].astype(float).values
        if '(in)' in col:
            vals = vals * IN_TO_MM

        t = pd.DatetimeIndex(pd.to_datetime(ds['Date'] + ' ' + ds['Time']))
        t = t + pd.Timedelta(hours=utc_offset_hours - SNOTEL_UTC_OFFSET_HOURS)
        s = pd.Series(vals, index=t.rename('datetime'))
        s = s[~s.index.duplicated(keep='first')].sort_index().resample('1h').mean()

        da = xr.DataArray(s.values, coords={'datetime': s.index}, dims='datetime')
        da = da.expand_dims(station_id=[stid])
        da = da.assign_coords(station_name=('station_id', [names.get(stid, stid)]))
        out.append(da)

    swe = xr.concat(out, dim='station_id', join='outer')
    swe.name = 'swe'
    swe.attrs = {'units': 'mm', 'long_name': 'Snow water equivalent (start of day)',
                 'utc_offset_hours': utc_offset_hours}
    swe = swe.to_dataset()
    swe['swe_accum'] = swe['swe'].diff(dim='datetime')
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
