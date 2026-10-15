import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

# default fSCA class edges (%) and names: <20 no snow, 20-70 patchy, >=70 snow
DEFAULT_BINS = (20, 70)
DEFAULT_LABELS = ("no snow", "patchy snow", "snow")

NODATA = -1


def build_fsca_cube(
    files: List[Union[str, Path]],
    aoi: Tuple[float, float, float, float],
    crs,
    var: str = "snow_fraction",
    res: Optional[float] = None,
) -> xr.DataArray:
    """
    Stack daily SPIRES .nc files into a (time, y, x) fSCA DataArray (percent) cropped to `aoi`.

    aoi : (xmin, ymin, xmax, ymax) in `crs`, e.g. nisar_80m.rio.bounds()
    crs : target CRS (e.g. nisar_80m.rio.crs); output is reprojected to it
    res : output resolution in crs units; default is the native SPIRES pixel size (~463 m)
    """
    crs = CRS.from_user_input(crs)
    xmin, ymin, xmax, ymax = aoi

    # crop in the native sinusoidal grid first (cheap); buffer so reprojection has edge pixels
    das = []
    for f in sorted(files):
        with xr.open_dataset(f) as ds:
            da = ds[var]
            sin = CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
            if not das:
                tf = Transformer.from_crs(crs, sin, always_xy=True)
                xs, ys = tf.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
                buf = 2 * 463.3
                xsl = (min(xs) - buf, max(xs) + buf)
                ysl = (max(ys) + buf, min(ys) - buf)  # y descends
            da = da.sel(x=slice(*xsl), y=slice(*ysl)).squeeze("time", drop=True)
            if da.sizes["x"] == 0 or da.sizes["y"] == 0:
                raise ValueError(f"aoi is outside the tile covered by {Path(f).name}")
            da = da.expand_dims(time=[pd.Timestamp(ds["time"].values[0]).normalize()]).load()
        da.attrs = {}
        das.append(da.rio.write_crs(sin))

    cube = xr.concat(das, dim="time")
    cube = cube.rio.reproject(crs, resolution=res, resampling=Resampling.nearest, nodata=np.nan)
    cube = cube.rio.clip_box(xmin, ymin, xmax, ymax)
    cube.name = "fsca"
    cube.attrs = {"long_name": "fractional snow covered area", "units": "%"}
    logger.info("fsca cube: %s from %s to %s", dict(cube.sizes), cube.time.values[0], cube.time.values[-1])
    return cube


def classify_fsca(
    fsca: xr.DataArray,
    bins: Sequence[float] = DEFAULT_BINS,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> xr.DataArray:
    """
    Assign integer snow-cover classes from fSCA (%). `bins` are the lower class edges beyond the first,
    so bins=(20, 70) gives 0: fsca < 20, 1: 20 <= fsca < 70, 2: fsca >= 70. NaN -> -1.
    """
    if len(labels) != len(bins) + 1:
        raise ValueError("need len(labels) == len(bins) + 1")
    cat = xr.apply_ufunc(np.digitize, fsca, kwargs={"bins": np.asarray(bins, dtype=float)}, dask="parallelized", output_dtypes=[np.int8])
    cat = cat.astype(np.int8).where(fsca.notnull(), NODATA)
    cat.name = "snow_class"
    cat.attrs = {"long_name": "categorical snow cover", "bins": list(bins), "class_labels": list(labels), "nodata": NODATA}
    return cat


def change_labels(class_labels: Sequence[str]) -> Dict[int, str]:
    """Names for the codes returned by classify_change."""
    n = len(class_labels)
    out = {c: f"{name} both days" for c, name in enumerate(class_labels)}
    for d in range(1, n):
        out[n + d - 1] = f"increase ({d} class{'es' if d > 1 else ''})"
        out[n + (n - 1) + d - 1] = f"decrease ({d} class{'es' if d > 1 else ''})"
    return out


def classify_change(
    cat: xr.DataArray,
    starts,
    ends,
    pair: Optional[Sequence] = None,
    tolerance: str = "1D",
) -> xr.DataArray:
    """
    Snow-cover change between the start and end date of each pair (from nisar_80m).

    With N classes and d = class_end - class_start:
      d == 0 -> class code 0..N-1 (stable in that class; e.g. 0 = no snow both days)
      d > 0  -> N + (d - 1)            (increase by d classes)
      d < 0  -> 2N - 1 + (|d| - 1)     (decrease by |d| classes)
    For the default 3 classes: 3/4 = increase by 1/2, 5/6 = decrease by 1/2. Either day missing -> -1.
    Dims (pair, y, x); pair/start/end coords match the inputs.
    """
    starts = pd.DatetimeIndex(starts).normalize()
    ends = pd.DatetimeIndex(ends).normalize()
    n = len(cat.attrs["class_labels"])
    pair = np.arange(len(starts)) if pair is None else np.asarray(pair)

    def pick(dates):
        out = cat.sel(time=xr.DataArray(dates, dims="pair"), method="nearest", tolerance=tolerance)
        return out.drop_vars("time").assign_coords(pair=pair)

    c1, c2 = pick(starts), pick(ends)
    d = c2 - c1
    code = xr.where(d == 0, c1, xr.where(d > 0, n + d - 1, 2 * n - 1 + (-d) - 1))
    code = code.where((c1 != NODATA) & (c2 != NODATA), NODATA).astype(np.int8)
    code = code.assign_coords(start=("pair", starts.values), end=("pair", ends.values))
    code.name = "snow_change"
    code.attrs = {"long_name": "categorical snow cover change", "nodata": NODATA, "change_labels": change_labels(cat.attrs["class_labels"])}
    return code

