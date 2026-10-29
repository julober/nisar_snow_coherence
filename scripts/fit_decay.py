"""
Fit the temporal coherence decay model per pixel over one or more track Zarr cubes, in spatial tiles, so the work
can be split over a Slurm job array (see ``slurm/fit_decay.sbatch``).

Input: the ``gunw_<res>.zarr`` stores written by ``scripts/gunw_zarr.py`` (dims ``(pair, y, x)``, coords ``start``
and ``end``). With several stores (one per track) all pairs are stacked and fit together; the tracks must be on the
same x/y grid or the script stops. ``--var`` must be the name of the variable in the stores.

The date window and ``--max-days`` are applied to every track: a pair is kept if its reference start is on or after
``--start-date``, its secondary start is on or before ``--end-date`` and (if given) it spans at most ``--max-days``.

Output: one Zarr with (y, x) maps ``tau`` (days), ``gamma_inf``, ``tau_err``, ``gamma_inf_err``, ``n_pairs`` and
``success``. Its chunks are the tiles, so array tasks never write the same chunk.

Steps (use the same data/fit arguments for all three, they define which pairs are fit):
    python scripts/fit_decay.py --init --zarr A.zarr B.zarr --var coherence_80m --output out.zarr   # once, prints tiles
    python scripts/fit_decay.py --tile-index 0 --zarr A.zarr B.zarr --var coherence_80m --output out.zarr
    python scripts/fit_decay.py --finalize --output out.zarr                                         # GeoTIFFs
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

# Allow `python scripts/fit_decay.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.taus import get_taus_spatial  # noqa: E402

logger = logging.getLogger(__name__)

# Output variable -> dtype; all are (y, x). NaN (or 0 / False) marks pixels that have not been fit.
OUT_VARS = {'tau': 'float32', 'gamma_inf': 'float32', 'tau_err': 'float32', 'gamma_inf_err': 'float32',
            'n_pairs': 'int16', 'success': 'bool'}


def select_pairs(ds: xr.Dataset, start_date, end_date, max_days) -> np.ndarray:
    """Positions along ``pair`` of the pairs inside the date window / temporal-baseline limit."""
    start, end = ds['start'].values, ds['end'].values
    keep = np.ones(start.shape, dtype=bool)
    if start_date is not None:
        keep &= start >= np.datetime64(start_date)
    if end_date is not None:
        # A bare end date means the whole day, so the cutoff is the start of the next day.
        keep &= end < np.datetime64(end_date) + np.timedelta64(1, 'D')
    if max_days is not None:
        keep &= (end - start) / np.timedelta64(1, 'D') <= max_days
    return np.flatnonzero(keep)


def open_tracks(stores: List[Path], var: str, start_date, end_date, max_days):
    """
    Open the stores lazily and select pairs. Returns ``(das, days, grid)``: one lazy (pair, y, x) DataArray per
    track (pairs already selected), the temporal baseline in days of all selected pairs (tracks in order), and the
    first store's dataset (for x, y, crs).
    """
    das, days, first = [], [], None
    for store in stores:
        ds = xr.open_zarr(store)
        if var not in ds:
            raise KeyError(f'{store} has no variable {var!r}; it has {list(ds.data_vars)}')
        if first is None:
            first = ds
        elif not (ds['x'].shape == first['x'].shape and ds['y'].shape == first['y'].shape
                  and np.array_equal(ds['x'].values, first['x'].values)
                  and np.array_equal(ds['y'].values, first['y'].values)):
            raise ValueError(f'{store} is not on the same x/y grid as {stores[0]}')
        idx = select_pairs(ds, start_date, end_date, max_days)
        logger.info(f'{store}: {len(idx)} of {ds.sizes["pair"]} pairs selected')
        das.append(ds[var].isel(pair=idx))
        days.append(((ds['end'].values - ds['start'].values) / np.timedelta64(1, 'D'))[idx])
    days = np.concatenate(days)
    if days.size == 0:
        raise ValueError('no pairs left after the date / max-days selection')
    return das, days, first


def n_tiles(ny: int, nx: int, tile: int) -> int:
    return -(-ny // tile) * -(-nx // tile)


def tile_slices(index: int, ny: int, nx: int, tile: int):
    """(y slice, x slice) of tile ``index``; tiles are numbered row by row."""
    n_x = -(-nx // tile)
    iy, ix = divmod(index, n_x)
    return slice(iy * tile, min((iy + 1) * tile, ny)), slice(ix * tile, min((ix + 1) * tile, nx))


def init_output(a, days, grid) -> int:
    """Create the empty output Zarr (fixed shape and chunks) and return its number of tiles."""
    import dask.array as da
    ny, nx = grid.sizes['y'], grid.sizes['x']
    fill = {'float32': np.nan, 'int16': 0, 'bool': False}
    data = {k: (('y', 'x'), da.full((ny, nx), fill[t], dtype=t, chunks=(a.tile_size, a.tile_size)))
            for k, t in OUT_VARS.items()}
    out = xr.Dataset(data, coords={'y': grid['y'].values, 'x': grid['x'].values})
    out.attrs.update(
        crs=grid.attrs.get('crs', ''), var=a.var, n_pairs_fit=int(days.size), tile_size=a.tile_size,
        stores=[str(s) for s in a.zarr], start_date=str(a.start_date), end_date=str(a.end_date),
        max_days=str(a.max_days), gamma_inf='fit' if a.gamma_inf is None else a.gamma_inf,
        tau_guess=a.tau_guess, tau_bounds=list(a.tau_bounds), min_pairs=a.min_pairs,
    )
    out.to_zarr(a.output, mode='w', compute=False, consolidated=True)
    return n_tiles(ny, nx, a.tile_size)


def fit_tile(a) -> None:
    """Fit tile ``a.tile_index`` and write it into its region of the output Zarr."""
    das, days, grid = open_tracks(a.zarr, a.var, a.start_date, a.end_date, a.max_days)
    ny, nx = grid.sizes['y'], grid.sizes['x']
    total = n_tiles(ny, nx, a.tile_size)
    if not 0 <= a.tile_index < total:
        sys.exit(f'--tile-index {a.tile_index} is outside 0..{total - 1}')
    expected = xr.open_zarr(a.output).attrs.get('n_pairs_fit')
    if expected != days.size:  # a different pair set than at --init would give maps that mix two selections
        sys.exit(f'{days.size} pairs selected now but {expected} at --init; use the same arguments as --init')

    ys, xs = tile_slices(a.tile_index, ny, nx, a.tile_size)
    block = np.concatenate([da.isel(y=ys, x=xs).values for da in das], axis=0)
    logger.info(f'tile {a.tile_index}/{total - 1}: y {ys.start}:{ys.stop}, x {xs.start}:{xs.stop}, '
                f'{block.shape[0]} pairs')
    res = get_taus_spatial(block, days, tau_guess=a.tau_guess, tau_bounds=tuple(a.tau_bounds),
                           gamma_inf=a.gamma_inf, min_pairs=a.min_pairs)
    # Only data variables, no coords: a region write must not carry the full y/x index.
    xr.Dataset({k: (('y', 'x'), res[k]) for k in OUT_VARS}).to_zarr(
        a.output, region={'y': ys, 'x': xs}, consolidated=True)
    logger.info(f'tile {a.tile_index}: fit {int(res["success"].sum())} of {res["success"].size} pixels')


def finalize(output: Path) -> None:
    """Write each output map as a GeoTIFF in ``<output stem>_tifs/``."""
    import rioxarray  # noqa: F401
    ds = xr.open_zarr(output).load()
    ds = ds.rio.write_crs(ds.attrs['crs'])
    out_dir = output.with_name(output.stem + '_tifs')
    out_dir.mkdir(exist_ok=True)
    for k in OUT_VARS:
        da = ds[k].astype('uint8') if k == 'success' else ds[k]  # GeoTIFF has no bool type
        da.rio.to_raster(out_dir / f'{k}.tif')
    logger.info(f'wrote {len(OUT_VARS)} GeoTIFFs to {out_dir}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('Steps')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--init', action='store_true', help='create the output Zarr, print the number of tiles')
    mode.add_argument('--tile-index', type=int, help='fit this tile (use $SLURM_ARRAY_TASK_ID)')
    mode.add_argument('--n-tiles', action='store_true', help='print the number of tiles and exit')
    mode.add_argument('--finalize', action='store_true', help='write GeoTIFFs from the finished output Zarr')
    ap.add_argument('--zarr', nargs='+', type=Path, help='input store(s), one per track, same grid')
    ap.add_argument('--var', help='variable to fit, as named in the stores (e.g. coherence_80m)')
    ap.add_argument('--output', required=True, type=Path, help='output Zarr')
    ap.add_argument('--start-date', default=None, help='keep pairs whose reference starts on/after this (YYYY-MM-DD)')
    ap.add_argument('--end-date', default=None, help='keep pairs whose secondary starts on/before this (YYYY-MM-DD)')
    ap.add_argument('--max-days', type=float, default=None, help='keep pairs up to this many days apart')
    ap.add_argument('--gamma-inf', type=float, default=None,
                    help='fix gamma_inf at this value (default: fit it per pixel)')
    ap.add_argument('--tau-guess', type=float, default=6, help='starting tau, days (default: %(default)s)')
    ap.add_argument('--tau-bounds', nargs=2, type=float, default=[0, 30], metavar=('LO', 'HI'),
                    help='bounds on tau, days (default: %(default)s)')
    ap.add_argument('--min-pairs', type=int, default=10, help='fewest valid pairs to fit a pixel (default: %(default)s)')
    ap.add_argument('--tile-size', type=int, default=256, help='tile side, pixels (default: %(default)s)')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    if a.finalize:
        finalize(a.output)
        return
    if not a.zarr or not a.var:
        ap.error('--zarr and --var are required except with --finalize')
    if a.tile_index is not None:
        fit_tile(a)
        return
    das, days, grid = open_tracks(a.zarr, a.var, a.start_date, a.end_date, a.max_days)
    if a.n_tiles:
        print(n_tiles(grid.sizes['y'], grid.sizes['x'], a.tile_size))
        return
    total = init_output(a, days, grid)
    logger.info(f'{days.size} pairs from {len(a.zarr)} track(s); output {a.output}')
    print(f'{total} tiles: submit with --array=0-{total - 1}')


if __name__ == '__main__':
    main()
