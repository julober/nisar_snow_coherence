"""
Build per-track Zarr stores of GUNW coherence, ready for fitting a temporal decorrelation model.

Reads the ``GUNW_product.h5`` files written by ``submit_gunw_pairs.sh`` (one per date pair,
``<gunw-root>/<ref>_<sec>_<ORBIT>/GUNW_product.h5``) and writes two Zarr stores per track, because the
20 m and 80 m coherence layers are on different grids:

    <out-dir>/track<TTT>/gunw_20m.zarr   variable ``coherence_20m``, dims (pair, y, x)
    <out-dir>/track<TTT>/gunw_80m.zarr   variable ``coherence_80m``, dims (pair, y, x)

Coordinates, identical in both stores:
    pair   'YYYYMMDD_YYYYMMDD' (local dates of the reference and secondary acquisitions)
    start  reference acquisition start, timezone-naive local standard time (UTC + utc_offset_hours), on ``pair``
    end    secondary acquisition start, same convention, on ``pair``
    x, y   pixel-center UTM coordinates (CRS in ``attrs['crs']``)

Example
-------
    python scripts/gunw_zarr.py --stack 019_A_021 --gunw-root /path/to/019_A_021/gunw

Downstream use (limit by time, combine tracks)
----------------------------------------------
    ds = xr.open_zarr('data/nisar/zarrs/track019/gunw_80m.zarr')
    keep = (ds.start.values >= np.datetime64('2026-01-01')) & (ds.end.values <= np.datetime64('2026-04-01'))
    ds = ds.isel(pair=keep)
    both = xr.concat([ds19.assign_coords(track=19), ds120.assign_coords(track=120)], dim='pair')
"""
import argparse
import logging
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import xarray as xr

# Allow `python scripts/gunw_zarr.py` from the repo root: put the repo root on the path so `scripts` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download import BASE_GRIDS, COORD_PATHS, LAYER_REGISTRY, _get_epsg  # noqa: E402

logger = logging.getLogger(__name__)
# Zarr v3 warns that fixed-width strings (our `pair` ids) have no formal spec; they read back fine with xarray.
warnings.filterwarnings('ignore', message='.*does not have a Zarr V3 specification.*')

POL = 'HH'
IDENT = 'science/LSAR/identification'
# Output store name -> (layer in LAYER_REGISTRY, variable name in the store).
RESOLUTIONS = {
    '20m': ('coherence_wrapped', 'coherence_20m'),
    '80m': ('coherence_unwrapped', 'coherence_80m'),
}


def find_gunws(gunw_root: Path, orbit: str) -> List[Path]:
    """GUNW files of one orbit type, one per pair folder ``<ref>_<sec>_<ORBIT>``."""
    return sorted(gunw_root.glob(f'*_{orbit.upper()}/GUNW_product.h5'))


def _local_time(raw: bytes, utc_offset_hours: float) -> np.datetime64:
    """ISO UTC string from the GUNW identification group -> naive local datetime64[ns]."""
    utc = datetime.fromisoformat(raw.decode().strip())
    return np.datetime64(utc + timedelta(hours=utc_offset_hours), 'ns')


def read_pair(path: Path, utc_offset_hours: float) -> dict:
    """
    Read what one GUNW contributes: start/end times, the coherence at each resolution, its x/y, and metadata.
    Returns a dict with keys ``start``, ``end``, ``pair``, ``track``, ``frame``, ``crs`` and, per resolution key
    in ``RESOLUTIONS``, ``data[res] = (coherence, x, y)``.
    """
    with h5py.File(path, 'r') as hf:
        start = _local_time(hf[f'{IDENT}/referenceZeroDopplerStartTime'][()], utc_offset_hours)
        end = _local_time(hf[f'{IDENT}/secondaryZeroDopplerStartTime'][()], utc_offset_hours)
        data = {}
        for res, (layer, _) in RESOLUTIONS.items():
            rel = LAYER_REGISTRY[layer]['path'].format(pol=POL)
            x_rel, y_rel = (p.format(pol=POL) for p in COORD_PATHS[LAYER_REGISTRY[layer]['coords']])
            data[res] = (hf[f'{BASE_GRIDS}/{rel}'][:], hf[f'{BASE_GRIDS}/{x_rel}'][:], hf[f'{BASE_GRIDS}/{y_rel}'][:])
        return {
            'start': start, 'end': end,
            'pair': f"{str(start)[:10].replace('-', '')}_{str(end)[:10].replace('-', '')}",
            'track': int(hf[f'{IDENT}/trackNumber'][()]), 'frame': int(hf[f'{IDENT}/frameNumber'][()]),
            'crs': f'EPSG:{_get_epsg(hf, POL)}', 'data': data,
        }


def _make_dataset(res: str, batch: List[dict], utc_offset_hours: float) -> xr.Dataset:
    """Stack a batch of pairs (already loaded in memory) into a (pair, y, x) Dataset for one resolution."""
    var = RESOLUTIONS[res][1]
    _, x, y = batch[0]['data'][res]
    ds = xr.Dataset(
        {var: (('pair', 'y', 'x'), np.stack([b['data'][res][0] for b in batch]).astype('float32'))},
        coords={
            'pair': [b['pair'] for b in batch], 'y': y, 'x': x,
            'start': ('pair', np.array([b['start'] for b in batch], dtype='datetime64[ns]')),
            'end': ('pair', np.array([b['end'] for b in batch], dtype='datetime64[ns]')),
        },
    )
    ds[var].attrs['long_name'] = f'GUNW coherence magnitude ({res} grid), HH'
    ds['start'].attrs.update(utc_offset_hours=utc_offset_hours, long_name='reference acquisition start, local standard time')
    ds['end'].attrs.update(utc_offset_hours=utc_offset_hours, long_name='secondary acquisition start, local standard time')
    # Fixed encoding so every appended batch uses the same units; xarray's default is picked per batch from the
    # first value and can fail to round-trip microsecond times.
    for t in ('start', 'end'):
        ds[t].encoding.update(units='nanoseconds since 1970-01-01', calendar='proleptic_gregorian', dtype='int64')
    ds.attrs.update(crs=batch[0]['crs'], track=batch[0]['track'], frame=batch[0]['frame'])
    return ds


def _existing_pairs(store: Path) -> Tuple[set, Optional[xr.Dataset]]:
    """Pair ids already in a store (empty set if it does not exist yet)."""
    if not store.exists():
        return set(), None
    ds = xr.open_zarr(store, consolidated=False)
    return set(ds['pair'].values.tolist()), ds


def build(files: List[Path], out_dir: Path, utc_offset_hours: float, max_days: Optional[int],
          batch_size: int, overwrite: bool) -> Dict[str, List[str]]:
    """Write/append the two stores of one track. Returns {'failed': [...], 'skipped': [...]} file lists."""
    problems = {'failed': [], 'skipped': []}

    # Start times are needed before reading rasters, to order the build and apply --max-days.
    heads = []
    for f in files:
        try:
            with h5py.File(f, 'r') as hf:
                s = _local_time(hf[f'{IDENT}/referenceZeroDopplerStartTime'][()], utc_offset_hours)
                e = _local_time(hf[f'{IDENT}/secondaryZeroDopplerStartTime'][()], utc_offset_hours)
        except (OSError, KeyError) as err:
            logger.warning(f'cannot read times of {f}: {err}')
            problems['failed'].append(str(f))
            continue
        if max_days is not None and (e - s) / np.timedelta64(1, 'D') > max_days:
            continue
        heads.append((s, e, f))
    heads.sort(key=lambda h: (h[0], h[1]))

    stores = {res: out_dir / f'gunw_{res}.zarr' for res in RESOLUTIONS}
    existing = {}
    for res, store in stores.items():
        if overwrite and store.exists():
            shutil.rmtree(store)
        existing[res] = _existing_pairs(store)
    # A pair counts as done only if it is in both stores; one missing from either is (re)written to the one lacking it.
    seen = {res: set(existing[res][0]) for res in RESOLUTIONS}

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_grid: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for res, (_, ds) in existing.items():
        if ds is not None:
            ref_grid[res] = (ds['x'].values, ds['y'].values)

    batch: List[dict] = []

    def flush():
        for res, store in stores.items():
            rows = [b for b in batch if b['pair'] not in seen[res]]
            if not rows:
                continue
            ds = _make_dataset(res, rows, utc_offset_hours)
            var = RESOLUTIONS[res][1]
            if store.exists():
                ds.to_zarr(store, append_dim='pair', consolidated=False)
            else:
                ny, nx = ds[var].shape[1:]
                ds[var].encoding['chunks'] = (batch_size, min(256, ny), min(256, nx))
                ds.to_zarr(store, mode='w', consolidated=False)
            seen[res].update(b['pair'] for b in rows)
            logger.info(f'{store.name}: +{len(rows)} pairs (now {len(seen[res])})')
        batch.clear()

    for s, e, f in heads:
        pair = f"{str(s)[:10].replace('-', '')}_{str(e)[:10].replace('-', '')}"
        if all(pair in seen[res] for res in RESOLUTIONS):
            continue
        try:
            p = read_pair(f, utc_offset_hours)
        except (OSError, KeyError) as err:
            logger.warning(f'cannot read {f}: {err}')
            problems['failed'].append(str(f))
            continue
        # Every pair must sit on the grid of the first one, or the stack would be silently misaligned.
        ok = True
        for res in RESOLUTIONS:
            _, x, y = p['data'][res]
            if res not in ref_grid:
                ref_grid[res] = (x, y)
            elif not (np.array_equal(ref_grid[res][0], x) and np.array_equal(ref_grid[res][1], y)):
                ok = False
        if not ok:
            logger.warning(f'{f}: grid differs from the rest of the track, skipped')
            problems['skipped'].append(str(f))
            continue
        batch.append(p)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('Example')[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Reruns resume: pairs already in a store are skipped and new ones are appended (appended pairs come '
               'after the existing ones, so sort by start if order matters). Use --overwrite to rebuild.',
    )
    ap.add_argument('--stack', required=True, help="ASF stack id, e.g. 019_A_021; the track is its first field")
    ap.add_argument('--gunw-root', required=True, type=Path,
                    help='folder holding <ref>_<sec>_<ORBIT>/GUNW_product.h5 (<PROD_ROOT>/<stack>/gunw on the cluster)')
    ap.add_argument('--out-dir', type=Path, default=Path('data/nisar/zarrs'),
                    help='parent folder; stores go in <out-dir>/track<TTT>/ (default: %(default)s)')
    ap.add_argument('--utc-offset-hours', type=float, default=-7,
                    help='fixed offset applied to the UTC times, no DST (default: %(default)s, MST)')
    ap.add_argument('--orbit', default='poe', help='orbit type of the GUNW folders, poe or moe (default: %(default)s)')
    ap.add_argument('--max-days', type=int, default=None,
                    help='only include pairs up to this many days apart (default: all)')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='pairs held in memory per write, also the Zarr chunk size along pair (default: %(default)s)')
    ap.add_argument('--overwrite', action='store_true', help='delete and rebuild this track\'s stores')
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    track = f"track{int(a.stack.split('_')[0]):03d}"
    files = find_gunws(a.gunw_root, a.orbit)
    if not files:
        sys.exit(f'no */GUNW_product.h5 with orbit {a.orbit.upper()} under {a.gunw_root}')
    logger.info(f'{a.stack}: {len(files)} GUNWs -> {a.out_dir / track}')
    problems = build(files, a.out_dir / track, a.utc_offset_hours, a.max_days, a.batch_size, a.overwrite)
    for kind, lst in problems.items():
        if lst:
            logger.warning(f'{kind} ({len(lst)}): ' + ', '.join(lst))


if __name__ == '__main__':
    main()
