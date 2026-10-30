"""
Calculate gamma_thermal, gamma_spatial and gamma_temporal for one NISAR track, in batches of pairs run with dask,
so a large stack of pairs does not have to fit in memory. This is the script version of
``notebooks/2_gamma_temporal.ipynb`` and reads and writes the same things.

Inputs (``<data-dir>`` is ``data`` by default, ``TTT``/``FFF`` are the zero-padded track and frame):
    <zarr-dir>/gunw_80m.zarr, gunw_20m.zarr   cubes (pair, y, x) with ``coherence`` and ``coherence_true`` (notebook 1);
                                              the 80 m cube also needs ``elevation`` (the DEM for the radar grid)
    <data-dir>/nisar/east_river/trackTTT_frameFFF/   backscatter and NEBZ GeoTIFFs (for the SNR)
    <gunw-root>/TTT_<A|D>_FFF/<ref>-<sec>_POE/GUNW_product.h5   the GUNWs made by submit_gunw_pairs.sh; the incidence
                                              angle, perpendicular baseline and reference slant range come from their
                                              metadata/radarGrid group, and the pair of each file from its times
    --aso <tif>                               optional, masks the cubes to the footprint of this raster

Outputs, in ``<out-dir>`` (default: ``<zarr-dir>``, which is ``<data-dir>/nisar/zarrs/all_pairs/trackTTT``):
    coherence_80m.zarr, coherence_20m.zarr    the input variables plus gamma_thermal, gamma_spatial,
                                              gamma_temporal_uncorrected (coherence / (thermal * spatial)) and
                                              gamma_temporal (coherence_true / (thermal * spatial))
    nearest_neighbor_tifs_80m/, _20m/         one GeoTIFF per nearest-neighbour pair (the pair whose secondary is the
                                              next acquisition after its reference) for each of ``--tif-vars``

How the work is split: pairs are processed in batches of ``--pair-chunk``. For each batch only the SNR files of the
dates it needs and the GUNW files of its pairs are opened, so memory use is set by the batch size and the number
of workers (peak is roughly ``--n-workers`` batches at once). Each batch writes its own slice of the output Zarr,
which is stored in ``(pair-chunk, tile, tile)`` chunks so the next step can read one spatial tile for all pairs
without touching every pixel.

Example, from the repo root (on the cluster use scripts/slurm/gamma_temporal.sbatch):
    python scripts/gamma_temporal.py --track 19 --frame 21 --gunw-root /path/to/stacks \
        --aso data/aso/er_20260318/ASO_EastRiver_2026Mar18_snowdepth_3m.tif
"""
import argparse
import logging
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dask
import h5py
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from pyproj import Transformer
from rasterio.enums import Resampling

# Allow `python scripts/gamma_temporal.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coherence import spatial_coherence  # noqa: E402
from scripts.download import (  # noqa: E402
    BASE_META, _interp_radar_grid, _match_gslc_files, _read_coords, _read_radar_grid_layer, build_snr_timeseries,
    parse_times,
)
from scripts.gunw_zarr import IDENT, POL, _local_time  # noqa: E402

logger = logging.getLogger(__name__)
# Zarr v3 warns that fixed-width strings (our `pair` ids) have no formal spec; they read back fine with xarray.
warnings.filterwarnings('ignore', message='.*does not have a Zarr V3 specification.*')

# Variables written on top of the ones in the input cubes. All are (pair, y, x).
GAMMA_VARS = ['gamma_thermal', 'gamma_spatial', 'gamma_temporal_uncorrected', 'gamma_temporal']
# Radar-grid layer -> its dataset in the GUNW's metadata/radarGrid group.
RADAR_GRID_LAYERS = {'incidence': 'incidenceAngle', 'baseline': 'perpendicularBaseline',
                     'range': 'referenceSlantRange'}
DIMS = ('pair', 'y', 'x')


@dataclass
class Job:
    """Everything a batch needs. Plain data, so it can be sent to a worker process."""
    zarr_dir: Path
    out_dir: Path
    resolutions: List[str]
    pairs: List[str]
    starts: np.ndarray               # reference time of each pair, datetime64[ns], local standard time
    ends: np.ndarray
    nearest_neighbor: np.ndarray     # bool per pair
    acq_times: np.ndarray            # datetime64[ns] of each GSLC acquisition that has backscatter and NEBZ
    bs_files: List[Path]             # same order as acq_times
    nebz_files: List[Path]
    gunw_files: Dict[str, Path]      # pair id -> GUNW_product.h5 (the radar-grid layers are read from it)
    dem: xr.DataArray                # masked 80 m elevation with a CRS
    masks: Dict[str, Optional[np.ndarray]]    # resolution -> bool (y, x) ASO footprint, None for no mask
    templates: Dict[str, xr.DataArray]        # resolution -> empty (y, x) array with the grid and CRS of the cube
    utc_offset: float
    spatial_res: float
    wavelength: float
    incidence_to_radians: bool       # the GUNW incidence angle is in degrees
    tif_vars: List[str]
    pair_chunk: int


def find_files(folder: Path, pattern: str) -> List[Path]:
    return sorted(item.resolve() for item in folder.rglob(pattern))


def find_gunws(gunw_root: Path, track: int, frame: int, orbit: str) -> List[Path]:
    """
    GUNW files of one track and frame: ``<gunw-root>/<TTT>_<A|D>_<FFF>/**/<ref>-<sec>_<ORBIT>/GUNW_product.h5``. The
    levels between the stack folder and the pair folder (such as ``gunw/``) do not matter.
    """
    stacks = sorted(gunw_root.glob(f'{track:03d}_[AD]_{frame:03d}'))
    if len(stacks) != 1:
        sys.exit(f'expected one {track:03d}_<A|D>_{frame:03d} folder in {gunw_root}, found {[s.name for s in stacks]}')
    return sorted(f for f in stacks[0].rglob('GUNW_product.h5') if f.parent.name.upper().endswith(f'_{orbit.upper()}'))


def index_gunws(files: List[Path], utc_offset: float) -> Dict[str, Path]:
    """
    ``'YYYYMMDD_YYYYMMDD'`` -> GUNW file. The pair id comes from the reference and secondary start times stored in the
    file (as ``gunw_zarr.py`` makes it), not from the folder name, whose dates may be UTC dates.
    """
    out = {}
    for f in files:
        try:
            with h5py.File(f, 'r') as hf:
                start = _local_time(hf[f'{IDENT}/referenceZeroDopplerStartTime'][()], utc_offset)
                end = _local_time(hf[f'{IDENT}/secondaryZeroDopplerStartTime'][()], utc_offset)
        except (OSError, KeyError) as err:
            logger.warning(f'cannot read the times of {f}: {err}')
            continue
        pair = f"{str(start)[:10].replace('-', '')}_{str(end)[:10].replace('-', '')}"
        if pair in out:
            logger.warning(f'{pair} is in both {out[pair]} and {f}; using the first')
        else:
            out[pair] = f
    return out


def get_crs(ds: xr.Dataset) -> str:
    """CRS of a cube: from its ``spatial_ref`` coordinate if it has one, else the ``crs`` attribute."""
    if ds.rio.crs is not None:
        return ds.rio.crs
    if 'crs' in ds.attrs:
        return ds.attrs['crs']
    raise ValueError('cannot find the CRS of the cube (no spatial_ref coordinate or crs attribute)')


def grid_template(ds: xr.Dataset) -> xr.DataArray:
    """Empty (y, x) array on the grid of a cube, to reproject onto."""
    empty = xr.DataArray(np.zeros((ds.sizes['y'], ds.sizes['x']), dtype='float32'), dims=('y', 'x'),
                         coords={'y': ds['y'].values, 'x': ds['x'].values})
    return empty.rio.write_crs(get_crs(ds))


def nearest_index(times: np.ndarray, when: np.ndarray) -> np.ndarray:
    """Position in ``times`` of the acquisition nearest each value of ``when`` (as ``sel(method='nearest')``)."""
    return np.abs(times[None, :] - when[:, None]).argmin(axis=1)


def batch_thermal(job: Job, starts: np.ndarray, ends: np.ndarray, pairs: List[str]) -> xr.DataArray:
    """
    gamma_thermal, ``1 / sqrt((1 + 1/SNR_1) * (1 + 1/SNR_2))``, for a batch of pairs; the same as
    ``coherence.thermal_coherence`` but written without a loop over the pairs. Only the SNR of the acquisitions
    nearest the starts and ends of the batch is built.
    """
    i_start, i_end = nearest_index(job.acq_times, starts), nearest_index(job.acq_times, ends)
    need = np.unique(np.concatenate([i_start, i_end]))
    snr = build_snr_timeseries([job.bs_files[i] for i in need], [job.nebz_files[i] for i in need],
                               utc_offset_hours=job.utc_offset)
    snr_1 = snr.sel(date=xr.DataArray(job.acq_times[i_start], dims='pair')).reset_coords(drop=True)
    snr_2 = snr.sel(date=xr.DataArray(job.acq_times[i_end], dims='pair')).reset_coords(drop=True)
    gamma = (1.0 / np.sqrt((1 + 1 / snr_1) * (1 + 1 / snr_2))).assign_coords(pair=pairs)
    return gamma.transpose(*DIMS).rio.write_crs(snr.rio.crs)


def dem_points(dem: xr.DataArray, epsg: int):
    """x and y of every DEM pixel in the CRS of the radar grid."""
    xx, yy = np.meshgrid(dem['x'].values, dem['y'].values)
    if dem.rio.crs == f'EPSG:{epsg}':
        return xx, yy
    return Transformer.from_crs(dem.rio.crs, f'EPSG:{epsg}', always_xy=True).transform(xx, yy)


def interpolate_gunw_layers(dem: xr.DataArray, files: List[Path], pairs: List[str]) -> Dict[str, xr.DataArray]:
    """
    Incidence angle, perpendicular baseline and reference slant range of the GUNWs ``files`` (one per pair),
    interpolated onto the DEM pixels: linearly in x/y, then linearly in height at the DEM elevation, as
    ``download.interpolate_radar_grid_to_dem`` does for the GeoTIFFs written by ``download_nisar``. Returns one
    ``(pair, y, x)`` array per layer in ``RADAR_GRID_LAYERS``.
    """
    z = dem.values.astype(np.float64)
    points = {}  # radar-grid EPSG -> (x, y) of the DEM pixels
    out = {name: [] for name in RADAR_GRID_LAYERS}
    for f in files:
        with h5py.File(f, 'r') as hf:
            epsg = int(hf[f'{BASE_META}/radarGrid/projection'][()])
            x, y = _read_coords(hf, 'metadata', POL)
            layers = {name: _read_radar_grid_layer(hf, dataset, name, x, y, epsg)
                      for name, dataset in RADAR_GRID_LAYERS.items()}
        if epsg not in points:
            points[epsg] = dem_points(dem, epsg)
        for name, layer in layers.items():
            layer = layer.sortby('band').sortby('y')  # the interpolator needs ascending heights and y
            out[name].append(_interp_radar_grid(
                layer.values.astype(np.float64), layer['band'].values, layer['y'].values, layer['x'].values,
                *points[epsg], z, dtype=layer.dtype))
    coords = {'pair': pairs, 'y': dem['y'].values, 'x': dem['x'].values}
    return {name: xr.DataArray(np.stack(arrays), dims=DIMS, coords=coords) for name, arrays in out.items()}


def batch_spatial(job: Job, pairs: List[str]) -> xr.DataArray:
    """
    gamma_spatial on the 80 m DEM grid for a batch of pairs. Only the GUNW files of these pairs are opened.
    Pairs without a GUNW are NaN.
    """
    have = [p for p in pairs if p in job.gunw_files]
    gamma = xr.DataArray(np.full((len(pairs), job.dem.sizes['y'], job.dem.sizes['x']), np.nan), dims=DIMS,
                         coords={'pair': pairs, 'y': job.dem['y'].values, 'x': job.dem['x'].values})
    # NaN is declared nodata, as in the notebook, so the bilinear reprojection skips NaN neighbours
    gamma = gamma.rio.write_crs(job.dem.rio.crs).rio.write_nodata(np.nan)
    if not have:
        return gamma
    layers = interpolate_gunw_layers(job.dem, [job.gunw_files[p] for p in have], have)
    theta = np.deg2rad(layers['incidence']) if job.incidence_to_radians else layers['incidence']
    result = spatial_coherence(layers['baseline'], theta, layers['range'], job.spatial_res, job.wavelength)
    gamma.loc[{'pair': have}] = result.transpose(*DIMS).values
    return gamma


def process_batch(job: Job, i0: int, i1: int, batch: int, n_batches: int) -> None:
    """Compute the gammas for pairs ``i0:i1``, write them into each output Zarr and export the nearest-neighbour tifs."""
    sl = slice(i0, i1)
    pairs = job.pairs[sl]
    thermal = batch_thermal(job, job.starts[sl], job.ends[sl], pairs)
    spatial = batch_spatial(job, pairs)

    for res in job.resolutions:
        # chunks=None: plain lazy indexing of the Zarr, so these reads do not start a second dask computation
        cube = xr.open_zarr(job.zarr_dir / f'gunw_{res}.zarr', chunks=None).isel(pair=sl)
        out = {}
        for name in cube.data_vars:
            if 'pair' in cube[name].dims:
                arr = cube[name].transpose(*DIMS).values
                mask = job.masks[res]
                out[name] = arr if mask is None else np.where(mask, arr, np.nan).astype(arr.dtype)
        # thermal comes from the GSLC grid and spatial from the 80 m grid: both go onto the grid of this cube
        template = job.templates[res]
        g_thermal = thermal.rio.reproject_match(template, resampling=Resampling.bilinear).transpose(*DIMS).values
        g_spatial = spatial.rio.reproject_match(template, resampling=Resampling.bilinear).transpose(*DIMS).values
        if not (g_thermal.shape == g_spatial.shape == out['coherence'].shape):
            raise ValueError(f'{res}: shapes differ, thermal {g_thermal.shape}, spatial {g_spatial.shape}, '
                             f'coherence {out["coherence"].shape}')
        denominator = g_thermal * g_spatial
        with np.errstate(divide='ignore', invalid='ignore'):
            out['gamma_thermal'] = g_thermal
            out['gamma_spatial'] = g_spatial
            out['gamma_temporal_uncorrected'] = out['coherence'] / denominator
            out['gamma_temporal'] = out['coherence_true'] / denominator
        for name in GAMMA_VARS:
            out[name] = out[name].astype('float32')

        # Only data variables, no coords: a region write must not carry the full index.
        xr.Dataset({k: (DIMS, v) for k, v in out.items()}).to_zarr(
            job.out_dir / f'coherence_{res}.zarr', region={'pair': sl, 'y': slice(None), 'x': slice(None)},
            consolidated=False)

        tif_dir = job.out_dir / f'nearest_neighbor_tifs_{res}'
        for k in np.flatnonzero(job.nearest_neighbor[sl]):
            for var in job.tif_vars:
                da = xr.DataArray(out[var][k], dims=('y', 'x'), coords={'y': template['y'], 'x': template['x']})
                da.rio.write_crs(template.rio.crs).rio.write_nodata(np.nan).rio.to_raster(
                    tif_dir / f'{var}_{pairs[k]}.tif')
    logger.info(f'batch {batch + 1}/{n_batches}: pairs {i0}:{i1} done')


def nearest_neighbor_pairs(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """True for pairs whose secondary is the next acquisition after their reference (dates taken from the pairs)."""
    dates = np.unique(np.concatenate([starts, ends]))
    return np.searchsorted(dates, ends) - np.searchsorted(dates, starts) == 1


def init_output(job: Job, res: str, cube: xr.Dataset, args) -> None:
    """Create the output Zarr for one resolution: all metadata and the 2D variables now, the (pair, y, x) data later."""
    mask = job.masks[res]
    if mask is not None:
        cube = cube.where(xr.DataArray(mask, dims=('y', 'x'), coords={'y': cube['y'], 'x': cube['x']}))
    # Non-index coordinates (start, end) and 2D variables are loaded so that they are written now.
    cube = cube.assign_coords({k: v.load() for k, v in cube.coords.items()})
    chunks = {'pair': args.pair_chunk, 'y': args.tile_size, 'x': args.tile_size}
    data = {}
    for name, da in cube.data_vars.items():
        data[name] = da.chunk(chunks) if 'pair' in da.dims else da.load()
    for name in GAMMA_VARS:
        data[name] = xr.zeros_like(cube['coherence'], dtype='float32').chunk(chunks)
    out = xr.Dataset(data, coords=cube.coords, attrs=cube.attrs)
    out.attrs.update(utc_offset_hours=job.utc_offset, spatial_res=job.spatial_res, wavelength=job.wavelength,
                     incidence_to_radians=int(job.incidence_to_radians), aso=str(args.aso))
    for var in out.variables.values():  # the chunks of the input store must not override the new ones
        for key in ('chunks', 'preferred_chunks', 'shards'):
            var.encoding.pop(key, None)
    out.to_zarr(job.out_dir / f'coherence_{res}.zarr', mode='w', compute=False, consolidated=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('Example')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', required=True, type=int)
    ap.add_argument('--frame', required=True, type=int)
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='default: %(default)s')
    ap.add_argument('--zarr-dir', type=Path, default=None,
                    help='folder with gunw_<res>.zarr (default: <data-dir>/nisar/zarrs/all_pairs/track<TTT>)')
    ap.add_argument('--out-dir', type=Path, default=None, help='output folder (default: --zarr-dir)')
    ap.add_argument('--gunw-root', required=True, type=Path,
                    help='folder holding <TTT>_<A|D>_<FFF>/<ref>-<sec>_<ORBIT>/GUNW_product.h5')
    ap.add_argument('--orbit', default='poe', help='orbit type of the GUNW folders, poe or moe (default: %(default)s)')
    ap.add_argument('--aso', type=Path, default=None, help='raster whose footprint masks the cubes (default: no mask)')
    ap.add_argument('--resolutions', nargs='+', choices=['80m', '20m'], default=['80m', '20m'])
    ap.add_argument('--utc-offset', type=float, default=-7, help='hours from UTC, no DST (default: %(default)s)')
    ap.add_argument('--spatial-res', type=float, default=3,
                    help='resolution (m) in the spatial-baseline term, about the GSLC resolution (default: %(default)s)')
    ap.add_argument('--wavelength', type=float, default=0.24, help='radar wavelength, m (default: %(default)s)')
    ap.add_argument('--no-incidence-conversion', action='store_true',
                    help='use the incidence angle as stored (degrees) in gamma_spatial instead of converting it to '
                         'radians, which is what notebook 2 does')
    ap.add_argument('--tif-vars', nargs='+', choices=GAMMA_VARS, default=['gamma_temporal'],
                    help='variables exported as GeoTIFFs for the nearest-neighbour pairs (default: %(default)s)')
    ap.add_argument('--pair-chunk', type=int, default=20, help='pairs per batch (default: %(default)s)')
    ap.add_argument('--tile-size', type=int, default=256, help='spatial chunk of the output, pixels (default: %(default)s)')
    ap.add_argument('--n-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)),
                    help='batches processed at once (default: SLURM_CPUS_PER_TASK, else all CPUs)')
    ap.add_argument('--scheduler', choices=['threads', 'processes', 'synchronous'], default='threads',
                    help='dask scheduler (default: %(default)s)')
    ap.add_argument('--overwrite', action='store_true', help='replace existing outputs')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    track_dir = f'track{args.track:03d}'
    zarr_dir = args.zarr_dir or args.data_dir / 'nisar' / 'zarrs' / 'all_pairs' / track_dir
    out_dir = args.out_dir or zarr_dir
    scene_dir = args.data_dir / 'nisar' / 'east_river' / f'{track_dir}_frame{args.frame:03d}'
    resolutions = list(dict.fromkeys(args.resolutions))

    # The 80 m cube is always read: its elevation is the DEM that the radar grid is interpolated onto.
    cubes = {res: xr.open_zarr(zarr_dir / f'gunw_{res}.zarr') for res in dict.fromkeys(['80m'] + resolutions)}
    for res, cube in cubes.items():
        missing = [v for v in ('coherence', 'coherence_true') if v not in cube]
        if res == '80m' and 'elevation' not in cube:
            missing.append('elevation')
        if missing:
            sys.exit(f'{zarr_dir}/gunw_{res}.zarr is missing {missing}')
        if not np.array_equal(cube['pair'].values, cubes['80m']['pair'].values):
            sys.exit(f'the {res} and 80m cubes do not have the same pairs')
    cube80 = cubes['80m']
    pairs = cube80['pair'].values.tolist()
    starts, ends = cube80['start'].values.astype('datetime64[ns]'), cube80['end'].values.astype('datetime64[ns]')

    # GSLC acquisitions (for the SNR) and radar-grid files (for the spatial baseline); only listed here, opened per batch
    matched = _match_gslc_files(find_files(scene_dir, '*backscatter*'), find_files(scene_dir, '*NEBZ*'))['matched']
    if not matched:
        sys.exit(f'no matching backscatter / NEBZ files in {scene_dir}')
    acquisitions = sorted((np.datetime64(parse_times(scene, args.utc_offset)[0], 'ns'), bs, nz)
                          for scene, bs, nz in matched)
    gunw_files = index_gunws(find_gunws(args.gunw_root, args.track, args.frame, args.orbit), args.utc_offset)
    logger.info(f'{len(pairs)} pairs, {len(acquisitions)} acquisitions with SNR, {len(gunw_files)} GUNW files')
    no_radar = [p for p in pairs if p not in gunw_files]
    if no_radar:
        logger.warning(f'{len(no_radar)} pairs have no GUNW file and get NaN gamma_spatial: {no_radar}')
    if len(no_radar) == len(pairs):
        sys.exit(f'none of the pairs has a GUNW file under {args.gunw_root}')

    # ASO footprint on each grid, as in the notebook (the raster has a length-1 band dimension)
    templates = {res: grid_template(cube) for res, cube in cubes.items()}
    masks = {res: None for res in cubes}
    if args.aso is not None:
        if not args.aso.exists():
            sys.exit(f'--aso {args.aso} does not exist')
        # open_rasterio with a str, not xr.open_dataarray(Path): that depends on xarray guessing the backend, which
        # fails with some rioxarray versions
        aso = rioxarray.open_rasterio(str(args.aso)).squeeze('band', drop=True)
        masks = {res: aso.rio.reproject_match(t).notnull().values for res, t in templates.items()}
    dem = cube80['elevation'].load()
    if masks['80m'] is not None:
        dem = dem.where(masks['80m'])
    dem = dem.rio.write_crs(templates['80m'].rio.crs)

    job = Job(
        zarr_dir=zarr_dir, out_dir=out_dir, resolutions=resolutions, pairs=pairs, starts=starts, ends=ends,
        nearest_neighbor=nearest_neighbor_pairs(starts, ends),
        acq_times=np.array([a[0] for a in acquisitions]), bs_files=[a[1] for a in acquisitions],
        nebz_files=[a[2] for a in acquisitions], gunw_files=gunw_files, dem=dem, masks=masks, templates=templates,
        utc_offset=args.utc_offset, spatial_res=args.spatial_res, wavelength=args.wavelength,
        incidence_to_radians=not args.no_incidence_conversion, tif_vars=args.tif_vars, pair_chunk=args.pair_chunk,
    )
    logger.info(f'{int(job.nearest_neighbor.sum())} nearest-neighbour pairs get GeoTIFFs of {args.tif_vars}')

    out_dir.mkdir(parents=True, exist_ok=True)
    for res in resolutions:
        store, tif_dir = out_dir / f'coherence_{res}.zarr', out_dir / f'nearest_neighbor_tifs_{res}'
        if store.exists():
            if not args.overwrite:
                sys.exit(f'{store} already exists; use --overwrite to replace it')
            shutil.rmtree(store)
        tif_dir.mkdir(exist_ok=True)
        init_output(job, res, cubes[res], args)

    starts_at = list(range(0, len(pairs), args.pair_chunk))
    tasks = [dask.delayed(process_batch)(job, i0, min(i0 + args.pair_chunk, len(pairs)), b, len(starts_at))
             for b, i0 in enumerate(starts_at)]
    logger.info(f'{len(tasks)} batches of up to {args.pair_chunk} pairs, {args.n_workers} at a time ({args.scheduler})')
    dask.compute(*tasks, scheduler=args.scheduler, num_workers=args.n_workers)
    logger.info(f'done, outputs in {out_dir}')


if __name__ == '__main__':
    main()
