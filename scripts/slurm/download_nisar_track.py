"""Download NISAR GUNW and/or GSLC layers for one track/frame, clipped to an AOI."""
import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

# import from this repo
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.download import download_nisar, download_nisar_gslc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)


def load_aoi(path):
    """Raster -> bounding-box GeoDataFrame; anything else is passed through to the library."""
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        with rasterio.open(path) as src:
            return gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=src.crs)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('track', type=int)
    p.add_argument('frame', type=int)
    p.add_argument('--aoi', required=True, help='Raster (bbox is used) or vector file')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--crid', default=None, help='Processing code, e.g. P05023')
    p.add_argument('--out-root', required=True)
    p.add_argument('--pol', default='HH')
    p.add_argument('--gunw-layers', nargs='*', default=[])
    p.add_argument('--gslc-layers', nargs='*', default=[])
    args = p.parse_args()

    out_dir = Path(args.out_root) / f'track{args.track:03d}_frame{args.frame:03d}'
    aoi = load_aoi(args.aoi)
    common = dict(track=args.track, frame=args.frame, start_date=args.start, end_date=args.end,
                  aoi=aoi, output_dir=out_dir, crid=args.crid)

    jobs = []
    if args.gunw_layers:
        jobs.append(('GUNW', lambda: download_nisar(layers=args.gunw_layers, polarization=args.pol, **common)))
    if args.gslc_layers:
        jobs.append(('GSLC', lambda: download_nisar_gslc(layers=args.gslc_layers, polarization=args.pol, **common)))

    failed = []
    for name, run in jobs:
        try:
            grans, files = run()
            print(f'{name}: {len(grans)} granule(s), {len(files)} file(s) in {out_dir}')
        except Exception:
            logging.exception(f'{name} download failed for track {args.track} frame {args.frame}')
            failed.append(name)

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
