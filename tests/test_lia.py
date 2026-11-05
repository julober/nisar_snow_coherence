"""Tests for scripts/lia.py and the nisar_pytools LOS interpolation it relies on (no network)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from nisar_pytools.utils.local_incidence_angle import interpolate_los_to_dem
from pyproj import Transformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.lia import derive_los_z, lia_from_los

RES = 10.0
THETA = np.radians(35.0)  # LOS incidence from vertical; sensor is to the east (target-to-sensor)
EPSG = 32613


def make_dem(elev_fn, n=60, x0=320000.0, y0=4310000.0, descending_y=True, epsg=EPSG):
    """Square DEM whose elevation is ``elev_fn(x, y)``; y descends like a GeoTIFF unless told otherwise."""
    x = x0 + RES * np.arange(n)
    y = y0 + RES * np.arange(n)
    if descending_y:
        y = y[::-1]
    xx, yy = np.meshgrid(x, y)
    da = xr.DataArray(elev_fn(xx, yy), dims=('y', 'x'), coords={'y': y, 'x': x})
    return da.rio.write_crs(f'EPSG:{epsg}')


def make_radar_grid(x_centre, y_centre, fn_x, heights=(-500.0, 1500.0, 3500.0, 5500.0), half=20000.0, step=1000.0):
    """Coarse radarGrid (z, y, x) with y descending. ``fn_x(x, y, h)`` gives los_x; los_y is 0."""
    x_rg = np.arange(x_centre - half, x_centre + half + step, step)
    y_rg = np.arange(y_centre + half, y_centre - half - step, -step)  # descending, as in NISAR products
    hh, yy, xx = np.meshgrid(np.asarray(heights), y_rg, x_rg, indexing='ij')
    los_x = fn_x(xx, yy, hh)
    return los_x, np.zeros_like(los_x), np.asarray(heights), x_rg, y_rg


def const_los(value):
    return lambda xx, yy, hh: np.full_like(xx, value)


def test_derive_los_z_is_positive_unit_length():
    lx = np.array([0.0, 0.5, np.sin(THETA)])
    ly = np.array([0.0, 0.5, 0.0])
    lz = derive_los_z(lx, ly)
    assert (lz >= 0).all()
    assert np.allclose(lx**2 + ly**2 + lz**2, 1.0)
    # x^2 + y^2 a hair above 1 (rounding) must not produce NaN
    assert derive_los_z(np.array([1.0 + 1e-9]), np.array([0.0]))[0] == 0.0


@pytest.mark.parametrize('descending_y', [True, False])
def test_flat_dem_gives_los_incidence(descending_y):
    dem = make_dem(lambda x, y: np.full_like(x, 1000.0), descending_y=descending_y)
    args = make_radar_grid(320300.0, 4310300.0, const_los(np.sin(THETA)))
    lia = lia_from_los(dem, *args[:2], *args[2:], radar_epsg=EPSG)
    assert np.allclose(lia.values, 35.0, atol=1e-3)


@pytest.mark.parametrize('descending_y', [True, False])
def test_tilted_plane_matches_analytic_angle(descending_y):
    """Slope facing the sensor (east) lowers the angle; facing away raises it."""
    slope = 0.2
    args = make_radar_grid(320300.0, 4310300.0, const_los(np.sin(THETA)))
    for sign in (+1, -1):  # +1: surface drops toward the east (faces the sensor)
        dem = make_dem(lambda x, y: 1000.0 - sign * slope * (x - 320000.0), descending_y=descending_y)
        lia = lia_from_los(dem, *args[:2], *args[2:], radar_epsg=EPSG)
        normal = np.array([sign * slope, 0.0, 1.0]) / np.hypot(slope, 1.0)
        los = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
        expected = np.degrees(np.arccos(normal @ los))
        assert np.allclose(lia.values[2:-2, 2:-2], expected, atol=1e-3)
    # sanity: facing the sensor is smaller than facing away
    assert np.degrees(np.arccos(np.array([slope, 0, 1]) / np.hypot(slope, 1) @ los)) < 35.0 < \
        np.degrees(np.arccos(np.array([-slope, 0, 1]) / np.hypot(slope, 1) @ los))


def test_interpolation_linear_field_descending_axes():
    """LOS varying linearly in x, y and height is recovered exactly on a DEM with descending y."""
    x0, y0 = 320000.0, 4310000.0
    a, bx, by, bh = 0.3, 2e-6, -1e-6, 5e-6
    heights = np.array([5500.0, 3500.0, 1500.0, -500.0])  # descending too: must be re-sorted
    x_rg = np.arange(x0 - 5000, x0 + 5001, 1000.0)[::-1]  # descending x
    y_rg = np.arange(y0 + 5000, y0 - 5001, -1000.0)
    hh, yy, xx = np.meshgrid(heights, y_rg, x_rg, indexing='ij')
    field = a + bx * (xx - x0) + by * (yy - y0) + bh * hh
    dem = make_dem(lambda x, y: 500.0 + 0.05 * (x - x0) + 0.02 * (y - y0))  # heights between layers
    got, _, _ = interpolate_los_to_dem(dem, field, field, field, heights, x_rg, y_rg)
    xx2, yy2 = np.meshgrid(dem.x.values, dem.y.values)
    expected = a + bx * (xx2 - x0) + by * (yy2 - y0) + bh * dem.values
    assert np.allclose(got, expected, atol=1e-9)


def test_pixels_outside_radar_grid_are_nan_not_shifted():
    dem = make_dem(lambda x, y: np.full_like(x, 1000.0), n=60)
    # radarGrid covers only the western half of the DEM
    args = make_radar_grid(320000.0, 4310300.0, const_los(np.sin(THETA)), half=300.0, step=100.0)
    lia = lia_from_los(dem, *args[:2], *args[2:], radar_epsg=EPSG)
    assert np.isnan(lia.sel(x=slice(320400, None)).values).all()
    inside = lia.sel(x=slice(None, 320250)).values
    assert np.isfinite(inside).all() and np.allclose(inside, 35.0, atol=1e-3)


def test_dem_in_other_crs_is_reprojected_to_radar_crs():
    """DEM in EPSG:32613 with a radarGrid in EPSG:32612: LOS must be sampled at the true location.

    los_x increases with the radar-grid easting, so sampling at the wrong coordinates (DEM easting
    used directly as radar easting) would give a visibly different angle.
    """
    radar_epsg = 32612
    dem = make_dem(lambda x, y: np.full_like(x, 1000.0), n=200)  # flat -> LIA = arccos(los_z)
    to_radar = Transformer.from_crs(EPSG, radar_epsg, always_xy=True)
    cx, cy = to_radar.transform(dem.x.values.mean(), dem.y.values.mean())

    a, b = 0.55, 5e-6  # los_x per metre of radar easting (stays well inside |los_x| < 1 over the grid)
    fn = lambda xx, yy, hh: a + b * (xx - cx)  # noqa: E731
    args = make_radar_grid(cx, cy, fn, half=20000.0, step=500.0)
    lia = lia_from_los(dem, *args[:2], *args[2:], radar_epsg=radar_epsg)

    assert lia.rio.crs == dem.rio.crs
    assert lia.shape == dem.shape and np.array_equal(lia.x, dem.x) and np.array_equal(lia.y, dem.y)

    xx, yy = np.meshgrid(dem.x.values, dem.y.values)
    xr_, _ = to_radar.transform(xx, yy)
    expected = np.degrees(np.arccos(np.sqrt(1 - (a + b * (xr_ - cx)) ** 2)))
    interior = (slice(10, -10), slice(10, -10))
    assert np.allclose(lia.values[interior], expected[interior], atol=0.05)

    # The two CRSs place the same point far apart, so sampling the radarGrid at the DEM's own
    # coordinates would miss it entirely; this guards the test against passing trivially.
    assert abs(cx - dem.x.values.mean()) > 1000.0


def test_dem_without_crs_raises():
    dem = make_dem(lambda x, y: np.zeros_like(x))
    dem = dem.drop_vars('spatial_ref')
    args = make_radar_grid(320300.0, 4310300.0, const_los(np.sin(THETA)))
    with pytest.raises(ValueError):
        lia_from_los(dem, *args[:2], *args[2:], radar_epsg=EPSG)
