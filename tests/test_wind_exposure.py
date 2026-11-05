"""Tests for scripts/wind_exposure.py (Winstral et al. 2002 Sx) and the wind-direction helper."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.stations import prevailing_direction
from scripts.wind_exposure import sx

DX = 10.0


def _ridge_dem(ncol=60, nrow=40, ridge_col=30, height=100.0):
    """Flat terrain with a north-south wall of ``height`` m at ``ridge_col``."""
    dem = np.zeros((nrow, ncol))
    dem[:, ridge_col] = height
    return dem


def test_flat_terrain_is_zero():
    out = sx(np.full((30, 30), 500.0), DX, azimuth=250, dmax=100)
    # Cells on the upwind (west) edge have no upwind terrain at all, so NaN is
    # expected there; everywhere else must be exactly flat.
    assert np.allclose(out[np.isfinite(out)], 0.0)
    assert np.isfinite(out[:, 10:]).all()


def test_west_wind_shelters_lee_of_ridge():
    dem = _ridge_dem()
    out = sx(dem, DX, azimuth=270, dmax=100)
    row = 20
    assert out[row, 31] > 60  # right behind the wall: steeply sheltered
    assert out[row, 33] > 0  # a few cells downwind: still sheltered
    assert out[row, 30] < 0  # on the crest: exposed (higher than all upwind)
    assert out[row, 10] == 0  # far upwind of the wall: flat
    assert out[row, 45] == 0  # beyond dmax (150 m) downwind: wall not in range


def test_reversing_wind_flips_the_sheltered_side():
    dem = _ridge_dem()
    west = sx(dem, DX, azimuth=270, dmax=100)
    east = sx(dem, DX, azimuth=90, dmax=100)
    row = 20
    assert west[row, 32] > 0 and east[row, 32] == 0  # east of wall
    assert east[row, 28] > 0 and west[row, 28] == 0  # west of wall


def test_larger_dmax_never_reduces_sx():
    # Max upward slope over a longer search can only stay the same or grow.
    rng = np.random.default_rng(0)
    dem = np.cumsum(np.cumsum(rng.normal(size=(50, 50)), 0), 1) * 3
    near = sx(dem, DX, 250, 100)
    far = sx(dem, DX, 250, 300)
    valid = ~np.isnan(near) & ~np.isnan(far)
    assert np.all(far[valid] >= near[valid] - 1e-9)


def _brute_force(dem, dx, azimuth, dmax, r, c, half_width=30, step=5):
    """Slow, independent re-implementation of Sx for one cell (paper Eq. 1-2)."""
    angles = []
    for az in np.arange(azimuth - half_width, azimuth + half_width + 1e-9, step):
        best = -np.inf
        for k in range(1, int(dmax // dx) + 1):
            rr = r - int(round(np.cos(np.deg2rad(az)) * k))
            cc = c + int(round(np.sin(np.deg2rad(az)) * k))
            if not (0 <= rr < dem.shape[0] and 0 <= cc < dem.shape[1]):
                break
            dist = dx * np.hypot(rr - r, cc - c)
            best = max(best, np.degrees(np.arctan((dem[rr, cc] - dem[r, c]) / dist)))
        if best > -np.inf:
            angles.append(best)
    return np.mean(angles)


def test_matches_brute_force_on_random_terrain():
    rng = np.random.default_rng(1)
    dem = np.cumsum(np.cumsum(rng.normal(size=(60, 60)), 0), 1) * 3
    out = sx(dem, DX, 247, 300)
    for r, c in [(30, 30), (45, 20), (10, 50), (55, 5)]:
        assert np.isclose(out[r, c], _brute_force(dem, DX, 247, 300, r, c), atol=1e-6)


def test_nan_cells_are_skipped_not_propagated():
    dem = _ridge_dem()
    dem[20, 25] = np.nan
    out = sx(dem, DX, 270, 100)
    assert np.isnan(out[20, 25])  # no value for a NaN cell itself
    assert np.isfinite(out[20, 31])  # neighbors still computed


def test_prevailing_direction_wraps_around_north():
    # 350 and 10 degrees must average to ~0, not 180.
    d, r = prevailing_direction([5, 5], [350, 10])
    assert min(d, 360 - d) < 1e-6 and r > 0.98


def test_prevailing_direction_ignores_calms():
    d, _ = prevailing_direction([0.0, 0.5, 6.0], [90, 90, 250], min_speed=1.0)
    assert np.isclose(d, 250)
