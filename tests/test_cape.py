"""Tests for the xarray/dask high-level API."""

import numpy as np
import pytest
import xarray as xr

from fastcape.cape import (
    mixed_layer_buoyancy_profile,
    mixed_layer_cape_cin,
    most_unstable_buoyancy_profile,
    most_unstable_cape_cin,
    surface_based_buoyancy_profile,
    surface_based_cape_cin,
)


def _make_sounding_dataset(p_pa, T_K, Td_K, nx=1, ny=1):
    """Create an xarray Dataset with spatial + vertical dimensions."""
    nlevels = len(p_pa)
    # Broadcast to (nx, ny, nlevels) — tile the sounding
    p_3d = np.broadcast_to(p_pa[np.newaxis, np.newaxis, :], (nx, ny, nlevels)).copy()
    T_3d = np.broadcast_to(T_K[np.newaxis, np.newaxis, :], (nx, ny, nlevels)).copy()
    Td_3d = np.broadcast_to(Td_K[np.newaxis, np.newaxis, :], (nx, ny, nlevels)).copy()

    ds = xr.Dataset(
        {
            'pressure': (['x', 'y', 'level'], p_3d),
            'temperature': (['x', 'y', 'level'], T_3d),
            'dewpoint': (['x', 'y', 'level'], Td_3d),
        },
        coords={
            'x': np.arange(nx),
            'y': np.arange(ny),
            'level': np.arange(nlevels),
        },
    )
    return ds


class TestSurfaceBasedCAPE:
    def test_basic(self, high_cape_sounding):
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=2, ny=3)

        cape, cin, lfc, el = surface_based_cape_cin(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )

        assert cape.dims == ('x', 'y')
        assert cape.shape == (2, 3)
        # All columns are the same sounding
        assert np.all(cape.values > 1000.0)
        assert np.all(cin.values <= 0.0)
        # LFC and EL should be valid pressures
        assert np.all(np.isfinite(lfc.values))
        assert np.all(np.isfinite(el.values))
        # LFC should be at higher pressure (lower altitude) than EL
        assert np.all(lfc.values > el.values)

    def test_1d(self, high_cape_sounding):
        """Test with a 1D sounding (no spatial dims)."""
        p, T, Td = high_cape_sounding
        ds = xr.Dataset({
            'pressure': ('level', p),
            'temperature': ('level', T),
            'dewpoint': ('level', Td),
        })
        cape, cin, lfc, el = surface_based_cape_cin(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert float(cape) > 1000.0
        assert float(lfc) > float(el)

    def test_dask(self, high_cape_sounding):
        """Test with dask-backed arrays."""
        dask = pytest.importorskip("dask")

        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=4, ny=4)

        # Chunk along spatial dims only (NOT the vertical)
        ds_chunked = ds.chunk({'x': 2, 'y': 2})

        cape, cin, lfc, el = surface_based_cape_cin(
            ds_chunked['pressure'], ds_chunked['temperature'], ds_chunked['dewpoint']
        )

        # Should be lazy
        assert hasattr(cape.data, 'dask')
        assert hasattr(lfc.data, 'dask')

        # Compute and verify
        cape_vals = cape.compute().values
        cin_vals = cin.compute().values
        assert np.all(cape_vals > 1000.0)
        assert np.all(cin_vals <= 0.0)

    def test_chunked_vs_unchunked(self, high_cape_sounding):
        """Chunked and unchunked should give the same results."""
        dask = pytest.importorskip("dask")

        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=3, ny=3)
        ds_chunked = ds.chunk({'x': 2, 'y': 2})

        cape1, cin1, lfc1, el1 = surface_based_cape_cin(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        cape2, cin2, lfc2, el2 = surface_based_cape_cin(
            ds_chunked['pressure'], ds_chunked['temperature'], ds_chunked['dewpoint']
        )

        np.testing.assert_allclose(cape1.values, cape2.compute().values, rtol=1e-10)
        np.testing.assert_allclose(cin1.values, cin2.compute().values, rtol=1e-10)
        np.testing.assert_allclose(lfc1.values, lfc2.compute().values, rtol=1e-10)
        np.testing.assert_allclose(el1.values, el2.compute().values, rtol=1e-10)


class TestMostUnstableCAPE:
    def test_basic(self, high_cape_sounding):
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td)
        cape, cin, lfc, el = most_unstable_cape_cin(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert cape.values.item() > 1000.0


class TestMixedLayerCAPE:
    def test_basic(self, high_cape_sounding):
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td)
        cape, cin, lfc, el = mixed_layer_cape_cin(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert cape.values.item() >= 0.0
        assert cin.values.item() <= 0.0


class TestBuoyancyProfile:
    def test_sb_shape_and_dims(self, high_cape_sounding):
        """Output should preserve spatial + vertical dimensions."""
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=2, ny=3)
        B = surface_based_buoyancy_profile(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert B.dims == ('x', 'y', 'level')
        assert B.shape == (2, 3, len(p))

    def test_sb_has_positive_buoyancy(self, high_cape_sounding):
        """High-CAPE sounding should have positive buoyancy values."""
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td)
        B = surface_based_buoyancy_profile(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert np.any(B.values > 0)

    def test_mu_runs(self, high_cape_sounding):
        """MU buoyancy profile should compute via xarray."""
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td)
        B = most_unstable_buoyancy_profile(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert B.dims == ('x', 'y', 'level')

    def test_ml_runs(self, high_cape_sounding):
        """ML buoyancy profile should compute via xarray."""
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td)
        B = mixed_layer_buoyancy_profile(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        assert B.dims == ('x', 'y', 'level')

    def test_dask(self, high_cape_sounding):
        """Should work with dask-backed arrays."""
        dask = pytest.importorskip("dask")
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=4, ny=4)
        ds_chunked = ds.chunk({'x': 2, 'y': 2})

        B = surface_based_buoyancy_profile(
            ds_chunked['pressure'], ds_chunked['temperature'], ds_chunked['dewpoint']
        )
        assert hasattr(B.data, 'dask')
        B_vals = B.compute().values
        assert B_vals.shape == (4, 4, len(p))

    def test_chunked_vs_unchunked(self, high_cape_sounding):
        """Chunked and unchunked should give the same results."""
        dask = pytest.importorskip("dask")
        p, T, Td = high_cape_sounding
        ds = _make_sounding_dataset(p, T, Td, nx=3, ny=3)
        ds_chunked = ds.chunk({'x': 2, 'y': 2})

        B1 = surface_based_buoyancy_profile(
            ds['pressure'], ds['temperature'], ds['dewpoint']
        )
        B2 = surface_based_buoyancy_profile(
            ds_chunked['pressure'], ds_chunked['temperature'], ds_chunked['dewpoint']
        )
        np.testing.assert_allclose(B1.values, B2.compute().values, rtol=1e-10)
