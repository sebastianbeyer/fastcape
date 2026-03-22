"""High-level xarray/dask API for CAPE/CIN calculations.

All public functions accept xarray DataArrays (with optional dask backing)
and return xarray DataArrays. Inputs must be in SI units: Pa for pressure,
K for temperature and dewpoint.
"""

import numpy as np
import xarray as xr

from ._numba_core import (
    _ml_cape_cin_batch,
    _mu_cape_cin_batch,
    _sb_cape_cin_batch,
)


# ---------------------------------------------------------------------------
# Input validation (runs once, before any computation)
# ---------------------------------------------------------------------------

def _validate_inputs(pressure, temperature, dewpoint, vertical_dim):
    """Check inputs for common mistakes before dispatching to Numba.

    Raises
    ------
    ValueError
        If inputs are inconsistent or likely in wrong units.
    """
    # 1. vertical_dim exists
    for name, da in [("pressure", pressure), ("temperature", temperature),
                     ("dewpoint", dewpoint)]:
        if vertical_dim not in da.dims:
            raise ValueError(
                f"{name!r} has no dimension {vertical_dim!r}. "
                f"Available dims: {list(da.dims)}"
            )

    # 2. Shapes must match
    if not (pressure.sizes == temperature.sizes == dewpoint.sizes):
        raise ValueError(
            f"Shape mismatch: pressure {dict(pressure.sizes)}, "
            f"temperature {dict(temperature.sizes)}, "
            f"dewpoint {dict(dewpoint.sizes)}"
        )

    # 3. Vertical dim must not be chunked (when dask-backed)
    if hasattr(pressure, "chunks") and pressure.chunks is not None:
        vdim_idx = list(pressure.dims).index(vertical_dim)
        chunks_along_v = pressure.chunks[vdim_idx]
        if len(chunks_along_v) > 1:
            raise ValueError(
                f"The vertical dimension {vertical_dim!r} must not be chunked "
                f"(found {len(chunks_along_v)} chunks). "
                f"Rechunk with .chunk({{{vertical_dim!r}: -1}}) first."
            )

    # 4. Check pressure looks like Pa (not hPa)
    # Sample first element along vertical dim (works for lazy arrays too)
    try:
        p_first = float(pressure.isel({vertical_dim: 0}).values.flat[0])
        p_last = float(pressure.isel({vertical_dim: -1}).values.flat[0])
    except Exception:
        return  # can't check, skip

    if not np.isnan(p_first):
        if p_first < 2000.0:
            raise ValueError(
                f"Pressure values look like hPa (first={p_first:.1f}). "
                f"fastcape requires Pa. Multiply by 100."
            )

    # 5. Check pressure is descending (surface first)
    if not np.isnan(p_first) and not np.isnan(p_last):
        if p_first < p_last:
            raise ValueError(
                f"Pressure must be descending (surface first, top last). "
                f"Got first={p_first:.0f} Pa < last={p_last:.0f} Pa. "
                f"Sort with .sortby({vertical_dim!r}, ascending=False)."
            )

    # 6. Check temperature looks like K (not degC)
    try:
        t_sample = float(temperature.isel({vertical_dim: 0}).values.flat[0])
    except Exception:
        return

    if not np.isnan(t_sample) and t_sample < 100.0:
        raise ValueError(
            f"Temperature values look like degC (sample={t_sample:.1f}). "
            f"fastcape requires K. Add 273.15."
        )

    # 7. Check dewpoint <= temperature
    try:
        td_sample = float(dewpoint.isel({vertical_dim: 0}).values.flat[0])
    except Exception:
        return

    if not np.isnan(td_sample) and not np.isnan(t_sample):
        if td_sample > t_sample + 1.0:
            raise ValueError(
                f"Dewpoint ({td_sample:.1f} K) exceeds temperature ({t_sample:.1f} K) "
                f"at the surface. Check input ordering or units."
            )


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def _make_wrapper(batch_func):
    """Create an apply_ufunc wrapper for a batch function returning 4 arrays."""
    def wrapper(pressure, temperature, dewpoint):
        orig_shape = pressure.shape[:-1]
        nlevels = pressure.shape[-1]
        ncols = int(np.prod(orig_shape)) if len(orig_shape) > 0 else 1

        p2d = np.ascontiguousarray(pressure.reshape(ncols, nlevels))
        t2d = np.ascontiguousarray(temperature.reshape(ncols, nlevels))
        td2d = np.ascontiguousarray(dewpoint.reshape(ncols, nlevels))

        cape, cin, lfc, el = batch_func(p2d, t2d, td2d)
        return (
            cape.reshape(orig_shape),
            cin.reshape(orig_shape),
            lfc.reshape(orig_shape),
            el.reshape(orig_shape),
        )
    return wrapper


_sb_wrapper = _make_wrapper(_sb_cape_cin_batch)
_mu_wrapper = _make_wrapper(_mu_cape_cin_batch)
_ml_wrapper = _make_wrapper(_ml_cape_cin_batch)

_output_core_dims = [[], [], [], []]
_output_dtypes = [np.float64] * 4


def surface_based_cape_cin(pressure, temperature, dewpoint, vertical_dim='level'):
    """Compute surface-based CAPE, CIN, LFC, and EL.

    Parameters
    ----------
    pressure : xr.DataArray
        Pressure [Pa], with a vertical dimension.
    temperature : xr.DataArray
        Temperature [K].
    dewpoint : xr.DataArray
        Dewpoint temperature [K].
    vertical_dim : str
        Name of the vertical dimension. Must not be chunked in dask.

    Returns
    -------
    cape : xr.DataArray, CAPE [J/kg]
    cin  : xr.DataArray, CIN [J/kg] (<= 0)
    lfc  : xr.DataArray, LFC pressure [Pa] (NaN if no LFC)
    el   : xr.DataArray, EL pressure [Pa] (NaN if no EL)
    """
    _validate_inputs(pressure, temperature, dewpoint, vertical_dim)
    return xr.apply_ufunc(
        _sb_wrapper,
        pressure, temperature, dewpoint,
        input_core_dims=[[vertical_dim]] * 3,
        output_core_dims=_output_core_dims,
        vectorize=False,
        dask='parallelized',
        output_dtypes=_output_dtypes,
    )


def most_unstable_cape_cin(pressure, temperature, dewpoint, vertical_dim='level'):
    """Compute most-unstable CAPE, CIN, LFC, and EL.

    Finds the parcel with maximum equivalent potential temperature
    in the bottom 300 hPa and computes CAPE/CIN from that level.

    Parameters
    ----------
    pressure : xr.DataArray
        Pressure [Pa].
    temperature : xr.DataArray
        Temperature [K].
    dewpoint : xr.DataArray
        Dewpoint temperature [K].
    vertical_dim : str
        Name of the vertical dimension.

    Returns
    -------
    cape : xr.DataArray, CAPE [J/kg]
    cin  : xr.DataArray, CIN [J/kg] (<= 0)
    lfc  : xr.DataArray, LFC pressure [Pa] (NaN if no LFC)
    el   : xr.DataArray, EL pressure [Pa] (NaN if no EL)
    """
    _validate_inputs(pressure, temperature, dewpoint, vertical_dim)
    return xr.apply_ufunc(
        _mu_wrapper,
        pressure, temperature, dewpoint,
        input_core_dims=[[vertical_dim]] * 3,
        output_core_dims=_output_core_dims,
        vectorize=False,
        dask='parallelized',
        output_dtypes=_output_dtypes,
    )


def mixed_layer_cape_cin(pressure, temperature, dewpoint, vertical_dim='level'):
    """Compute mixed-layer CAPE, CIN, LFC, and EL.

    Uses pressure-weighted mean temperature and dewpoint over the
    bottom 100 hPa as the starting parcel.

    Parameters
    ----------
    pressure : xr.DataArray
        Pressure [Pa].
    temperature : xr.DataArray
        Temperature [K].
    dewpoint : xr.DataArray
        Dewpoint temperature [K].
    vertical_dim : str
        Name of the vertical dimension.

    Returns
    -------
    cape : xr.DataArray, CAPE [J/kg]
    cin  : xr.DataArray, CIN [J/kg] (<= 0)
    lfc  : xr.DataArray, LFC pressure [Pa] (NaN if no LFC)
    el   : xr.DataArray, EL pressure [Pa] (NaN if no EL)
    """
    _validate_inputs(pressure, temperature, dewpoint, vertical_dim)
    return xr.apply_ufunc(
        _ml_wrapper,
        pressure, temperature, dewpoint,
        input_core_dims=[[vertical_dim]] * 3,
        output_core_dims=_output_core_dims,
        vectorize=False,
        dask='parallelized',
        output_dtypes=_output_dtypes,
    )
