# fastcape

Fast CAPE/CIN calculations for gridded atmospheric data using Numba.
Reimplements MetPy's core thermodynamic physics with raw float arrays in SI
units, achieving huge speedup over MetPy on typical grids.

MetPy's CAPE/CIN calculations are designed for single-sounding analysis with
pint unit tracking. This makes them too slow for high-resolution gridded data.

fastcape works by compiling the physics into Numba `@njit` kernels with
`prange` thread parallelism, and wrapping with `xarray.apply_ufunc` for dask
chunk-level parallelism. By default it uses Bolton approximation instead of
Romps for LCL computation, which leads to slightly different results from MetPy
(~2%). See [physics notes](#physics-notes).

## Installation

### With pixi (recommended)

```bash
git clone <repo-url> && cd fastcape
pixi install
pixi run pip install -e .
```

### With pip

```bash
pip install -e .
```

## Usage

All inputs must be in **SI units**: pressure in Pa, temperature and dewpoint in
K. Pressure arrays must be **descending** (surface first, top of atmosphere
last). Vertical dimension must not be chunked in dask.

All three functions return **four** values: `cape`, `cin`, `lfc`, `el`.

### Surface-based CAPE/CIN

```python
import numpy as np
import xarray as xr
from fastcape import surface_based_cape_cin

# Create xarray data (e.g. from ERA5, ICON, etc.)
ds = xr.Dataset({
    'pressure':    (['x', 'y', 'level'], p_array),    # Pa
    'temperature': (['x', 'y', 'level'], T_array),    # K
    'dewpoint':    (['x', 'y', 'level'], Td_array),   # K
})

cape, cin, lfc, el = surface_based_cape_cin(
    ds['pressure'], ds['temperature'], ds['dewpoint'],
    vertical_dim='level',
)
# cape: xr.DataArray [J/kg], shape (x, y)
# cin:  xr.DataArray [J/kg], shape (x, y), always <= 0
# lfc:  xr.DataArray [Pa],   shape (x, y), NaN where no LFC exists
# el:   xr.DataArray [Pa],   shape (x, y), NaN where no EL exists
```

### Most-unstable CAPE/CIN

Finds the parcel with maximum equivalent potential temperature in the bottom
300 hPa:

```python
from fastcape import most_unstable_cape_cin

cape, cin, lfc, el = most_unstable_cape_cin(
    ds['pressure'], ds['temperature'], ds['dewpoint'],
)
```

### Mixed-layer CAPE/CIN

Uses pressure-weighted mean temperature and dewpoint over the bottom 100 hPa:

```python
from fastcape import mixed_layer_cape_cin

cape, cin, lfc, el = mixed_layer_cape_cin(
    ds['pressure'], ds['temperature'], ds['dewpoint'],
)
```

### Converting specific humidity to dewpoint

If your data provides specific humidity instead of dewpoint (common for
ERA5, ICON, etc.):

```python
from fastcape import dewpoint_from_specific_humidity

# All inputs as xr.DataArrays or numpy arrays in SI units
Td = dewpoint_from_specific_humidity(pressure_pa, temperature_K, q_kgkg)
```

### With dask (lazy / chunked)

Chunk along spatial dimensions only — the vertical dimension must **not** be
chunked:

```python
ds_chunked = ds.chunk({'x': 50, 'y': 50})  # do NOT chunk 'level'

cape, cin, lfc, el = surface_based_cape_cin(
    ds_chunked['pressure'],
    ds_chunked['temperature'],
    ds_chunked['dewpoint'],
)
# All outputs are lazy (dask-backed) — call .compute() to materialize
cape_values = cape.compute()
```

### Controlling parallelism

fastcape uses Numba's `prange` for thread-level parallelism within each chunk.
By default it uses all available cores.

```python
import numba
numba.set_num_threads(4)  # limit to 4 threads
```

Or via environment variable:

```bash
NUMBA_NUM_THREADS=4 python my_script.py
```

### Low-level Numba kernels

For direct use without xarray:

```python
import numpy as np
from fastcape._numba_core import _cape_cin_column, _sb_cape_cin_batch

# Single column
p = np.array([100000., 95000., 85000., 70000., 50000., 30000., 20000.])  # Pa
T = np.array([300., 296., 287., 275., 253., 225., 210.])                  # K
Td = np.array([290., 285., 275., 260., 240., 210., 195.])                 # K

cape, cin, lfc, el = _cape_cin_column(p, T, Td)

# Batch (parallel over columns)
# Arrays shape: (ncols, nlevels)
p_2d = np.tile(p, (1000, 1))
T_2d = np.tile(T, (1000, 1))
Td_2d = np.tile(Td, (1000, 1))

cape_arr, cin_arr, lfc_arr, el_arr = _sb_cape_cin_batch(p_2d, T_2d, Td_2d)
```

### Vectorized helpers

```python
from fastcape import lcl_romps, equivalent_potential_temperature

# LCL using Romps (2017) exact analytical solution (vectorized)
p_lcl, T_lcl = lcl_romps(
    np.array([100000.]),  # Pa
    np.array([300.]),     # K
    np.array([290.]),     # K
)

# Bolton (1980) equivalent potential temperature (vectorized)
theta_e = equivalent_potential_temperature(
    np.array([85000.]),   # Pa
    np.array([293.15]),   # K
    np.array([291.15]),   # K
)
```

## Benchmarks

Run the benchmark:

```bash
pixi run bench
```

Typical results on Apple M-series (single socket):

| Grid | Columns | MetPy | fastcape | Speedup |
|------|---------|-------|----------|---------|
| 10x10 | 100 | 0.36 s | 0.5 ms | ~650x |
| 50x50 | 2,500 | 8.9 s | 12 ms | ~750x |
| 100x100 | 10,000 | — | 49 ms | — |
| 200x200 | 40,000 | — | 183 ms | — |

MetPy is too slow to benchmark beyond 2,500 columns. Each column has 30
vertical levels. The first call includes ~1 s of Numba JIT compilation overhead
(excluded from timings above).

## Testing

```bash
pixi run test
```

Runs 27 tests validating against MetPy reference values:

- Saturation vapor pressure, mixing ratio, virtual temperature
- Dry lapse rate, moist lapse rate (within 0.1 K of MetPy)
- LCL (Romps), equivalent potential temperature
- Full CAPE/CIN/LFC/EL pipeline (within ~2% of MetPy for CAPE)
- xarray dimensions, dask lazy evaluation, chunked vs unchunked agreement

## Physics notes

- **Saturation vapor pressure**: Ambaum (2020) Eq. 13, same formula as MetPy
- **Moist adiabat**: RK4 with 4 substeps per level (vs MetPy's scipy LSODA).
  Agrees within 0.1 K
- **LCL**: Bolton (1980) approximation by default inside column kernels (~0.1 K
  accuracy). Romps (2017) exact solution available as a vectorized helper
- **Virtual temperature correction**: Applied to both parcel and environment
  profiles
- **Integration**: Trapezoidal in ln(p) space with interpolated zero-crossings,
  matching MetPy's approach
- **LFC**: First intersection where parcel becomes warmer than environment
  (bottom LFC). Falls back to LCL when parcel is immediately buoyant
- **EL**: Last intersection where parcel crosses below environment (top EL)
- **CAPE/CIN difference vs MetPy**: ~2% systematic offset from using Bolton LCL
  instead of Romps LCL. This is within typical observational uncertainty
