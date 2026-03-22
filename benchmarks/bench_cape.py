"""Benchmark: MetPy vs fastcape timing comparison.

Usage:
    pixi run python benchmarks/bench_cape.py
"""

import time

import numpy as np


def generate_synthetic_soundings(ncols, nlevels=30):
    """Generate synthetic sounding data for benchmarking.

    Returns pressure (Pa), temperature (K), dewpoint (K) arrays
    with shape (ncols, nlevels).
    """
    rng = np.random.default_rng(42)

    # Pressure levels from 1000 hPa to 200 hPa
    p_1d = np.linspace(100000., 20000., nlevels)  # Pa, descending

    # Base temperature profile: standard lapse rate with noise
    T_base = 300.0 - 6.5 * np.arange(nlevels) * 300 / nlevels  # rough standard atmosphere
    Td_base = T_base - 5.0 - np.linspace(0, 15, nlevels)  # dewpoint depression increasing

    p = np.broadcast_to(p_1d[np.newaxis, :], (ncols, nlevels)).copy()
    T = T_base[np.newaxis, :] + rng.normal(0, 2, (ncols, nlevels))
    Td = Td_base[np.newaxis, :] + rng.normal(0, 1, (ncols, nlevels))

    # Ensure Td <= T
    Td = np.minimum(Td, T - 0.1)

    return p, T, Td


def bench_fastcape(p, T, Td, n_warmup=2, n_runs=5):
    """Benchmark fastcape."""
    from fastcape._numba_core import _sb_cape_cin_batch

    # Warm-up (JIT compilation)
    for _ in range(n_warmup):
        _sb_cape_cin_batch(p[:1], T[:1], Td[:1])

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        cape, cin = _sb_cape_cin_batch(p, T, Td)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times), cape, cin


def bench_metpy(p, T, Td, n_runs=3):
    """Benchmark MetPy (loop over columns)."""
    try:
        from metpy.calc import cape_cin, dewpoint_from_relative_humidity, parcel_profile
        from metpy.units import units
    except ImportError:
        print("MetPy not available for benchmarking")
        return None, None, None

    ncols = p.shape[0]

    times = []
    for _ in range(n_runs):
        capes = np.empty(ncols)
        cins = np.empty(ncols)
        start = time.perf_counter()
        for i in range(ncols):
            p_u = p[i] * units.Pa
            T_u = (T[i] - 273.15) * units.degC  # convert to what MetPy expects
            Td_u = (Td[i] - 273.15) * units.degC
            try:
                prof = parcel_profile(p_u, T_u[0], Td_u[0]).to('degC')
                c, ci = cape_cin(p_u, T_u, Td_u, prof)
                capes[i] = c.magnitude
                cins[i] = ci.magnitude
            except Exception:
                capes[i] = np.nan
                cins[i] = np.nan
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times), capes, cins


def main():
    grid_sizes = [10, 50, 100, 200]

    print(f"{'Grid':>10s}  {'Columns':>8s}  {'MetPy (s)':>10s}  {'fastcape (s)':>12s}  {'Speedup':>8s}")
    print("-" * 60)

    for n in grid_sizes:
        ncols = n * n
        p, T, Td = generate_synthetic_soundings(ncols)

        t_fast, cape_fast, cin_fast = bench_fastcape(p, T, Td)

        if ncols <= 2500:  # Only run MetPy for smaller grids (too slow otherwise)
            t_metpy, cape_metpy, cin_metpy = bench_metpy(p, T, Td, n_runs=1)
        else:
            t_metpy = None

        if t_metpy is not None:
            speedup = t_metpy / t_fast
            print(f"{n}x{n:>4d}  {ncols:>8d}  {t_metpy:>10.3f}  {t_fast:>12.4f}  {speedup:>7.1f}x")
        else:
            print(f"{n}x{n:>4d}  {ncols:>8d}  {'skip':>10s}  {t_fast:>12.4f}  {'N/A':>8s}")


if __name__ == '__main__':
    main()
