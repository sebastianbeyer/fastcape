"""Numba-compiled per-column kernels for CAPE/CIN calculation.

All inputs/outputs are raw floats in SI units: pressure in Pa, temperature in K.
Pressure arrays are expected in descending order (surface first, top last).
"""

import math

import numpy as np
from numba import njit, prange

from . import _constants as c

# ---------------------------------------------------------------------------
# Scalar thermodynamic helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _water_latent_heat_vaporization(T):
    """Variable latent heat of vaporization [J/kg]. Ambaum (2020) Eq 15."""
    return c.Lv - (c.Cp_l - c.Cp_v) * (T - c.T0)


@njit(cache=True)
def _sat_vapor_pressure(T):
    """Saturation vapor pressure over liquid water [Pa]. Ambaum (2020) Eq 13."""
    L = _water_latent_heat_vaporization(T)
    heat_power = (c.Cp_l - c.Cp_v) / c.Rv
    exp_term = (c.Lv / c.T0 - L / T) / c.Rv
    return c.sat_pressure_0c * (c.T0 / T) ** heat_power * math.exp(exp_term)


@njit(cache=True)
def _sat_mixing_ratio(p, T):
    """Saturation mixing ratio [kg/kg]."""
    es = _sat_vapor_pressure(T)
    return c.epsilon * es / (p - es)


@njit(cache=True)
def _virtual_temperature(T, w):
    """Virtual temperature [K] from temperature and mixing ratio."""
    return T * (w + c.epsilon) / (c.epsilon * (1.0 + w))


@njit(cache=True)
def _dry_lapse(p, T0, p0):
    """Dry adiabatic temperature at pressure p, starting from T0 at p0."""
    return T0 * (p / p0) ** c.kappa


@njit(cache=True)
def _moist_lapse_rhs(p, T):
    """RHS of the moist adiabat ODE: dT/dp. Bakhshaii (2013)."""
    rs = _sat_mixing_ratio(p, T)
    numer = c.Rd * T + c.Lv * rs
    denom = c.Cp_d + (c.Lv * c.Lv * rs * c.epsilon) / (c.Rd * T * T)
    return numer / (denom * p)


@njit(cache=True)
def _moist_lapse_rk4(pressures, T_start, p_start):
    """Integrate moist adiabat along pressure levels using RK4.

    Parameters
    ----------
    pressures : 1D array
        Pressure levels (Pa), in the order to integrate along.
    T_start : float
        Starting temperature (K).
    p_start : float
        Starting pressure (Pa).

    Returns
    -------
    temps : 1D array
        Temperature at each pressure level.
    """
    n = len(pressures)
    temps = np.empty(n, dtype=np.float64)
    T = T_start
    p_prev = p_start
    n_substeps = 4

    for i in range(n):
        p_target = pressures[i]
        dp_total = p_target - p_prev
        dp = dp_total / n_substeps
        for _ in range(n_substeps):
            k1 = dp * _moist_lapse_rhs(p_prev, T)
            k2 = dp * _moist_lapse_rhs(p_prev + 0.5 * dp, T + 0.5 * k1)
            k3 = dp * _moist_lapse_rhs(p_prev + 0.5 * dp, T + 0.5 * k2)
            k4 = dp * _moist_lapse_rhs(p_prev + dp, T + k3)
            T = T + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            p_prev = p_prev + dp
        temps[i] = T
        p_prev = p_target  # reset to exact target to avoid drift
    return temps


# ---------------------------------------------------------------------------
# LCL (Bolton approximation)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _lcl_bolton(p, T, Td):
    """LCL pressure and temperature using Bolton (1980) approximation.

    Parameters
    ----------
    p : float
        Surface pressure [Pa].
    T : float
        Surface temperature [K].
    Td : float
        Surface dewpoint [K].

    Returns
    -------
    p_lcl, T_lcl : floats
        LCL pressure [Pa] and temperature [K].
    """
    T_lcl = 1.0 / (1.0 / (Td - 56.0) + math.log(T / Td) / 800.0) + 56.0
    p_lcl = p * (T_lcl / T) ** (c.Cp_d / c.Rd)
    return p_lcl, T_lcl


# ---------------------------------------------------------------------------
# Parcel profile
# ---------------------------------------------------------------------------

@njit(cache=True)
def _parcel_profile(pressure, T_sfc, Td_sfc):
    """Compute parcel temperature profile (dry below LCL, moist above).

    Parameters
    ----------
    pressure : 1D array
        Pressure levels [Pa], descending (surface first).
    T_sfc : float
        Surface temperature [K].
    Td_sfc : float
        Surface dewpoint [K].

    Returns
    -------
    profile : 1D array
        Parcel temperature [K] at each pressure level.
    p_lcl : float
        LCL pressure [Pa].
    T_lcl : float
        LCL temperature [K].
    """
    n = len(pressure)
    profile = np.empty(n, dtype=np.float64)
    p_lcl, T_lcl = _lcl_bolton(pressure[0], T_sfc, Td_sfc)

    # Find split index: first level where pressure < p_lcl
    split = n  # default: all dry
    for i in range(n):
        if pressure[i] < p_lcl:
            split = i
            break

    # Dry adiabatic below LCL
    for i in range(split):
        profile[i] = _dry_lapse(pressure[i], T_sfc, pressure[0])

    # Moist adiabatic above LCL
    if split < n:
        moist_pressures = pressure[split:]
        moist_temps = _moist_lapse_rk4(moist_pressures, T_lcl, p_lcl)
        for i in range(len(moist_temps)):
            profile[split + i] = moist_temps[i]

    return profile, p_lcl, T_lcl


# ---------------------------------------------------------------------------
# Intersection finding
# ---------------------------------------------------------------------------

@njit(cache=True)
def _find_intersections(x, a, b, direction):
    """Find intersections of two curves sharing x-coordinates.

    Works in log(x) space for pressure coordinates.

    Parameters
    ----------
    x : 1D array
        x values (e.g. pressure in Pa).
    a, b : 1D arrays
        Two y-value curves.
    direction : int
        0 = all, 1 = increasing (a crosses above b), -1 = decreasing.

    Returns
    -------
    x_int, y_int : 1D arrays
        Intersection coordinates.
    """
    n = len(x)
    # Temporary storage (max possible intersections = n-1)
    xi = np.empty(n - 1, dtype=np.float64)
    yi = np.empty(n - 1, dtype=np.float64)
    signs = np.empty(n - 1, dtype=np.float64)
    count = 0

    log_x = np.empty(n, dtype=np.float64)
    for i in range(n):
        log_x[i] = math.log(x[i])

    diff_prev = a[0] - b[0]
    for i in range(1, n):
        diff_curr = a[i] - b[i]
        # Check for sign change
        if diff_prev * diff_curr < 0.0:
            lx0 = log_x[i - 1]
            lx1 = log_x[i]
            dy0 = a[i - 1] - b[i - 1]
            dy1 = a[i] - b[i]
            lx_int = (dy1 * lx0 - dy0 * lx1) / (dy1 - dy0)
            y_int = ((lx_int - lx0) / (lx1 - lx0)) * (a[i] - a[i - 1]) + a[i - 1]
            xi[count] = math.exp(lx_int)
            yi[count] = y_int
            # sign_change: positive means a crossed above b (increasing)
            signs[count] = 1.0 if diff_curr > 0.0 else -1.0
            count += 1
        diff_prev = diff_curr

    if count == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if direction == 0:
        return xi[:count].copy(), yi[:count].copy()

    # Filter by direction
    out_x = np.empty(count, dtype=np.float64)
    out_y = np.empty(count, dtype=np.float64)
    out_count = 0
    for i in range(count):
        if (direction == 1 and signs[i] > 0.0) or (direction == -1 and signs[i] < 0.0):
            out_x[out_count] = xi[i]
            out_y[out_count] = yi[i]
            out_count += 1
    return out_x[:out_count].copy(), out_y[:out_count].copy()


@njit(cache=True)
def _find_append_zero_crossings(x, y):
    """Insert interpolated zero-crossing points into (x, y) arrays.

    Works in log(x) space. x must be descending (pressure).
    Returns sorted arrays with zero-crossings inserted.
    """
    n = len(x)
    # Find zero crossings of y
    zeros_x = np.empty(n, dtype=np.float64)
    zeros_y = np.empty(n, dtype=np.float64)
    nz = 0

    for i in range(1, n):
        if y[i - 1] * y[i] < 0.0:
            lx0 = math.log(x[i - 1])
            lx1 = math.log(x[i])
            # Linear interpolation in log-p space
            frac = -y[i - 1] / (y[i] - y[i - 1])
            lx_cross = lx0 + frac * (lx1 - lx0)
            zeros_x[nz] = math.exp(lx_cross)
            zeros_y[nz] = 0.0
            nz += 1

    # Concatenate original + crossings
    total = n + nz
    out_x = np.empty(total, dtype=np.float64)
    out_y = np.empty(total, dtype=np.float64)
    for i in range(n):
        out_x[i] = x[i]
        out_y[i] = y[i]
    for i in range(nz):
        out_x[n + i] = zeros_x[i]
        out_y[n + i] = zeros_y[i]

    # Sort by descending pressure (= descending x)
    idx = np.argsort(out_x)[::-1]
    sorted_x = np.empty(total, dtype=np.float64)
    sorted_y = np.empty(total, dtype=np.float64)
    for i in range(total):
        sorted_x[i] = out_x[idx[i]]
        sorted_y[i] = out_y[idx[i]]

    # Remove near-duplicates
    keep = np.ones(total, dtype=np.bool_)
    for i in range(1, total):
        if abs(sorted_x[i] - sorted_x[i - 1]) < 1e-6:
            keep[i] = False
    nkeep = 0
    for i in range(total):
        if keep[i]:
            nkeep += 1
    rx = np.empty(nkeep, dtype=np.float64)
    ry = np.empty(nkeep, dtype=np.float64)
    j = 0
    for i in range(total):
        if keep[i]:
            rx[j] = sorted_x[i]
            ry[j] = sorted_y[i]
            j += 1
    return rx, ry


# ---------------------------------------------------------------------------
# CAPE/CIN column kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _integrate_cape_cin(p, tv_par, tv_env, p_lfc, p_el):
    """Trapezoidal integration for CAPE and CIN in log(p) space.

    Returns
    -------
    cape, cin : floats
    """
    n = len(p)
    diff = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff[i] = tv_par[i] - tv_env[i]

    x_zc, y_zc = _find_append_zero_crossings(p, diff)
    nzc = len(x_zc)

    cape = 0.0
    cin = 0.0
    tol = 1.0  # Pa tolerance for boundary comparison

    for i in range(1, nzc):
        p_mid_lo = x_zc[i]      # lower pressure (higher altitude)
        p_mid_hi = x_zc[i - 1]  # higher pressure (lower altitude)

        dp_log = math.log(x_zc[i - 1]) - math.log(x_zc[i])

        # CAPE region: between LFC and EL
        if p_mid_hi <= p_lfc + tol and p_mid_lo >= p_el - tol:
            cape += 0.5 * (y_zc[i - 1] + y_zc[i]) * dp_log

        # CIN region: between surface and LFC
        if p_mid_hi <= p[0] + tol and p_mid_lo >= p_lfc - tol:
            cin += 0.5 * (y_zc[i - 1] + y_zc[i]) * dp_log

    cape = c.Rd * cape
    cin = c.Rd * cin

    if cape < 0.0:
        cape = 0.0
    if cin > 0.0:
        cin = 0.0

    return cape, cin


@njit(cache=True)
def _compute_column(pressure, temperature, dewpoint, T_parcel, Td_parcel, p_parcel):
    """Core column computation: CAPE, CIN, LFC, EL from arbitrary parcel.

    Parameters
    ----------
    pressure, temperature, dewpoint : 1D arrays
        Full sounding [Pa, K, K], descending.
    T_parcel, Td_parcel, p_parcel : float
        Parcel starting conditions.

    Returns
    -------
    cape : float [J/kg]
    cin  : float [J/kg], <= 0
    lfc  : float [Pa], NaN if no LFC
    el   : float [Pa], NaN if no EL
    """
    NAN = np.nan
    n = len(pressure)
    if n < 2:
        return 0.0, 0.0, NAN, NAN

    # Build parcel profile
    p_lcl, T_lcl = _lcl_bolton(p_parcel, T_parcel, Td_parcel)

    prof = np.empty(n, dtype=np.float64)
    split = n
    for i in range(n):
        if pressure[i] >= p_lcl:
            prof[i] = _dry_lapse(pressure[i], T_parcel, p_parcel)
        else:
            split = i
            break
    if split < n:
        moist_temps = _moist_lapse_rk4(pressure[split:], T_lcl, p_lcl)
        for j in range(len(moist_temps)):
            prof[split + j] = moist_temps[j]

    # Virtual temperature correction
    w_parcel = _sat_mixing_ratio(p_parcel, Td_parcel)

    tv_env = np.empty(n, dtype=np.float64)
    tv_par = np.empty(n, dtype=np.float64)
    for i in range(n):
        w_env = _sat_mixing_ratio(pressure[i], dewpoint[i])
        tv_env[i] = _virtual_temperature(temperature[i], w_env)
        if pressure[i] >= p_lcl:
            tv_par[i] = _virtual_temperature(prof[i], w_parcel)
        else:
            w_sat = _sat_mixing_ratio(pressure[i], prof[i])
            tv_par[i] = _virtual_temperature(prof[i], w_sat)

    # Find LFC (first increasing intersection, skipping surface)
    start = 1
    lfc_x, _ = _find_intersections(pressure[start:], tv_par[start:], tv_env[start:], 1)
    if len(lfc_x) == 0:
        any_positive = False
        for i in range(n):
            if pressure[i] < p_lcl and tv_par[i] > tv_env[i]:
                any_positive = True
                break
        if not any_positive:
            return 0.0, 0.0, NAN, NAN
        p_lfc = p_lcl
    else:
        p_lfc = lfc_x[0]  # bottom LFC (highest pressure)

    # Find EL (last decreasing intersection)
    el_x, _ = _find_intersections(pressure[start:], tv_par[start:], tv_env[start:], -1)
    if len(el_x) == 0:
        p_el = pressure[n - 1]
        el_found = False
    else:
        p_el = el_x[len(el_x) - 1]  # top EL (lowest pressure)
        el_found = True

    # Integrate
    cape, cin = _integrate_cape_cin(pressure, tv_par, tv_env, p_lfc, p_el)

    return cape, cin, p_lfc, p_el if el_found else NAN


@njit(cache=True)
def _cape_cin_column(pressure, temperature, dewpoint):
    """Compute CAPE, CIN, LFC, EL for a single surface-based sounding.

    Parameters
    ----------
    pressure : 1D array, Pressure [Pa], descending (surface first).
    temperature : 1D array, Temperature [K].
    dewpoint : 1D array, Dewpoint temperature [K].

    Returns
    -------
    cape, cin, lfc, el : floats
        CAPE [J/kg], CIN [J/kg] (<= 0), LFC [Pa], EL [Pa].
        LFC/EL are NaN when not found.
    """
    n = len(pressure)
    if n < 2:
        return 0.0, 0.0, np.nan, np.nan

    # Strip trailing NaNs
    valid_n = n
    for i in range(n - 1, -1, -1):
        if math.isnan(pressure[i]) or math.isnan(temperature[i]) or math.isnan(dewpoint[i]):
            valid_n = i
        else:
            break
    if valid_n < 2:
        return 0.0, 0.0, np.nan, np.nan

    return _compute_column(
        pressure[:valid_n], temperature[:valid_n], dewpoint[:valid_n],
        temperature[0], dewpoint[0], pressure[0],
    )


# ---------------------------------------------------------------------------
# MU-CAPE and ML-CAPE column kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _equivalent_potential_temperature_bolton(p, T, Td):
    """Bolton (1980) equivalent potential temperature [K]."""
    r = _sat_mixing_ratio(p, Td)
    e = _sat_vapor_pressure(Td)
    t_l = 56.0 + 1.0 / (1.0 / (Td - 56.0) + math.log(T / Td) / 800.0)
    # Potential temperature using (p - e) as dry pressure
    theta = T * (c.P0 / (p - e)) ** c.kappa
    th_l = theta * (T / t_l) ** (0.28 * r)
    return th_l * math.exp(r * (1.0 + 0.448 * r) * (3036.0 / t_l - 1.78))


@njit(cache=True)
def _mu_cape_cin_column(pressure, temperature, dewpoint, depth_pa=30000.0):
    """Most-unstable CAPE/CIN/LFC/EL. Finds max theta_e in bottom depth_pa."""
    n = len(pressure)
    if n < 2:
        return 0.0, 0.0, np.nan, np.nan

    p_bottom = pressure[0]
    max_theta_e = -1e30
    best_idx = 0

    for i in range(n):
        if p_bottom - pressure[i] > depth_pa:
            break
        theta_e = _equivalent_potential_temperature_bolton(pressure[i], temperature[i], dewpoint[i])
        if theta_e > max_theta_e:
            max_theta_e = theta_e
            best_idx = i

    return _compute_column(
        pressure, temperature, dewpoint,
        temperature[best_idx], dewpoint[best_idx], pressure[best_idx],
    )


@njit(cache=True)
def _ml_cape_cin_column(pressure, temperature, dewpoint, depth_pa=10000.0):
    """Mixed-layer CAPE/CIN/LFC/EL. Pressure-weighted mean T, Td over bottom depth_pa."""
    n = len(pressure)
    if n < 2:
        return 0.0, 0.0, np.nan, np.nan

    p_bottom = pressure[0]
    sum_T = 0.0
    sum_Td = 0.0
    sum_w = 0.0

    for i in range(n):
        if p_bottom - pressure[i] > depth_pa:
            break
        # Pressure-weighted mean: weight by layer thickness
        if i == 0:
            dp = 0.5 * (pressure[0] - pressure[1]) if n > 1 else 1.0
        elif i < n - 1 and p_bottom - pressure[i + 1] <= depth_pa:
            dp = 0.5 * (pressure[i - 1] - pressure[i + 1])
        else:
            # Last level in layer
            dp = 0.5 * (pressure[i - 1] - pressure[i])
        sum_T += temperature[i] * dp
        sum_Td += dewpoint[i] * dp
        sum_w += dp

    if sum_w <= 0.0:
        return 0.0, 0.0, np.nan, np.nan

    T_mean = sum_T / sum_w
    Td_mean = sum_Td / sum_w

    return _compute_column(
        pressure, temperature, dewpoint,
        T_mean, Td_mean, p_bottom,
    )


# ---------------------------------------------------------------------------
# Batch processing (parallel over columns)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _sb_cape_cin_batch(pressure_2d, temperature_2d, dewpoint_2d):
    """Surface-based CAPE/CIN/LFC/EL for multiple columns.

    Parameters
    ----------
    pressure_2d : 2D array, shape (ncols, nlevels)
    temperature_2d : 2D array, shape (ncols, nlevels)
    dewpoint_2d : 2D array, shape (ncols, nlevels)

    Returns
    -------
    cape, cin, lfc, el : 1D arrays, shape (ncols,)
    """
    ncols = pressure_2d.shape[0]
    cape = np.empty(ncols, dtype=np.float64)
    cin = np.empty(ncols, dtype=np.float64)
    lfc = np.empty(ncols, dtype=np.float64)
    el = np.empty(ncols, dtype=np.float64)
    for i in prange(ncols):
        cape[i], cin[i], lfc[i], el[i] = _cape_cin_column(
            pressure_2d[i], temperature_2d[i], dewpoint_2d[i]
        )
    return cape, cin, lfc, el


@njit(parallel=True, cache=True)
def _mu_cape_cin_batch(pressure_2d, temperature_2d, dewpoint_2d, depth_pa=30000.0):
    """Most-unstable CAPE/CIN/LFC/EL for multiple columns."""
    ncols = pressure_2d.shape[0]
    cape = np.empty(ncols, dtype=np.float64)
    cin = np.empty(ncols, dtype=np.float64)
    lfc = np.empty(ncols, dtype=np.float64)
    el = np.empty(ncols, dtype=np.float64)
    for i in prange(ncols):
        cape[i], cin[i], lfc[i], el[i] = _mu_cape_cin_column(
            pressure_2d[i], temperature_2d[i], dewpoint_2d[i], depth_pa
        )
    return cape, cin, lfc, el


@njit(parallel=True, cache=True)
def _ml_cape_cin_batch(pressure_2d, temperature_2d, dewpoint_2d, depth_pa=10000.0):
    """Mixed-layer CAPE/CIN/LFC/EL for multiple columns."""
    ncols = pressure_2d.shape[0]
    cape = np.empty(ncols, dtype=np.float64)
    cin = np.empty(ncols, dtype=np.float64)
    lfc = np.empty(ncols, dtype=np.float64)
    el = np.empty(ncols, dtype=np.float64)
    for i in prange(ncols):
        cape[i], cin[i], lfc[i], el[i] = _ml_cape_cin_column(
            pressure_2d[i], temperature_2d[i], dewpoint_2d[i], depth_pa
        )
    return cape, cin, lfc, el
