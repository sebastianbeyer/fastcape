"""Vectorized numpy/scipy thermodynamic helpers.

These operate on arrays and are used for preprocessing before calling
the Numba column kernels.
"""

import numpy as np
from scipy.special import lambertw

from . import _constants as c


def _sat_vapor_pressure(T):
    """Saturation vapor pressure over liquid water [Pa]. Vectorized."""
    L = c.Lv - (c.Cp_l - c.Cp_v) * (T - c.T0)
    heat_power = (c.Cp_l - c.Cp_v) / c.Rv
    exp_term = (c.Lv / c.T0 - L / T) / c.Rv
    return c.sat_pressure_0c * (c.T0 / T) ** heat_power * np.exp(exp_term)


def _sat_mixing_ratio(p, T):
    """Saturation mixing ratio [kg/kg]. Vectorized."""
    es = _sat_vapor_pressure(T)
    return c.epsilon * es / (p - es)


def _specific_humidity_from_dewpoint(p, Td):
    """Specific humidity from dewpoint [kg/kg]. Vectorized."""
    w = _sat_mixing_ratio(p, Td)
    return w / (1.0 + w)


def _relative_humidity_from_dewpoint(T, Td):
    """Relative humidity from temperature and dewpoint. Vectorized."""
    return _sat_vapor_pressure(Td) / _sat_vapor_pressure(T)


def dewpoint_from_specific_humidity(p, T, q):
    """Compute dewpoint temperature from specific humidity.

    Parameters
    ----------
    p : array-like
        Pressure [Pa].
    T : array-like
        Temperature [K] (unused, kept for MetPy-compatible signature).
    q : array-like
        Specific humidity [kg/kg].

    Returns
    -------
    Td : array
        Dewpoint temperature [K].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # q -> mixing ratio -> vapor pressure
    w = q / (1.0 - q)
    e = w * p / (c.epsilon + w)

    # Invert Ambaum (2020) SVP formula numerically using Newton's method.
    # Start from Magnus approximation for initial guess.
    e_safe = np.maximum(e, 1e-10)
    ln_ratio = np.log(e_safe / 611.2)
    Td = 243.5 * ln_ratio / (17.67 - ln_ratio) + 273.15

    # Two Newton iterations on f(T) = SVP(T) - e
    for _ in range(2):
        es = _sat_vapor_pressure(Td)
        # d(SVP)/dT via Clausius-Clapeyron: des/dT = L * es / (Rv * T^2)
        L = c.Lv - (c.Cp_l - c.Cp_v) * (Td - c.T0)
        des_dT = L * es / (c.Rv * Td * Td)
        Td = Td - (es - e) / des_dT

    return Td


def lcl_romps(p, T, Td):
    """LCL pressure and temperature using Romps (2017) analytical solution.

    Parameters
    ----------
    p : array-like
        Pressure [Pa].
    T : array-like
        Temperature [K].
    Td : array-like
        Dewpoint [K].

    Returns
    -------
    p_lcl, T_lcl : arrays
        LCL pressure [Pa] and temperature [K].
    """
    p = np.asarray(p, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    Td = np.asarray(Td, dtype=np.float64)

    q = _specific_humidity_from_dewpoint(p, Td)
    Rm = (1.0 - q) * c.Rd + q * c.Rv
    cpm = (1.0 - q) * c.Cp_d + q * c.Cp_v
    moist_heat_ratio = cpm / Rm

    spec_heat_diff = c.Cp_l - c.Cp_v
    a = moist_heat_ratio + spec_heat_diff / c.Rv
    b = -(c.Lv + spec_heat_diff * c.T0) / (c.Rv * T)
    cv = b / a

    RH = _relative_humidity_from_dewpoint(T, Td)
    w_arg = RH ** (1.0 / a) * cv * np.exp(cv)
    w_minus1 = np.real(lambertw(w_arg, k=-1))

    T_lcl = cv / w_minus1 * T
    p_lcl = p * (T_lcl / T) ** moist_heat_ratio
    return p_lcl, T_lcl


def equivalent_potential_temperature(p, T, Td):
    """Bolton (1980) equivalent potential temperature [K]. Vectorized.

    Parameters
    ----------
    p : array-like
        Pressure [Pa].
    T : array-like
        Temperature [K].
    Td : array-like
        Dewpoint [K].

    Returns
    -------
    theta_e : array
        Equivalent potential temperature [K].
    """
    p = np.asarray(p, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    Td = np.asarray(Td, dtype=np.float64)

    r = _sat_mixing_ratio(p, Td)
    e = _sat_vapor_pressure(Td)

    t_l = 56.0 + 1.0 / (1.0 / (Td - 56.0) + np.log(T / Td) / 800.0)
    theta = T * (c.P0 / (p - e)) ** c.kappa
    th_l = theta * (T / t_l) ** (0.28 * r)
    return th_l * np.exp(r * (1.0 + 0.448 * r) * (3036.0 / t_l - 1.78))
