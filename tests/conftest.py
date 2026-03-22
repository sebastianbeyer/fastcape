"""Test fixtures: reference soundings for CAPE/CIN validation."""

import numpy as np
import pytest


@pytest.fixture
def high_cape_sounding():
    """High-CAPE severe weather sounding (MetPy example).

    Pressure in Pa (descending), temperature and dewpoint in K.
    """
    # Pressure [hPa] -> [Pa]
    p_hpa = np.array([
        1008., 1000., 950., 900., 850., 800., 750., 700., 650., 600.,
        550., 500., 450., 400., 350., 300., 250., 200.,
        175., 150., 125., 100., 80., 70., 60., 50.,
        40., 30., 25., 20.
    ])
    p = p_hpa * 100.0  # Pa

    # Temperature [degC] -> [K]
    T_degC = np.array([
        29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1,
        -0.5, -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4,
        -59.2, -66.5, -74.1, -78.5, -76.0, -71.6, -66.7, -61.3,
        -56.3, -51.7, -50.7, -47.5
    ])
    T = T_degC + 273.15

    # Relative humidity -> dewpoint
    rh = np.array([
        .85, .65, .36, .39, .82, .72, .75, .86, .65, .22, .52,
        .66, .64, .20, .05, .75, .76, .45, .25, .48, .76, .88,
        .56, .88, .39, .67, .15, .04, .94, .35
    ])
    # Compute dewpoint from RH using our sat vapor pressure
    from fastcape._thermo import _sat_vapor_pressure
    es = _sat_vapor_pressure(T)
    e = rh * es
    # Invert Ambaum formula numerically (use Bolton approximation for Td)
    # Td from vapor pressure: use iterative or Bolton approximation
    # Bolton: Td = (243.5 * ln(e/611.2)) / (17.67 - ln(e/611.2)) + 273.15
    # This is approximate but good enough for test fixture generation.
    # Actually let's use a more direct approach with MetPy for reference values.
    Td = _dewpoint_from_e(e)

    return p, T, Td


@pytest.fixture
def stable_sounding():
    """Stable sounding with zero CAPE (strong inversion)."""
    p = np.array([1000., 950., 900., 850., 800., 700., 600., 500., 400., 300., 200.]) * 100.0
    # Strong inversion: temperature increases then decreases slowly
    T = np.array([280., 285., 283., 278., 273., 263., 253., 240., 225., 210., 195.], dtype=np.float64)
    # Very dry dewpoints (large depression)
    Td = np.array([270., 265., 260., 255., 250., 240., 230., 220., 205., 190., 175.], dtype=np.float64)
    return p, T, Td


@pytest.fixture
def marginal_sounding():
    """Marginal instability sounding (small CAPE, some CIN)."""
    p = np.array([1000., 950., 900., 850., 800., 750., 700., 600., 500., 400., 300., 200.]) * 100.0
    T = np.array([293.15, 289.15, 285.15, 281.15, 277.15, 273.15, 269.15,
                  260.15, 249.15, 235.15, 218.15, 200.15], dtype=np.float64)
    Td = np.array([289.15, 286.15, 280.15, 274.15, 268.15, 263.15, 258.15,
                   248.15, 237.15, 220.15, 205.15, 185.15], dtype=np.float64)
    return p, T, Td


def _dewpoint_from_e(e):
    """Approximate dewpoint from vapor pressure using Magnus formula.

    Parameters
    ----------
    e : array
        Vapor pressure [Pa].

    Returns
    -------
    Td : array
        Dewpoint [K].
    """
    # Magnus formula constants (over liquid water)
    # e = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    # Solving for T:
    e = np.maximum(e, 1e-10)  # avoid log(0)
    ln_ratio = np.log(e / 611.2)
    Td_C = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return Td_C + 273.15
