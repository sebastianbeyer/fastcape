"""Tests for Numba kernels against MetPy reference values."""

import numpy as np
import pytest

from fastcape._numba_core import (
    _cape_cin_column,
    _dry_lapse,
    _lcl_bolton,
    _ml_cape_cin_column,
    _moist_lapse_rk4,
    _mu_cape_cin_column,
    _parcel_profile,
    _sat_mixing_ratio,
    _sat_vapor_pressure,
    _virtual_temperature,
)


class TestSatVaporPressure:
    def test_at_0c(self):
        """SVP at 0°C should be ~611.2 Pa."""
        es = _sat_vapor_pressure(273.15)
        assert abs(es - 611.2) < 1.0  # within 1 Pa

    def test_at_20c(self):
        """SVP at 20°C should be ~2338 Pa."""
        es = _sat_vapor_pressure(293.15)
        assert abs(es - 2338.0) < 20.0

    def test_at_100c(self):
        """SVP at 100°C should be ~101325 Pa."""
        es = _sat_vapor_pressure(373.15)
        assert abs(es - 101325.0) < 2500.0  # Ambaum formula less accurate at extremes

    def test_vs_metpy(self):
        """Compare with MetPy at several temperatures."""
        metpy = pytest.importorskip("metpy")
        from metpy.calc import saturation_vapor_pressure
        from metpy.units import units

        temps_K = [253.15, 273.15, 283.15, 293.15, 303.15, 313.15]
        for T in temps_K:
            our = _sat_vapor_pressure(T)
            ref = saturation_vapor_pressure(T * units.K).to('Pa').magnitude
            assert abs(our - ref) / ref < 1e-6, f"SVP mismatch at T={T}: {our} vs {ref}"


class TestSatMixingRatio:
    def test_vs_metpy(self):
        metpy = pytest.importorskip("metpy")
        from metpy.calc import saturation_mixing_ratio
        from metpy.units import units

        for p_hpa, T_C in [(1000, 20), (850, 10), (500, -20)]:
            p_pa = p_hpa * 100.0
            T_K = T_C + 273.15
            our = _sat_mixing_ratio(p_pa, T_K)
            ref = saturation_mixing_ratio(p_pa * units.Pa, T_K * units.K).magnitude
            assert abs(our - ref) / max(ref, 1e-10) < 1e-5


class TestDryLapse:
    def test_vs_metpy(self):
        metpy = pytest.importorskip("metpy")
        from metpy.calc import dry_lapse
        from metpy.units import units

        T0 = 300.0  # K
        p0 = 100000.0  # Pa
        for p in [95000., 85000., 70000., 50000.]:
            our = _dry_lapse(p, T0, p0)
            ref = dry_lapse(p * units.Pa, T0 * units.K, p0 * units.Pa).magnitude
            assert abs(our - ref) < 0.01, f"Dry lapse mismatch at p={p}"


class TestMoistLapse:
    def test_vs_metpy(self):
        """Moist lapse rate should agree with MetPy within 0.1 K."""
        metpy = pytest.importorskip("metpy")
        from metpy.calc import moist_lapse
        from metpy.units import units

        pressures_pa = np.array([85000., 70000., 50000., 30000., 20000.])
        T_start = 278.15  # 5°C at 925 hPa
        p_start = 92500.0

        our = _moist_lapse_rk4(pressures_pa, T_start, p_start)
        ref = moist_lapse(
            pressures_pa * units.Pa, T_start * units.K, p_start * units.Pa
        ).magnitude

        for i in range(len(pressures_pa)):
            assert abs(our[i] - ref[i]) < 0.1, (
                f"Moist lapse mismatch at p={pressures_pa[i]}: "
                f"{our[i]:.4f} vs {ref[i]:.4f}"
            )


class TestLCLBolton:
    def test_reasonable(self):
        """LCL should be between surface and 500 hPa for typical conditions."""
        p_lcl, T_lcl = _lcl_bolton(100000., 300., 290.)
        assert 50000. < p_lcl < 100000.
        assert 250. < T_lcl < 300.

    def test_saturated(self):
        """When T == Td, LCL should be at surface."""
        p_lcl, T_lcl = _lcl_bolton(100000., 290., 290.)
        assert abs(p_lcl - 100000.) < 100.  # within 1 hPa


class TestVirtualTemperature:
    def test_vs_metpy(self):
        metpy = pytest.importorskip("metpy")
        from metpy.calc import virtual_temperature
        from metpy.units import units

        T = 283.0  # K
        w = 0.012  # kg/kg
        our = _virtual_temperature(T, w)
        ref = virtual_temperature(T * units.K, w * units('kg/kg')).magnitude
        assert abs(our - ref) < 0.01


class TestParcelProfile:
    def test_starts_at_surface(self):
        """Parcel profile should start at surface temperature."""
        p = np.array([100000., 95000., 90000., 85000., 80000., 70000., 50000.])
        T_sfc = 300.0
        Td_sfc = 290.0
        prof, p_lcl, T_lcl = _parcel_profile(p, T_sfc, Td_sfc)
        assert abs(prof[0] - T_sfc) < 0.01

    def test_decreases_with_altitude(self):
        """Temperature should generally decrease going up."""
        p = np.array([100000., 95000., 90000., 85000., 80000., 70000., 50000.])
        prof, _, _ = _parcel_profile(p, 300., 290.)
        for i in range(1, len(prof)):
            assert prof[i] < prof[i - 1]


class TestCAPECIN:
    def test_high_cape(self, high_cape_sounding):
        """High-CAPE sounding should produce substantial CAPE."""
        p, T, Td = high_cape_sounding
        cape, cin, lfc, el = _cape_cin_column(p, T, Td)
        assert cape > 1000.0, f"Expected high CAPE, got {cape}"

    def test_stable(self, stable_sounding):
        """Stable sounding should produce zero or near-zero CAPE."""
        p, T, Td = stable_sounding
        cape, cin, lfc, el = _cape_cin_column(p, T, Td)
        assert cape < 50.0, f"Expected near-zero CAPE, got {cape}"

    def test_cin_nonpositive(self, high_cape_sounding):
        """CIN should always be <= 0."""
        p, T, Td = high_cape_sounding
        _, cin, _, _ = _cape_cin_column(p, T, Td)
        assert cin <= 0.0

    def test_vs_metpy(self, high_cape_sounding):
        """Compare CAPE against MetPy for high-CAPE sounding."""
        metpy = pytest.importorskip("metpy")
        from metpy.calc import cape_cin, dewpoint_from_relative_humidity, parcel_profile
        from metpy.units import units

        # Use MetPy's own example sounding for a clean comparison
        p_hpa = np.array([
            1008., 1000., 950., 900., 850., 800., 750., 700., 650., 600.,
            550., 500., 450., 400., 350., 300., 250., 200.,
            175., 150., 125., 100., 80., 70., 60., 50.,
            40., 30., 25., 20.
        ])
        T_degC = np.array([
            29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1,
            -0.5, -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4,
            -59.2, -66.5, -74.1, -78.5, -76.0, -71.6, -66.7, -61.3,
            -56.3, -51.7, -50.7, -47.5
        ])
        rh = np.array([
            .85, .65, .36, .39, .82, .72, .75, .86, .65, .22, .52,
            .66, .64, .20, .05, .75, .76, .45, .25, .48, .76, .88,
            .56, .88, .39, .67, .15, .04, .94, .35
        ])

        p_u = p_hpa * units.hPa
        T_u = T_degC * units.degC
        Td_u = dewpoint_from_relative_humidity(T_u, rh * units.dimensionless)

        prof = parcel_profile(p_u, T_u[0], Td_u[0]).to('K')
        cape_ref, cin_ref = cape_cin(p_u, T_u, Td_u, prof)
        cape_ref = cape_ref.magnitude
        cin_ref = cin_ref.magnitude

        # Our calculation
        p_pa = p_hpa * 100.0
        T_K = T_degC + 273.15
        Td_K = Td_u.to('K').magnitude

        cape_ours, cin_ours, _, _ = _cape_cin_column(p_pa, T_K, Td_K)

        # Tolerance: 200 J/kg or 5% — Bolton LCL vs Romps LCL causes systematic offset
        cape_tol = max(200.0, 0.05 * cape_ref)
        assert abs(cape_ours - cape_ref) < cape_tol, (
            f"CAPE mismatch: {cape_ours:.1f} vs {cape_ref:.1f}"
        )

    def test_empty(self):
        """Empty/short arrays should return zeros."""
        p = np.array([100000.])
        T = np.array([300.])
        Td = np.array([290.])
        cape, cin, lfc, el = _cape_cin_column(p, T, Td)
        assert cape == 0.0 and cin == 0.0


class TestMUCAPE:
    def test_high_cape(self, high_cape_sounding):
        """MU-CAPE should be >= SB-CAPE."""
        p, T, Td = high_cape_sounding
        sb_cape, _, _, _ = _cape_cin_column(p, T, Td)
        mu_cape, _, _, _ = _mu_cape_cin_column(p, T, Td)
        # MU-CAPE should be at least as large as SB-CAPE
        assert mu_cape >= sb_cape - 10.0  # small tolerance


class TestMLCAPE:
    def test_runs(self, high_cape_sounding):
        """ML-CAPE should compute without error."""
        p, T, Td = high_cape_sounding
        cape, cin, lfc, el = _ml_cape_cin_column(p, T, Td)
        assert cape >= 0.0
        assert cin <= 0.0
