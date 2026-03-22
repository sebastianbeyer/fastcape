"""Tests for vectorized thermo helpers."""

import numpy as np
import pytest


class TestLCLRomps:
    def test_vs_metpy(self):
        metpy = pytest.importorskip("metpy")
        from metpy.calc import lcl
        from metpy.units import units

        from fastcape._thermo import lcl_romps

        p = np.array([100000., 95000., 90000.])
        T = np.array([300., 295., 290.])
        Td = np.array([290., 285., 280.])

        p_lcl_ours, T_lcl_ours = lcl_romps(p, T, Td)

        for i in range(len(p)):
            ref_p, ref_t = lcl(p[i] * units.Pa, T[i] * units.K, Td[i] * units.K)
            ref_p = ref_p.to('Pa').magnitude
            ref_t = ref_t.to('K').magnitude
            assert abs(p_lcl_ours[i] - ref_p) < 100., (
                f"LCL pressure mismatch at i={i}: {p_lcl_ours[i]:.1f} vs {ref_p:.1f}"
            )
            assert abs(T_lcl_ours[i] - ref_t) < 0.1, (
                f"LCL temp mismatch at i={i}: {T_lcl_ours[i]:.3f} vs {ref_t:.3f}"
            )


class TestEquivalentPotentialTemperature:
    def test_vs_metpy(self):
        metpy = pytest.importorskip("metpy")
        from metpy.calc import equivalent_potential_temperature as ept_metpy
        from metpy.units import units

        from fastcape._thermo import equivalent_potential_temperature

        p = np.array([85000.])
        T = np.array([293.15])
        Td = np.array([291.15])

        ours = equivalent_potential_temperature(p, T, Td)[0]
        ref = ept_metpy(
            p[0] * units.Pa, T[0] * units.K, Td[0] * units.K
        ).magnitude

        assert abs(ours - ref) / ref < 1e-4, f"theta_e mismatch: {ours:.2f} vs {ref:.2f}"
