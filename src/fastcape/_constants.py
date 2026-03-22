"""Physical constants in SI units, computed to match MetPy exactly."""

# Molar gas constant [J / (mol K)]
R = 8.314462618

# Molecular weights [kg/mol]
Mw = 18.015268e-3  # water
Md = 28.96546e-3    # dry air

# Gas constants [J / (kg K)]
Rd = R / Md
Rv = R / Mw

# Specific heat ratios (dimensionless)
_dry_air_gamma = 1.4
_wv_gamma = 1.330

# Specific heats [J / (kg K)]
Cp_d = _dry_air_gamma * Rd / (_dry_air_gamma - 1)
Cp_v = _wv_gamma * Rv / (_wv_gamma - 1)
Cp_l = 4219.4    # liquid water
Cp_i = 2090.0    # ice

# Latent heats [J/kg]
Lv = 2.50084e6   # vaporization
Lf = 3.337e5     # fusion
Ls = Lv + Lf     # sublimation

# Dimensionless ratios
epsilon = Mw / Md
kappa = Rd / Cp_d

# Other
g = 9.80665           # gravitational acceleration [m/s^2]
T0 = 273.16           # water triple point [K]
sat_pressure_0c = 611.2  # saturation vapor pressure at 0°C [Pa] (6.112 hPa)
P0 = 100000.0         # reference pressure [Pa] (1000 hPa)
