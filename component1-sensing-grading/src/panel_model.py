import numpy as np
from pvlib.ivtools.sdm import fit_desoto
from pvlib.pvsystem import calcparams_desoto, i_from_v

PANEL = dict(v_mp=5.0, i_mp=0.20, v_oc=6.0, i_sc=0.22,
             alpha_sc=0.00011, beta_voc=-0.020, cells_in_series=10)

LOAD_OHMS = 120.0 + 0.1      # load resistor + INA219 shunt
LUX_PER_WM2 = 120.0          # data contract: G (W/m2) = lux / 120

# a panel in the sun gets warmer than the air (NOCT 45 C):
# about (45 - 20) / 800 = 0.03125 C for every W/m2
TEMP_RISE_PER_WM2 = (45.0 - 20.0) / 800.0

# single diode model parameters, fitted once from the datasheet values
PARAMS = fit_desoto(**PANEL)[0]


def expected_current(irradiance_lux, panel_temp_c):
    # current (mA) a HEALTHY panel gives into my 120 ohm load
    g = np.asarray(irradiance_lux, dtype=float) / LUX_PER_WM2
    t = np.nan_to_num(np.asarray(panel_temp_c, dtype=float), nan=25.0)

    il, io, rs, rsh, nnsvth = calcparams_desoto(
        g, t, PARAMS["alpha_sc"], PARAMS["a_ref"], PARAMS["I_L_ref"],
        PARAMS["I_o_ref"], PARAMS["R_sh_ref"], PARAMS["R_s"])

    # find the voltage where panel current = V / R (bisection)
    lo = np.zeros_like(g)
    hi = np.full_like(g, 8.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        diff = i_from_v(mid, il, io, rs, rsh, nnsvth) - mid / LOAD_OHMS
        lo = np.where(diff > 0, mid, lo)
        hi = np.where(diff > 0, hi, mid)

    return np.clip((lo + hi) / 2 / LOAD_OHMS, 0, None) * 1000


def expected_voltage(irradiance_lux, panel_temp_c):
    # voltage (V) on the load for a healthy panel
    return expected_current(irradiance_lux, panel_temp_c) / 1000 * LOAD_OHMS


def expected_temp_rise(irradiance_lux):
    # how much warmer than the air a healthy panel is (C)
    return np.asarray(irradiance_lux, dtype=float) / LUX_PER_WM2 * TEMP_RISE_PER_WM2