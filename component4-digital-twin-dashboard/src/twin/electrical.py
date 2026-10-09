
import numpy as np
import pandas as pd
import pvlib


def expected_operating_point(g_wm2, t_cell_c, ds: dict) -> pd.DataFrame:
    g = np.maximum(np.asarray(g_wm2, float), 0.5)  
    t = np.asarray(t_cell_c, float)


    il, i0, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_desoto(
        g, t,
        alpha_sc=ds["alpha_isc_a_per_c"],
        a_ref=ds["desoto_a_ref"],
        I_L_ref=ds["desoto_I_L_ref"],
        I_o_ref=ds["desoto_I_o_ref"],
        R_sh_ref=ds["desoto_R_sh_ref"],
        R_s=ds["desoto_R_s"])

    sd = pvlib.pvsystem.singlediode(il, i0, rs, rsh, nnsvth)


    return pd.DataFrame({
        "expected_power_mw": sd["p_mp"] * 1000,
        "expected_voltage_v": sd["v_mp"],
        "expected_current_ma": sd["i_mp"] * 1000,
        "expected_voc_v": sd["v_oc"],
        "expected_isc_ma": sd["i_sc"] * 1000,
    })