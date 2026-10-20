# Component 1 - Stage 3: entropy weighting
import numpy as np
import pandas as pd


try:
    from panel_model import expected_current, expected_temp_rise, LOAD_OHMS
except ImportError:
    from src.panel_model import expected_current, expected_temp_rise, LOAD_OHMS


INDICATORS = {
    "power_ratio": "benefit",   
    "voltage_ratio": "benefit", 
    "current_ratio": "benefit", 
    "temp_rise": "cost",        
    "dust_level": "cost",       
}


def make_indicators(df):
    # compare each reading with a HEALTHY panel at the same light and temperature
    # (physics model in panel_model.py), 1 = as good as a healthy panel
    df = df.copy()
    i_exp = np.clip(expected_current(df["irradiance_lux"], df["panel_temp1_c"]), 0.1, None)
    v_exp = i_exp / 1000 * LOAD_OHMS

    df["power_ratio"] = (df["power_mw"] / (i_exp * v_exp)).clip(0, 1)
    df["voltage_ratio"] = (df["voltage_v"] / v_exp).clip(0, 1)
    df["current_ratio"] = (df["current_ma"] / i_exp).clip(0, 1)

    # temp rise ABOVE what a healthy panel gets in this sunlight
    panel_cols = [c for c in ["panel_temp1_c", "panel_temp2_c"] if c in df.columns]
    panel_temp = df[panel_cols].mean(axis=1)
    df["temp_rise"] = panel_temp - df["ambient_temp_c"] - expected_temp_rise(df["irradiance_lux"])

    # my device gives dust in % (DSM501B), the old dataset had volts (GP2Y10)
    df["dust_level"] = df["dust_lpo_pct"] if "dust_lpo_pct" in df.columns else df["dust_vo_v"]
    return df


def normalize(df):
    norm = pd.DataFrame(index=df.index)
    for col, direction in INDICATORS.items():
        lo, hi = df[col].min(), df[col].max()
        if direction == "benefit":
            norm[col] = (df[col] - lo) / (hi - lo)
        else:
            norm[col] = (hi - df[col]) / (hi - lo)
    return norm


def entropy_weights(norm):
    p = (norm + 1e-12) / (norm + 1e-12).sum()

    
    n = len(norm)
    entropy = -(p * np.log(p)).sum() / np.log(n)

    
    info = 1 - entropy
    return info / info.sum()