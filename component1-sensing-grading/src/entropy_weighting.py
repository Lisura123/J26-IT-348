# Component 1 - Stage 3: entropy weighting

import numpy as np
import pandas as pd


INDICATORS = {
    "power_ratio": "benefit",   
    "voltage_ratio": "benefit", 
    "current_ratio": "benefit", 
    "temp_rise": "cost",        
    "dust_level": "cost",       
}


def make_indicators(df):
    df = df.copy()
    irr = (df["irradiance_lux"] / 120).clip(lower=1)   # W/m2 (data contract: lux / 120)

    power_per_irr = df["power_mw"] / irr
    current_per_irr = df["current_ma"] / irr

    # healthy reference = 95th percentile, so a few high values do not decide it
    df["power_ratio"] = (power_per_irr / power_per_irr.quantile(0.95)).clip(0, 1)
    df["current_ratio"] = (current_per_irr / current_per_irr.quantile(0.95)).clip(0, 1)
    df["voltage_ratio"] = (df["voltage_v"] / df["voltage_v"].quantile(0.95)).clip(0, 1)

    # average of the panel temperature sensors we have (1 on our device, 2 in the dataset)
    panel_cols = [c for c in ["panel_temp1_c", "panel_temp2_c"] if c in df.columns]
    panel_temp = df[panel_cols].mean(axis=1)
    df["temp_rise"] = panel_temp - df["ambient_temp_c"]
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