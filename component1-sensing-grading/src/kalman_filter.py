# Stage 2: Kalman filter

import numpy as np


def kalman_1d(readings, q, r):

    estimates = []
    x = None   
    p = 1.0    

    for z in readings:
        if x is None:
            if np.isnan(z):
                estimates.append(np.nan)
                continue
            x = z
            p = r
            estimates.append(x)
            continue

        
        p = p + q

        
        if not np.isnan(z):
            k = p / (p + r)        
            x = x + k * (z - x)    
            p = (1 - k) * p        

        estimates.append(x)

    return np.array(estimates)


def apply_kalman(df, col, q, r):
    out = df[col].copy()
    for session, rows in df.groupby("session_id"):
        out.loc[rows.index] = kalman_1d(rows[col].to_numpy(dtype=float), q, r)
    return out


def estimate_noise(df, col):
    jumps = df.groupby("session_id")[col].diff()
    noise_std = jumps.std() / np.sqrt(2)
    return noise_std ** 2   # R is a variance, so square it



KALMAN_SETTINGS = {
    "voltage_v": 0.3,
    "current_ma": 0.3,
    "panel_temp1_c": 0.1,
    "panel_temp2_c": 0.1,
    "ambient_temp_c": 0.1,
    "humidity_pct": 0.1,
    "pressure_hpa": 0.1,
}


def run_kalman_stage(df):
    # filter every sensor in KALMAN_SETTINGS and return a new copy
    df = df.copy()
    for col, ratio in KALMAN_SETTINGS.items():
        r = estimate_noise(df, col)
        df[col] = apply_kalman(df, col, ratio * r, r)

    # power = voltage x current, so we calculate it again
    # from the filtered values
    df["power_mw"] = df["voltage_v"] * df["current_ma"]
    return df