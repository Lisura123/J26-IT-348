import numpy as np
import pandas as pd

from .config import DATA, LUX_PER_WM2, LUX_SATURATION, PROBE_INVALID

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _seconds(ts: pd.Series) -> np.ndarray:
    return (ts - EPOCH).dt.total_seconds().to_numpy()


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(DATA / "weather_hourly.csv")
    w["timestamp_utc"] = pd.to_datetime(w["timestamp_utc"], utc=True)
    return w.sort_values(["site_id", "timestamp_utc"])


def interpolate_weather(df, weather, cols=("wind_speed_10m_ms", "ghi_estimate_wm2")):
    
    out = pd.DataFrame(index=df.index, columns=list(cols), dtype=float)
    for site, rows in df.groupby("site_id"):
        w = weather[weather.site_id == site]
        x = _seconds(w["timestamp_utc"])        # hourly weather times
        xi = _seconds(rows["timestamp_utc"])    # 5-min reading times
        for c in cols:
            out.loc[rows.index, c] = np.interp(xi, x, w[c].to_numpy())
    return out


def estimate_irradiance(lux: pd.Series, ghi_api: pd.Series):
    """G [W/m2] = lux / 120; if the lux sensor saturates, use weather-API GHI instead.
    Panel tilt is only 8 deg, so GHI is a fair stand-in for plane-of-array irradiance."""
    saturated = lux >= LUX_SATURATION
    g = np.where(saturated,
                 np.maximum(ghi_api, LUX_SATURATION / LUX_PER_WM2),  # at least 833 W/m2
                 lux / LUX_PER_WM2)
    src = np.where(saturated, "WEATHER_API_GHI (lux saturated)", f"LUX_SENSOR / {LUX_PER_WM2:g}")
    return pd.Series(g, index=lux.index), pd.Series(src, index=lux.index)


def clean_probes(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of the two panel probes; DS18B20 error codes become NaN."""
    t1 = df["panel_temp1_c"].where(~df["panel_temp1_c"].isin(PROBE_INVALID))
    t2 = df["panel_temp2_c"].where(~df["panel_temp2_c"].isin(PROBE_INVALID))
    return pd.DataFrame({
        "panel_temp_mean_c": pd.concat([t1, t2], axis=1).mean(axis=1),
        "panel_temp_spread_c": (t1 - t2).abs(),   # large spread = hotspot (shading)
        "probe_fault": t1.isna() | t2.isna(),
    }, index=df.index)


def prepare(df: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add irradiance, wind and probe columns to raw sensor readings."""
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    weather = load_weather() if weather is None else weather
    wx = interpolate_weather(df, weather)
    df["irradiance_wm2_est"], df["irradiance_source"] = estimate_irradiance(
        df["irradiance_lux"], wx["ghi_estimate_wm2"])
    df["wind_speed_ms_used"] = wx["wind_speed_10m_ms"].astype(float)
    df["wind_source"] = "WEATHER_API (hourly, interpolated)"
    return df.join(clean_probes(df))