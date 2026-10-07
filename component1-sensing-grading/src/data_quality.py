# Stage 1: data quality checks
import numpy as np
import pandas as pd

# DS18B20 sends these values when something is wrong
TEMP_DISCONNECTED = -127
TEMP_MISSING = -999


def add_flag(df, rows, flag_name):
    # add a flag to the selected rows
    # if a row already has a flag, join them with "|"
    for i in df[rows].index:
        if df.at[i, "quality_flag"] == "OK":
            df.at[i, "quality_flag"] = flag_name
        else:
            df.at[i, "quality_flag"] += "|" + flag_name
    return df


def remove_duplicates(df):
    dup_ids = df[df["record_id"].duplicated()]["record_id"].unique()

    df = df.drop_duplicates(subset="record_id", keep="first").copy()
    df = add_flag(df, df["record_id"].isin(dup_ids), "DUPLICATE_RECEIVED")
    return df


def fix_temperature(df):
    for num, col in [(1, "panel_temp1_c"), (2, "panel_temp2_c")]:
        disconnected = df[col] == TEMP_DISCONNECTED
        missing = df[col] == TEMP_MISSING

        df = add_flag(df, disconnected, f"TEMP{num}_SENTINEL_-127")
        df = add_flag(df, missing, f"TEMP{num}_MISSING_-999")
        df.loc[disconnected | missing, col] = np.nan

    # Use the other temperature sensor if one is faulty; if both are faulty, keep the value as NaN
    df["panel_temp1_c"] = df["panel_temp1_c"].fillna(df["panel_temp2_c"])
    df["panel_temp2_c"] = df["panel_temp2_c"].fillna(df["panel_temp1_c"])
    return df

def add_session_id(df):
    # A new session starts whenever the gap between two readings exceeds 60 seconds.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values(["device_id", "timestamp_utc"]).copy()

    gap = df.groupby("device_id")["timestamp_utc"].diff().dt.total_seconds()
    new_session = gap.isna() | (gap > 60)
    df["session_id"] = new_session.cumsum()
    return df


def fix_current(df):
    # Use the session median to detect reversed current: flip reversed sessions, but set small negative noise to 0.
    session_median = df.groupby("session_id")["current_ma"].transform("median")
    reversed_wire = session_median < 0

    df = add_flag(df, reversed_wire, "CURRENT_SIGN_REVERSED")
    df.loc[reversed_wire, "current_ma"] = -df.loc[reversed_wire, "current_ma"]

    # after flipping, any small negative left is just noise
    df.loc[df["current_ma"] < 0, "current_ma"] = 0.0
    return df

def fix_spikes(df):
    checks = [
        ("voltage_v", "SPIKE_VOLTAGE", 0.1),
        ("current_ma", "SPIKE_CURRENT", 3.0),
        ("irradiance_lux", "SPIKE_IRRADIANCE", 1000.0),
    ]

    for col, flag_name, min_jump in checks:
        neighbour_median = df.groupby("session_id")[col].transform(
            lambda s: s.rolling(5, center=True, min_periods=1).median()
        )
        jump = (df[col] - neighbour_median).abs()

        spike = (jump > 0.5 * neighbour_median.abs()) & (jump > min_jump)

        df = add_flag(df, spike, flag_name)
        df.loc[spike, col] = neighbour_median[spike]
    return df


def fix_humidity(df):
    df["humidity_pct"] = df["humidity_pct"].clip(upper=100)
    return df


def run_quality_checks(raw):
    # main function - runs all the checks one by one
    df = raw.copy()
    df["quality_flag"] = "OK"

    df = remove_duplicates(df)
    df = fix_temperature(df)
    df = add_session_id(df)
    df = fix_current(df)
    df = fix_spikes(df)
    df = fix_humidity(df)

    return df