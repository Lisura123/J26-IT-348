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
    # MQTT sometimes sends the same reading two times (when ACK is lost)
    # same record_id, different message_id
    # keep the first one and flag it
    dup_ids = df[df["record_id"].duplicated()]["record_id"].unique()

    df = df.drop_duplicates(subset="record_id", keep="first").copy()
    df = add_flag(df, df["record_id"].isin(dup_ids), "DUPLICATE_RECEIVED")
    return df


def fix_temperature(df):
    # -127 and -999 are not real temperatures
    # change them to NaN and add a flag
    for num, col in [(1, "panel_temp1_c"), (2, "panel_temp2_c")]:
        disconnected = df[col] == TEMP_DISCONNECTED
        missing = df[col] == TEMP_MISSING

        df = add_flag(df, disconnected, f"TEMP{num}_SENTINEL_-127")
        df = add_flag(df, missing, f"TEMP{num}_MISSING_-999")
        df.loc[disconnected | missing, col] = np.nan

    # we have 2 temp sensors on the panel
    # if one is bad, take the value from the other one
    # if both are bad, it stays NaN for now
    df["panel_temp1_c"] = df["panel_temp1_c"].fillna(df["panel_temp2_c"])
    df["panel_temp2_c"] = df["panel_temp2_c"].fillna(df["panel_temp1_c"])
    return df


def run_quality_checks(raw):
    # main function - runs all the checks one by one
    df = raw.copy()
    df["quality_flag"] = "OK"

    df = remove_duplicates(df)
    df = fix_temperature(df)

    return df