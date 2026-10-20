import glob
import pandas as pd

RAW_FILES = "member1_raw_sensor_stream_SITE-*.csv"
HEALTH_FILES = "member1_sensor_health_SITE-*.csv"


def load_files(pattern, data_dir):
    files = sorted(glob.glob(f"{data_dir}/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No files like {pattern} in {data_dir}")
    # ignore_index: every site file starts at row 0, so make one new index
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_raw(data_dir="../data"):
    # the raw sensor stream of all sites (input of the pipeline)
    return load_files(RAW_FILES, data_dir)


def load_health(data_dir="../data"):
    # the reference file of all sites (label is only used to CHECK the grade)
    return load_files(HEALTH_FILES, data_dir)