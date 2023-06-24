"""
N: nbr of sensors
L: length of the sequence
F: nbr of features
"""
import sys
import time
import configparser
import os
from pathlib import Path
# from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd

from main_api import PurpleAir
from sensors import Sensor, sensors_groups

from main_purpleair_spider import fetch_data

UTCNOW = pd.Timestamp.utcnow()


def to_clean_df(
        group_name: str,
        sensor_index: int,
        avg: int,
        dfs: pd.DataFrame,
) -> pd.DataFrame:
    assert isinstance(dfs, pd.DataFrame), type(dfs)
    assert len(set(dfs.sensor_index)) == 1, set(dfs.sensor_index)
    # assert dfs.voc.isna().all(), dfs.voc
    # dfs = dfs.drop(columns=['sensor_index', 'voc'])
    dfs = dfs.drop(columns=['sensor_index'])
    assert all(t.name == 'float64' for t in dfs.dtypes), dfs.info(buf=sys.stderr)
    # print(dfs)

    # Add NaN when there are missing data:
    hour = 60
    day = 24 * hour
    year = 365 * day
    resample_rule = {
        0: '1S',
        10: '10T',
        30: '30T',
        hour: '1H',
        6 * hour: '6H',
        day: '1D',
    }
    dfs = dfs.resample(resample_rule[avg]).last()
    # print(dfs)

    return dfs


def save_csv(df: pd.DataFrame, datadir: bytes | str | os.PathLike = 'data/'):
    assert isinstance(df, pd.DataFrame), type(df)
    sensor_path = Path(datadir) / "purpleair_csv_N_LF_shape" / f"{group_name}_{avg}avg" / f"purpleair_sensor_{sensor_index}.csv"
    assert os.path.exists(datadir), datadir
    os.makedirs(sensor_path.parent, exist_ok=True)
    sensor_path = str(sensor_path)

    df.to_csv(sensor_path)
    _loaded_df = read_csv(sensor_path)
    # print(_loaded_df)
    assert type(df.index) is type(_loaded_df.index), (type(df.index), type(_loaded_df.index))
    assert df.shape == _loaded_df.shape, (df.shape, _loaded_df.shape)
    assert np.allclose(df.to_numpy(), _loaded_df.to_numpy(), equal_nan=True), (df.to_numpy(), _loaded_df.to_numpy())
    # assert np.array_equal(df.to_numpy(), _loaded_df.to_numpy(), equal_nan=True), (df.to_numpy(), _loaded_df.to_numpy())


def read_csv(sensor_path: bytes | str | os.PathLike) -> pd.DataFrame:
    #loaded_df = pd.read_csv(sensor_path, index_col='time_stamp', parse_dates=True,)
    loaded_df = pd.read_csv(sensor_path, index_col=0, parse_dates=True,)
    return loaded_df


if __name__ == "__main__":
    data_path = Path('./PurpleAirSF/')

    pd.set_option('display.max_columns', None, 'display.expand_frame_repr', False)
    start_time = time.perf_counter()
    start_time_process = time.process_time()

    config = configparser.ConfigParser()
    config.read('keys/Airly_API_key.conf')
    config.read('keys/PurpleAir_API_key.conf')

    # PurpleAir
    purple_air = PurpleAir(config)

    # check data folder exists:
    assert os.path.exists(data_path)

    print(f"sensors_groups.keys():", sensors_groups.keys())
    sensor_group_names = [
        #'santa_monica',
        #'agrinio',
        'san_francisco',
    ]
    print(f"sensor_group_names:", sensor_group_names)
    average_in_minutes_list = [24 * 60, 6 * 60, 60,] # 10,]  # frequency in minutes
    print(f"average_in_minutes_list: {average_in_minutes_list}")
    print()

    is_first = True
    for (group_name, sensor_index, avg), dfs in fetch_data(
        purple_air_api=purple_air,
        sensor_group_names=sensor_group_names,
        average_in_minutes=average_in_minutes_list,
        concat_dfs=True,
        #end_date=None,
        end_date=pd.to_datetime('2023-05-15', utc=True),  # keep the same date for having fixed calls
        time_span=pd.Timedelta(days=1),
        datadir=data_path,
        workers=250,
    ):
        if is_first:
            is_first = False
            print('Example dfs:')
            print(dfs)
            print("dfs.drop(columns=['sensor_index', 'voc']")
            preproc_df = to_clean_df(group_name, sensor_index, avg, dfs)
            print(f'Example numpy (shape={preproc_df.shape}):')
            print(preproc_df)
        pass
        preproc_df = to_clean_df(group_name, sensor_index, avg, dfs)
        save_csv(preproc_df, datadir=data_path)
    print()
    print()

