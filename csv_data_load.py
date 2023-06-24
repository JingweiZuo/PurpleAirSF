"""
N: nbr of sensors
L: length of the sequence
F: nbr of features
"""
from pathlib import Path
import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from main_purpleair_to_csv import read_csv


# DATA_FOLDER = Path('data/purpleair_csv_N_LF_shape/san_francisco_60avg'); FREQ = '6h'
# DATA_FOLDER = Path('data/purpleair_csv_N_LF_shape/san_francisco_360avg'); FREQ = '6h'
# DATA_FOLDER = Path('data/purpleair_csv_N_LF_shape/san_francisco_1440avg'); FREQ = '1d'
DATA_FOLDER = Path('data/purpleair_csv_N_LF_shape/san_francisco_10avg'); FREQ = '10m'


def preproc_df_func(df):
    df = df.drop(columns='voc')
    return df


def get_preproc_df_func_range(start, end):
    def preproc_df_func(df):
        df = df.drop(columns='voc')
        df.sort_index(inplace=True)
        df = df.loc[start:end]
        return df
    return preproc_df_func


def filter_df_func(df):
    return True


def get_filter_df_func_total_nan_rate(threshold: float):
    """Note:
        tot nan rate 0.0, there are no nans,
        tot nan rate 1.0, all are nans.

    Args:
        threshold: filter out df with total_nan_rate higher than this number
    """
    def filter_df_func(df):
        return df.isna().sum().sum() / (df.shape[0] * df.shape[1]) <= threshold
    return filter_df_func


def load_all(
        preproc_df_func,
        filter_df_func,
) -> tuple[list[pd.DataFrame], list[int]]:
    """
        Args:
            preproc_df_func: to use on the loaded df, returns a new df.
            filter_df_func: if returns False skip the df, if True keep it
                    (this is done on the preproc df).
        Returns:
            sensors list, sensor_indexes
    """
    min_date = None
    max_date = None

    sensors = []
    sensor_indexes = []
    print('Loading CSVs...')
    for filename in tqdm(os.listdir(DATA_FOLDER)):
        filename_match = re.match(r'purpleair_sensor_(?P<id>[0-9]+).csv', filename)
        assert filename_match, filename
        sensor_idx = int(filename_match['id'])

        filepath = DATA_FOLDER / filename
        # print(filepath)
        sensor_df = read_csv(filepath)

        # preprocess df:
        sensor_df = preproc_df_func(sensor_df)
        sensor_df.sort_index(inplace=True)
        if min_date is None:
            min_date = sensor_df.index.min()
        else:
            min_date = min(min_date, sensor_df.index.min())
        if max_date is None:
            max_date = sensor_df.index.max()
        else:
            max_date = min(max_date, sensor_df.index.max())

        # filter sensors:
        if filter_df_func(sensor_df):
            sensor_indexes.append(sensor_idx)
            sensors.append(sensor_df)

    freq = None
    _sensors = []
    if sensors:
        df = sensors[0]
        assert isinstance(df.index, pd.DatetimeIndex), type(df.index)
        freq = df.index[1] - df.index[0]
        assert ((df.index[1:] - df.index[:-1]) == freq).all(None)
    for df in sensors:
        assert ((df.index[1:] - df.index[:-1]) == freq).all(None)
        ix = pd.date_range(start=min_date, end=max_date, inclusive='both', freq=freq)
        df = df.reindex(ix, copy=False)
        _sensors.append(df)
    sensors = _sensors

    # filter again after the extension of the index range:
    final_sensors = []
    final_sensor_indexes = []
    for sensor_df, sensor_idx in zip(sensors, sensor_indexes):
        if filter_df_func(sensor_df):
            final_sensors.append(sensor_df)
            final_sensor_indexes.append(sensor_idx)
    sensors, sensor_indexes = final_sensors, final_sensor_indexes

    return sensors, sensor_indexes


def nan_stats_plots(sensors, datadir='data', filename_tag='', save=False, title='NAN rate'):
    filename_tag += '_'
    all_nans = []
    for df in sensors:
        nans = df.isna().sum(axis=1) / df.shape[1]
        all_nans.append(nans)
    all_nans = pd.concat(all_nans, axis=1)
    all_nans.fillna(1., inplace=True)
    nan_stats = all_nans.T.describe().T
    count_non_nan_stats = nan_stats['count']
    nan_stats.drop('count', axis=1, inplace=True)
    nan_stats.plot()
    plt.title(title)
    if save:
        plt.savefig(str(Path(datadir) / f'{filename_tag}nans-stats.png'))
    plt.show()
    # count_non_nan_stats.plot()
    # (all_nans.shape[1] - (all_nans == 1.).sum(axis=1)).plot()  # count
    (1. - ((all_nans == 1.).sum(axis=1) / all_nans.shape[1])).plot()  # rate
    plt.title(title)
    if save:
        plt.savefig(str(Path(datadir) / f'{filename_tag}count_non_nan.png'))
    plt.show()
    print('Done.')


def plot_total_nan_rate(sensors,
                        start, end, total_nan_rate_threshold, freq,
                        datadir='data',
                        save=False):
    plt.scatter(
        range(len(sensors)),
        [df.isna().sum().sum() / (df.shape[0] * df.shape[1]) for df in sensors]
    )
    plt.title(f'Total nan rate ({start} to {end}; filter thr {total_nan_rate_threshold}; freq {freq})')
    if save:
        plt.savefig(str(Path(datadir) / f'total_nan_rate_{start}_{end}_{total_nan_rate_threshold}__{freq}.png'))
    plt.show()
    print('Done.')


def sensors_to_array_NLF(sensors: list[pd.DataFrame]) -> np.ndarray:
    """
    N: nbr of sensors
    L: length of the sequence
    F: nbr of features
    """
    assert sensors
    assert all(len(sensors[0]) == len(df.index) for df in sensors)
    assert all((sensors[0].index == df.index).all() for df in sensors)

    array = np.stack([df.to_numpy() for df in sensors], axis=0)
    assert array.ndim == 3, array.ndim
    assert array.shape[0] == len(sensors)
    assert array.shape[1] == sensors[0].shape[0]
    assert array.shape[2] == sensors[0].shape[1]

    return array


def save_array(sensors_array: np.ndarray, datadir='data', filename='array.npy'):
    assert isinstance(sensors_array, np.ndarray), type(sensors_array)
    filepath = str(Path(datadir, filename))

    np.save(filepath, sensors_array)  # , fmt='%.18e')
    _loaded_array = load_array(datadir=datadir, filename=filename)
    assert sensors_array.dtype is _loaded_array.dtype, (sensors_array.dtype, _loaded_array.dtype)
    assert np.allclose(sensors_array, _loaded_array, equal_nan=True), (sensors_array, _loaded_array)
    assert np.array_equal(sensors_array, _loaded_array, equal_nan=True), (sensors_array, _loaded_array)


def load_array(datadir='data', filename='array.npy'):
    filepath = str(Path(datadir, filename))
    return np.load(filepath)


def missing_val_process(arr):
    # Count the number of inherent 0 values
    arr_no_nan = np.nan_to_num(arr.astype(float), nan=0.0)
    nbr_zeros = np.count_nonzero(arr_no_nan == 0)
    print("Number of zeros is ", nbr_zeros)
    print("Number of total records is ", arr_no_nan.size)
    print("Inherent Zero rate is ", nbr_zeros/arr_no_nan.size)
    print(np.count_nonzero(arr_no_nan == 0) /arr_no_nan.size )

    return arr_no_nan


if __name__ == "__main__":
    pd.set_option('display.max_columns', None, 'display.expand_frame_repr', False)

    SAVE_ARRAYS = True

    start, end = '2021-10-01', '2023-05-15'
    total_nan_rate_threshold = .02
    datadir = 'data'
    array_filename = f"array_{FREQ}_{start}_{end}_{total_nan_rate_threshold}_NLF_shape.npy"

    sensors, sensor_indexes = load_all(get_preproc_df_func_range(start, end),
                                       get_filter_df_func_total_nan_rate(total_nan_rate_threshold))
    if SAVE_ARRAYS:
        with open(Path(datadir, Path(array_filename).stem + '_sensor_indexes.json'), 'w') as fp:
            json.dump(sensor_indexes, fp)
    print(f"Final sensors number", len(sensors))
    nan_stats_plots(sensors, datadir='data', filename_tag=f"{start}_{end}_{total_nan_rate_threshold}_{FREQ}",
                    save=False,
                    title=f'NAN rate (freq {FREQ})')
    plot_total_nan_rate(sensors, start, end, total_nan_rate_threshold, FREQ,
                        datadir='data',
                        save=False)

    array_NLF = sensors_to_array_NLF(sensors)
    array_NLF_no_nan = missing_val_process(array_NLF)
    
    if SAVE_ARRAYS:
        save_array(array_NLF_no_nan,
                   datadir=datadir,
                   filename=array_filename)


