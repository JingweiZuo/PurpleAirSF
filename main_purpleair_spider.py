import sys
import time

import json
import configparser
from collections import defaultdict
from collections.abc import Generator
import pickle
import os
from pathlib import Path
import math
import warnings
import multiprocessing as mp
# from tqdm.contrib.concurrent import process_map

from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from main_api import PurpleAir
from sensors import Sensor, sensors_groups
from connection_utils import try_again_if_connection_fails_decorator

UTCNOW = pd.Timestamp.utcnow()
LOCK = mp.Lock()  # lock must be global (but it works only with forked processes then I guess, so only on linux?)


class kwargs_wrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, kwargs):
        return self.func(**kwargs)


def get_start_end_timestamps(
        purple_air_api: PurpleAir,
        average_in_minutes=60,
        end_date: None | pd.Timestamp = None,
        time_span: pd.Timedelta = pd.Timedelta(days=5 * 365),
        verbose: bool = False,
) -> tuple[int, int, int, int, pd.Timestamp]:
    """

    :param purple_air_api:
    :param average_in_minutes: frequency of the returned data
    :param end_date: if None, UTCNOW - pd.Timedelta(hours=1)
    :param time_span:
    :param tqdm_pos: index position for tqdm progress bar when multiple process are running,
        None if only one process is running.
    :param tqdm_label: desc to add to tqdm when multiple process are running
    :param verbose:
    :return: 4-tuple: start_timestamp, end_timestamp, n_spans (how many df to fetch),
                      max_span_sec (max range of one df fetched).
    """
    if verbose:
        print(f"time_span:", time_span)
        print(f"average_in_minutes:", average_in_minutes)
    if end_date is None:
        end_date = UTCNOW - pd.Timedelta(hours=1)
    try:
        _ = UTCNOW - end_date
    except TypeError as err:
        assert str(err) == 'Cannot subtract tz-naive and tz-aware datetime-like objects.', err
        raise TypeError("'end_date' should be tz-aware")
    if end_date > UTCNOW:
        raise ValueError("'end_date' in the future")

    start_timestamp = int((end_date - time_span).timestamp())
    end_timestamp = int(end_date.timestamp())
    max_span_sec = purple_air_api.csv_limits(average_in_minutes) * 60
    n_spans = math.ceil(time_span.total_seconds() / max_span_sec)
    if verbose:
        print(f"start_timestamp: {pd.to_datetime(start_timestamp, unit='s', utc=True)} ({start_timestamp})",
              f"end_timestamp:   {pd.to_datetime(end_timestamp, unit='s', utc=True)} ({end_timestamp})",
              sep='\n')
        print(f'max_span_sec (for average={average_in_minutes}): {max_span_sec}')
        print(f'n_spans: {n_spans}')

    return start_timestamp, end_timestamp, n_spans, max_span_sec, end_date


def fetch_sensor(
        purple_air_api: PurpleAir,
        sensor: Sensor,
        concat_dfs: bool = True,
        average_in_minutes=60,  # frequency in minutes
        end_date: None | pd.Timestamp = None,
        time_span: pd.Timedelta = pd.Timedelta(days=5 * 365),
        tqdm_pos: None | int = None,
        tqdm_label: str = '',
        tqdm_leave: bool = False,
        datadir: bytes | str | os.PathLike = 'data/',
) -> pd.DataFrame | list[pd.DataFrame]:
    """Fetch a single sensor."""

    start_timestamp, end_timestamp, n_spans, max_span_sec, end_date = get_start_end_timestamps(
        purple_air_api, average_in_minutes, end_date, time_span, verbose=False)

    # print(f"Fetching sensor {sensor}...")
    if sensor.key is not None:
        raise NotImplementedError(sensor)

    # init dfs:
    dfs = []

    # LOCK:
    # It needs the lock because the folder and metadata file are shared for different avgs,
    # with no lock, you get errors when two processes try to create a sensor and one starts
    # to create the folder or writing the metadata file, some conflicts raises (e.g. folder
    # already exist error, because the folder was crated by another, while the other passed
    # the condition the folder was not existing, but now actually exists; another problem
    # could be the creation of the metadata file which is shared, so two process try to
    # create the same file, this is a minor error assuming the pivot is the same, but
    # asserts could fail because one of the two process overwrites the newly created file
    # the other process just created.

    # init paths and metadata:
    sensor_path = Path(datadir) / f"purpleair_sensor_{sensor.index}"
    metadata_file = sensor_path / f"metadata.json"
    LOCK.acquire()
    if metadata_file.exists():
        LOCK.release()
        with open(metadata_file, 'r') as fp:
            try:
                metadata = json.load(fp)
            except json.decoder.JSONDecodeError as err:
                raise RuntimeError(metadata_file)
        if 'pivot' not in metadata:
            raise RuntimeError(f"No pivot found in the metadata file {str(metadata_file)!r}")
        if metadata['pivot'] != end_date.isoformat():
            pivot_timestamp = int(pd.to_datetime(metadata['pivot'], utc=True).timestamp())
            end_pivot_diff = (end_timestamp - pivot_timestamp) % max_span_sec
            if end_pivot_diff == 0:
                srt_ts = end_timestamp  # `end_timestamp - pivot_timestamp` is a multiple of `max_span_sec`
            else:
                adj = max_span_sec - end_pivot_diff
                srt_ts = end_timestamp + adj
                assert end_timestamp <= int(UTCNOW.timestamp())

                if srt_ts > int((UTCNOW - pd.Timedelta(minutes=15)).timestamp()):
                    end_ts = srt_ts
                    srt_ts = end_ts - max_span_sec
                    df = try_again_if_connection_fails_decorator(purple_air_api.csv)(
                        sensor.index,
                        start_timestamp=srt_ts,
                        end_timestamp=end_ts,
                        average=average_in_minutes)
                    # don't save this file
                    dfs.append(df)

                if srt_ts > int((UTCNOW - pd.Timedelta(minutes=15)).timestamp()):
                    raise AssertionError("'end_date' should be older")
        else:
            srt_ts = end_timestamp
    else:
        assert not sensor_path.exists(), (sensor_path, metadata_file, sensor_path.exists(), metadata_file.exists(),
                                          'metadata and its parent dir should be created together, '
                                          'cannot be possible the existence of one without the other.')
        os.mkdir(sensor_path)
        # os.makedirs(sensor_path, exist_ok=False)

        with open(metadata_file, 'w') as fp:
            metadata = {
                'pivot': end_date.isoformat()
            }
            json.dump(metadata, fp)
        srt_ts = end_timestamp

        LOCK.release()

    # loop and fetch time spans:
    # rng = trange(n_spans,
    #                 desc=f"#pid {os.getpid()}, " + tqdm_label + f"sensor_{sensor.index}",
    #                 leave=tqdm_leave,
    #                 position=tqdm_pos)
    rng = range(n_spans)
    for i in rng:
        end_ts = srt_ts
        srt_ts = end_ts - max_span_sec
        # print('srt_ts', srt_ts, pd.to_datetime(srt_ts, unit='s', utc=True), 'end_ts:', end_ts, pd.to_datetime(end_ts, unit='s', utc=True))

        current_file = sensor_path / f"{srt_ts}.{end_ts}.{average_in_minutes}.pickle"
        if not current_file.exists():
            """
            # timestamps are in UTC
            s=0
            s=3600*.5
            s=3600
            _utcnow = pd.Timestamp.utcnow()
            purple_air_api.csv(sensor.index,
                                start_timestamp=int(_utcnow.timestamp())-3600+s,
                                end_timestamp=int(_utcnow.timestamp())+s,
                                average=average_in_minutes)
            """
            df = try_again_if_connection_fails_decorator(purple_air_api.csv)(
                sensor.index,
                start_timestamp=srt_ts,
                end_timestamp=end_ts,
                average=average_in_minutes)
            # print(df)
            with open(current_file, "wb") as fp:
                pickle.dump(df, fp)
        else:
            # print("Using cache")
            with open(current_file, "rb") as fp:
                df = pickle.load(fp)
        dfs.append(df)

    # patch for older data stored with older code that were saved in tz-naive format:
    for df in dfs:
        assert isinstance(df.index, pd.DatetimeIndex), type(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

    # concat:
    if concat_dfs:
        # concat:
        dfs = pd.concat(dfs, axis=0, join='outer')
        # sort:
        dfs.sort_index(inplace=True)
        # # handle nan and nan in duplicates:
        # assert dfs.isna().all(axis=0).voc
        # dfs.drop(columns='voc', inplace=True)
        # drop duplicated index:
        if not np.array_equal(dfs.duplicated(keep=False), dfs.index.duplicated(keep=False)):
            _non_matching_duplicates = dfs.loc[dfs.duplicated(keep=False) != dfs.index.duplicated(keep=False)]
            _warn_msg = (
                f"duplicated record with different values (sensor.index={sensor.index}):\n"
                f"{_non_matching_duplicates}\nkeeping first.")
            print(f"WARNING (sensor.index={sensor.index}):", _warn_msg, sep='\n')
            warnings.warn(_warn_msg)
        dfs = dfs[~dfs.duplicated(keep='first')]

        dfs = dfs.loc[pd.to_datetime(start_timestamp, unit='s', utc=True):pd.to_datetime(end_timestamp, unit='s', utc=True)]

    sys.stdout.flush()
    sys.stderr.flush()
    return dfs


def fetch_data(
        purple_air_api: PurpleAir,
        sensor_group_names: list[str],
        average_in_minutes: int | list[int] = 60,  # frequency in minutes
        concat_dfs: bool = True,
        end_date: None | pd.Timestamp = None,
        time_span: pd.Timedelta = pd.Timedelta(days=5 * 365),
        datadir: bytes | str | os.PathLike = 'data/',
        workers: int = 100,
) -> Generator[tuple[tuple[str, int, int], pd.DataFrame | list[pd.DataFrame]], None, None]:
    """

    Args:
        purple_air_api:
        sensor_group_names: list of sensors groups names
        average_in_minutes: frequency of the returned data
        concat_dfs: given a sensor, get a combined frame or get the list of raw csv calls
        end_date: if None, UTCNOW - pd.Timedelta(hours=1)
        time_span:
        tqdm_pos: index position for tqdm progress bar when multiple process are running,
            None if only one process is running.
        tqdm_label: desc to add to tqdm when multiple process are running
        datadir: path to directory where are saved the sensors data

    Yield:
        (('group_name', sensor_index, average_in_minutes), dfs)
    """
    print(f"cpu_count: {mp.cpu_count()}")
    print(f"Main process: {os.getppid()}")
    print(f"Fetching groups {sensor_group_names} for average_in_minutes {average_in_minutes}...")

    for avg in average_in_minutes:
        _ = get_start_end_timestamps(purple_air_api, avg, end_date, time_span, verbose=True)

    if isinstance(average_in_minutes, int):
        average_in_minutes = [average_in_minutes]

    headers = []
    kwargs_list = []
    i = 0
    for group_name in sensor_group_names:
        group = sensors_groups[group_name]
        for sensor in group:
            for avg in average_in_minutes:
                headers.append((group_name, sensor.index, avg))
                kwargs_list.append(dict(
                    purple_air_api=purple_air_api,
                    sensor=sensor,
                    concat_dfs=concat_dfs,
                    average_in_minutes=avg,
                    end_date=end_date,
                    time_span=time_span,
                    tqdm_pos=i + 1,
                    tqdm_label=f"group_name {group_name!r}, avg_min {average_in_minutes}: ",
                    tqdm_leave=True,
                    datadir=datadir,
                ))
                i += 1
    # assert not Counter([s for _, s, _ in headers]), Counter([s for _, s, _ in headers])  # there are more avgs

    # # data = {}
    # lock = mp.Lock()  # note: when using pool better to lock process wnen writing on stdout e.g. with tqdm
    with mp.Pool(workers) as pool:
        for (group_name, sensor_index, avg), dfs in tqdm(
                zip(headers, pool.imap_unordered(kwargs_wrapper(fetch_sensor), kwargs_list)),
                desc=f"Main fetching data process (#pid:{os.getppid()})",
                position=0,
                total=len(headers),
        ):
            # data[group_name, sensor_index, avg] = dfs
            yield (group_name, sensor_index, avg), dfs
    sys.stdout.flush()
    sys.stderr.flush()
    # for _ in range(i + 1):
    #     print(flush=True)
    # # return data


if __name__ == "__main__":
    pd.set_option('display.max_columns', None, 'display.expand_frame_repr', False)
    start_time = time.perf_counter()
    start_time_process = time.process_time()
    start_time_thread = time.thread_time()

    config = configparser.ConfigParser()
    config.read('keys/PurpleAir_API_key.conf')

    # PurpleAir
    purple_air = PurpleAir(config)

    print(f"sensors_groups.keys():", sensors_groups.keys())
    sensor_group_names = [
        'santa_monica',
        'agrinio',
        'san_francisco',
    ]
    print(f"sensor_group_names:", sensor_group_names)
    average_in_minutes_list = [24 * 60, 6 * 60, 60, 10,]  # frequency in minutes
    print(f"average_in_minutes_list: {average_in_minutes_list}")
    print()

    data = defaultdict(dict)
    for (group_name, sensor_index, avg), dfs in fetch_data(
        purple_air_api=purple_air,
        sensor_group_names=sensor_group_names,
        average_in_minutes=average_in_minutes_list,
        concat_dfs=True,
        end_date=None,
        # end_date=pd.to_datetime('2023-04-01', utc=True),  # keep the same date for having fixed calls
        time_span=pd.Timedelta(days=5 * 365),
        datadir='data/',
        workers=500,
    ):
        data[group_name, avg][sensor_index] = dfs
    print()
    print()

    print(data.keys())
    first_k = next(iter(data.keys()))
    print(first_k, data[first_k].keys())
    first_k_lev2 = next(iter(data[first_k].keys()))
    print(f"{(first_k, first_k_lev2)}:", data[first_k][first_k_lev2], sep='\n')

    # timeing
    tot_time = time.perf_counter() - start_time
    tot_time_process = time.process_time() - start_time_process
    tot_time_thread = time.thread_time() - start_time_thread
    print(f"Evaluations took {timedelta(seconds=round(tot_time))!s} "
          f"({timedelta(seconds=round(tot_time_process))!s} total processes time;"
          f"({timedelta(seconds=round(tot_time_thread))!s} total processes time;).")
