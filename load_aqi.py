import builtins as __builtin__
from collections.abc import Iterable
from collections import deque
from io import BytesIO
import math
import os
from pathlib import Path
from typing import Literal, get_args
from zipfile import ZipFile

import pandas as pd

from load_dataset_beijing import parse_idx, load_dataset, get_dataset_name_from_index, get_archive
from air import caqi_1h


SUPPORTED_AQI_TYPES = Literal['caqi_1h']


def _get_aqi_filename(filename: str | bytes | os.PathLike,
                      data_dir: str | bytes | os.PathLike,
                      aqi_type: str,
                      iaqi: bool) -> Path:
    global SUPPORTED_AQI_TYPES
    if aqi_type not in get_args(SUPPORTED_AQI_TYPES):
        raise ValueError(f"'aqi_type' ('{aqi_type}') not in supported aqi types: {get_args(SUPPORTED_AQI_TYPES)}")
    assert isinstance(iaqi, bool)
    assert isinstance(aqi_type, str)
    aqi_filename = Path(os.path.join(data_dir, f'aqi_{aqi_type}_i{iaqi}_' + os.path.basename(filename)))
    # print(filename, aqi_filename, sep='\t\t')
    return aqi_filename


def load_aqi(
            idx: int | Iterable[int] = None,
            aqi_type: SUPPORTED_AQI_TYPES = 'caqi_1h',
            iaqi: bool = False,
            data_dir: str | bytes | os.PathLike = 'data/',
        ) -> pd.DataFrame:
    iaqi = bool(iaqi)
    indexes = parse_idx(idx)
    filenames = get_dataset_name_from_index(idx)
    with get_archive() as archive:
        missing_aqi_indexes = []
        cache_hit_aqi_indexes = []
        aqi_filenames = []
        for i, filename in zip(indexes, filenames):
            aqi_filename = _get_aqi_filename(filename, data_dir, aqi_type, iaqi)
            aqi_filenames.append(aqi_filename)
            if not aqi_filename.exists():
                # print(f"Cache miss.")
                missing_aqi_indexes.append(i)
            else:
                # print(f"Cache hit.")
                cache_hit_aqi_indexes.append(i)
        # print()
        iter_missing = iter(load_dataset(missing_aqi_indexes))
        j_missing = 0
        for i, filename, aqi_filename in zip(indexes, filenames, aqi_filenames):
            if missing_aqi_indexes and i == missing_aqi_indexes[j_missing]:
                print(f"Computing '{aqi_filename}'... ")
                # compute AQI
                _filename, df = next(iter_missing)
                assert filename == _filename, (filename, _filename)
                aqi = caqi_1h(df)
                # print(aqi)
                print(f"Saving '{aqi_filename}'... ")
                # save AQI
                aqi.to_csv(aqi_filename)
                # print(f"Computing and saving '{aqi_filename}' done.")
                j_missing += 1
            else:
                print(f"Loading '{aqi_filename}'...")  # , end='', flush=True)
                # load AQI
                aqi = pd.read_csv(aqi_filename)
                aqi['time_stamp'] = pd.to_datetime(aqi['time_stamp'])
                aqi.set_index('time_stamp', inplace=True)
                # print(aqi)
                # print(f"Loading '{aqi_filename}' done.")
        if missing_aqi_indexes:
            try:
                next(iter_missing)
            except StopIteration:
                pass
            else:
                raise AssertionError('More datasets to iter than expected')
        return aqi


if __name__ == "__main__":
    pd.set_option('display.max_rows', pd.get_option("display.max_rows"))
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None, 'display.expand_frame_repr', True)
    # pd.set_option('display.width', 999, 'display.expand_frame_repr', True)
    pd.set_option('display.expand_frame_repr', False)

    print('#' * 80)
    load_aqi(2)
    print('#' * 80)
    load_aqi([2, 3])

    print('#' * 80)
    load_aqi(2, iaqi=True)
    print('#' * 80)
    load_aqi([2, 3], iaqi=True)

