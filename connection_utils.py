import functools
import re
import requests
import time
import os
import random
import warnings
from multiprocessing import Value


def _randrng(a, b):
    if b < a:
        b, a = a, b
    return a + (b - a) * random.random()


class FailsReporter:
    def __init__(self):
        self._fails = Value('i', 0)
        self.start_time = time.perf_counter()

    @property
    def fails_per_sec(self):
        return self.fails / (time.perf_counter() - self.start_time)

    # todo: use a queue and do also fails_per_sec_per_last_minute or above use a param forget_after=#seconds to return it

    @property
    def fails(self):
        return self._fails.value

    def report(self, fails=1):
        with self._fails.get_lock():
            # '+=' is not atomic, it needs the lock.
            self._fails.value += fails


FAILS = FailsReporter()


def _handle_fail(wait: float, err: Exception, *, wait_incremental_factor=1.2):
    FAILS.report()
    print(
        f"{type(err).__qualname__}({err})",
        f"Fails: {FAILS.fails:>4}   per second: {FAILS.fails_per_sec:>2}",
        f"Wait {wait} seconds to reestablish connection and then start from were it was interrupted "
        f"(#PID:{os.getpid()}).",
        sep='\n'
    )
    time.sleep(wait)
    wait *= wait_incremental_factor
    return wait


def try_again_if_connection_fails_decorator(func):
    warn_fails_per_sec_threshold = 30
    start_waiting_time = _randrng(30., 120.)  # 60 // 2
    wait_incremental_factor = 1.3  # 1.25
    if FAILS.fails_per_sec > warn_fails_per_sec_threshold:
        warnings.warn(f"More than {FAILS.fails_per_sec} fails per second.")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try_again = True
        wait = start_waiting_time
        while try_again:
            try_again = False
            try:
                result = func(*args, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError,
                    TimeoutError) as err:
                wait = _handle_fail(wait, err, wait_incremental_factor=wait_incremental_factor)
                try_again = True
            else:
                return result
    return wrapper

