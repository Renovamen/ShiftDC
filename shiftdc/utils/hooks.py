import time
import random
from functools import wraps
from contextlib import contextmanager

import torch

def retry(max_retries: int = 3, base_delay: float = 5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e

                    sleep_time = base_delay * (2 ** attempt) * random.uniform(0.8, 1.2)

                    print(f"Retry {attempt + 1}/{max_retries}: {e}. Waiting {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@contextmanager
def time_counter(enable: bool):
    if enable:
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time

        print("-" * 50)
        print("-- time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield
