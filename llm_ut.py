# llm_ut.py
import time
from functools import wraps
import logging

log = logging.getLogger("llm-utils")

def retry_with_backoff(retries=3, delay=1.0, backoff=2.0):
    """
    Decorator to retry a function with exponential backoff
    if it raises an exception (network, API error, etc.)
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            wait = delay
            for attempt in range(1, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    log.warning(f"[Retry {attempt}/{retries}] {e}")
                    if attempt == retries:
                        raise
                    time.sleep(wait)
                    wait *= backoff
        return inner
    return wrapper
