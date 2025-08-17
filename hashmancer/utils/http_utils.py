import time
import requests
from requests import RequestException


def post_with_retry(url: str, *, retries: int = 3, backoff: float = 1.0, **kwargs):
    """Send a POST request retrying on transient errors."""
    delay = backoff
    for attempt in range(retries):
        try:
            return requests.post(url, **kwargs)
        except RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def get_with_retry(url: str, *, retries: int = 3, backoff: float = 1.0, **kwargs):
    """Send a GET request retrying on transient errors."""
    delay = backoff
    for attempt in range(retries):
        try:
            return requests.get(url, **kwargs)
        except RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
