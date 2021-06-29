"""
A filesystem cache of previous queries to Geocod.io, so that we don't waste too many queries
(I only get 2500 free queries per day).

Assumes that the project root (or wherever you started your python shell/Jupyter notebook from) has
a file "geocodio_api_key.json" with the contents:
    {
        "key": "API_KEY_HERE"
    }
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List
from geocodio import GeocodioClient

# A cache of format "address -> Geocode response dict" that we will keep committed
# in the repo.
parent_dir = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH = Path(parent_dir + '/data/geocode_cache.json')

client = GeocodioClient(json.loads(Path(parent_dir + '/geocodio_api_key.json').read_text())['key'])

def load_cache() -> Dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open() as f:
        return json.load(f)

def overwrite_cache(cache: Dict[str, dict]) -> None:
    with CACHE_PATH.open('w') as f:
        json.dump(cache, f)

def lookup(addresses: Iterable[str]) -> List[dict]:
    """
    Please don't run this function in parallel, because the cache isn't thread-safe.

    Since it takes an Iterable, it's easy to use this with a Pandas series:
        df['geocode_results'] = geocode_cache.lookup(df['address'])
    """
    cache = load_cache()
    addresses_to_lookup = list(set(addresses) - set(cache.keys()))

    if len(addresses_to_lookup):
        api_results = client.geocode(addresses_to_lookup)
        for address, response in zip(addresses_to_lookup, api_results):
            cache[address] = dict(response)

        overwrite_cache(cache)

    return [cache[address] for address in addresses]
