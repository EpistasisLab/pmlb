import pandas as pd
from pmlb import fetch_data, fetch_nearest_dataset_names


def test_nearest_dataset_is_itself():
    """Tests whether the nearest dataset is itself"""

    for test_dataset in ['vote', 'analcatdata_aids', 'car']:
        df = fetch_data(test_dataset, local_cache_dir='../datasets')
        assert(nearest_datasets(df) == test_dataset)

