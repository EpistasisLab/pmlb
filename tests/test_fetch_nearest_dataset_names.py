import pandas as pd
from pmlb import fetch_data, nearest_datasets


def test_nearest_dataset_is_itself():
    """Tests whether the nearest dataset is itself"""

    for test_dataset in ['lupus', 'analcatdata_aids']:
        df = fetch_data(test_dataset, local_cache_dir='../datasets/')
        nearest = nearest_datasets(df, task='classification', n=10)
        print('nearest to',test_dataset,':',nearest)
        assert(nearest[0] == test_dataset)

