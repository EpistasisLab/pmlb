import pandas as pd
from pmlb import fetch_data, dataset_names, get_updated_datasets
from pandas_profiling import ProfileReport
import pathlib
import os

def make_profiling(dataset, write_dir, dat_dir='datasets/'):
    df = fetch_data(dataset, local_cache_dir=dat_dir)
    write_path = write_dir.joinpath(dataset + '.html')

    if len(df.columns) > 20:
        profile = ProfileReport(df, title=dataset, explorative=True, minimal=True)
    else:
        profile = ProfileReport(df, title=dataset, explorative=True)

    profile.to_file(write_path)

def datasets_to_gen():
    if 'regenerate_profiles' in os.environ:
        return dataset_names
    return get_updated_datasets()


if __name__ =='__main__':

    # write_dir = 'docs_sources/profile/'
    write_dir = pathlib.Path('docs_sources/profile/')
    write_dir.mkdir(exist_ok=True)

    datasets = datasets_to_gen()

    for dataset in datasets:
        write_path = write_dir.joinpath(dataset + '.html')
        make_profiling(dataset, write_dir)
