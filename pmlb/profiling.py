import pandas as pd
from pmlb import fetch_data, dataset_names, get_updated_datasets
from pandas_profiling import ProfileReport
import pathlib

def make_profiling(dataset, write_dir, dat_dir='datasets/'):
    df = fetch_data(dataset, local_cache_dir=dat_dir)
    write_path = pathlib.Path(write_dir, dataset + '.html')
    
    profile = ProfileReport(
        df, title=dataset, minimal=True, explorative=True)
    
    profile.to_file(write_path)


if __name__ =='__main__':

    write_dir = 'docs_sources/profilings/'
    p = pathlib.Path(write_dir)
    if not p.exists():
        p.mkdir(parents=True)

    updated_datasets = get_updated_datasets()

    for dataset in dataset_names[11:13]:
        write_path = pathlib.Path(write_dir, dataset + '.html')

        if (dataset not in updated_datasets and write_path.exists()):
            # don't update if the dataset has not changed
            continue
        make_profiling(dataset, write_dir)
