# -*- coding: utf-8 -*-

"""
PMLB was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - William La Cava (lacava@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pandas as pd
import os
from .dataset_lists import classification_dataset_names, regression_dataset_names
import requests
import warnings
import subprocess
import pathlib

dataset_names = classification_dataset_names + regression_dataset_names
GITHUB_URL = 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets'
suffix = '.tsv.gz'

def fetch_data(dataset_name, return_X_y=False, local_cache_dir=None, dropna=True):
    """Download a data set from the PMLB, (optionally) store it locally, and return the data set.

    You must be connected to the internet if you are fetching a data set that is not cached locally.

    Parameters
    ----------
    dataset_name: str
        The name of the data set to load from PMLB.
    return_X_y: bool (default: False)
        Whether to return the data in scikit-learn format, with the features and labels stored in separate NumPy arrays.
    local_cache_dir: str (default: None)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    dropna: bool
        If True, pmlb will drop NAs in exported dataset.

    Returns
    ----------
    dataset: pd.DataFrame or (array-like, array-like)
        if return_X_y == False: A pandas DataFrame containing the fetched data set.
        if return_X_y == True: A tuple of NumPy arrays containing (features, labels)

    """
    if dataset_name not in dataset_names:
        raise ValueError('Dataset not found in PMLB.')


    if local_cache_dir is None:
        dataset_url = get_dataset_url(GITHUB_URL,
                                        dataset_name, suffix)
        dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
    else:
        dataset_path = os.path.join(local_cache_dir, dataset_name,
                                    dataset_name+suffix)

        # Use the local cache if the file already exists there
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, sep='\t', compression='gzip')
        # Download the data to the local cache if it is not already there
        else:
            dataset_url = get_dataset_url(GITHUB_URL,
                                            dataset_name, suffix)
            dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
            dataset_dir = os.path.split(dataset_path)[0]
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
            dataset.to_csv(dataset_path, sep='\t', compression='gzip',
                    index=False)

    if dropna:
        dataset.dropna(inplace=True)
    if return_X_y:
        X = dataset.drop('target', axis=1).values
        y = dataset['target'].values
        return (X, y)
    else:
        return dataset


def get_dataset_url(GITHUB_URL, dataset_name, suffix):
    dataset_url = '{GITHUB_URL}/{DATASET_NAME}/{DATASET_NAME}{SUFFIX}'.format(
                                GITHUB_URL=GITHUB_URL,
                                DATASET_NAME=dataset_name,
                                SUFFIX=suffix
                                )

    re = requests.get(dataset_url)
    if re.status_code != 200:
        raise ValueError('Dataset not found in PMLB.')
    return dataset_url


def get_updated_datasets():
    """Looks at commit and returns a list of datasets that were updated."""
    cmd = 'git diff --name-only HEAD HEAD~1'
    res = subprocess.check_output(cmd.split(), universal_newlines=True)
    changed_datasets = set()
    for path in res.splitlines():
        path = pathlib.Path(path)
        if path.parts[0] != 'datasets':
            continue
        if path.name == 'metadata.yaml' or path.name.endswith('.tsv.gz'):
            changed_datasets.add(path.parts[-2])
    changed_datasets &= set(dataset_names)
    changed_datasets = sorted(changed_datasets)
    print(f'changed datasets: {changed_datasets}')
    return changed_datasets

from sklearn.neighbors import NearestNeighbors
from pmlb.write_metadata import generate_summarystats

def fetch_nearest_dataset_names(X,y=None, **kwargs):
    """
    X: numpy array
        an n_samples x n_features array of independent variables
    y: numpy array or None (default: None)
        a n_samples array of dependent variables
    """
    df = pd.DataFrame({**{'x_'+str(i):x for i,x in enumerate(X.transpose)}
                       **{'target':y}})
    return fetch_nearest_dataset_names(df, **kwargs)

def fetch_nearest_dataset_names(df, n=1, 
        dimensions=['#instances','#features'],
        task=None):
    """Returns names of most similar datasets to df, in order of similarity. 

    Parameters
    ----------
    df: pandas Dataframe 
        a dataframe of n_samples x n_features+1 with a target column labeled
        'target'
    n: int (default: 1)
        the number of dataset names to return
    dimensions: list of str (default ['NumberOfInstances','NumberOfFeatures']
        a list of dataset characteristics to include in similarity calculation.
        Dimensions must correspond to columns of datasets/all_summary_stats.csv.
    task: str or None (default: None)
        specify classification or regression for summary stat generation. If 
        None, we use classification unless the target column has more than 5 
        unique values.

    Returns
    -------
    dataset_names: an n-element list of dataset names in order of most similar 
        to least similar.
    """

    # load pmlb summary stats
    pmlb_stats = pd.read_csv('datasets/all_summary_stats.csv')
    assert(all([d in pmlb_stats.columns for d in dimensions]))

    all_names = pmlb_stats['dataset'].values
    pmlb_stats = pmlb_stats[dimensions]  

    # get summary stats for dataset
    if task==None:
        task = 'classification' if df['target'].nunique()<5 else 'regression'
    dataset_stats = generate_summarystats(df, 'dataset', task)
    dataset_stats = dataset_stats[dimensions]

    # find nearest neighbors
    nn = NearestNeighbors(n_neighbors=n).fit(pmlb_stats.values)
    dataset_names = all_names[nn.kneighbors(dataset_stats.values, 
                                            n_neighbors=n, 
                                            return_distance=False)
                             ]

    return dataset_names
