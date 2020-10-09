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
from .dataset_lists import (
    dataset_names,
    classification_dataset_names, 
    regression_dataset_names)
import requests
import warnings
import subprocess
import pathlib

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .support_funcs import (
    generate_summarystats, 
    get_dataset_stats,
    last_commit_message
)
import numpy as np

GITHUB_URL = 'https://github.com/EpistasisLab/pmlb/raw/master/datasets'
suffix = '.tsv.gz'

def fetch_data(dataset_name, return_X_y=False, local_cache_dir=None, dropna=True):
    """Download a data set from the PMLB, (optionally) store it locally, and return the data set.

    You must be connected to the internet if you are fetching a data set that is not cached locally.

    Parameters
    ----------
    dataset_name: str
        The name of the data set to load from PMLB.
    return_X_y: bool (default: False)
        Whether to return the data in scikit-learn format, with the features 
        and labels stored in separate NumPy arrays.
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

    if local_cache_dir is None:
        if dataset_name not in dataset_names:
            raise ValueError('Dataset not found in PMLB.')
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
            if dataset_name not in dataset_names:
                raise ValueError('Dataset not found in PMLB.')
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

def get_updated_datasets(local_cache_dir='datasets'):
    """Looks at commit and returns a list of datasets that were updated."""
    cmd = 'git diff --name-only HEAD HEAD~1'
    res = subprocess.check_output(cmd.split(), universal_newlines=True).rstrip()
    changed_datasets = set()
    changed_metadatas = set()
    for path in res.splitlines():
        path = pathlib.Path(path)
        if path.parts[0] != 'datasets':
            continue
        if path.name.endswith('.tsv.gz'):
            changed_datasets.add(path.parts[-2])
        if path.name == 'metadata.yaml':
            changed_metadatas.add(path.parts[-2])
            
    datasets_remain = [x.name for x in pathlib.Path(local_cache_dir).iterdir()]
    changed_metadatas &= set(datasets_remain)
    changed_datasets &= set(datasets_remain)

    changed_datasets = sorted(changed_datasets)
    changed_metadatas = sorted(changed_metadatas)
    print(
        f'changed datasets: {changed_datasets}\n'
        f'changed metadata: {changed_metadatas}'
    )
    return {'changed_datasets': changed_datasets,
            'changed_metadatas': changed_metadatas}

def nearest_datasets(X, y=None, task='classification', n=1, 
        dimensions=['n_instances', 'n_features']):
    """
    X: numpy array or pandas DataFrame
        an n_samples x n_features array of independent variables
    y: numpy array or None (default: None)
        a n_samples array of dependent variables
    task: 'regression' or 'classification' (default: 'classification')
        specify the task.
    n: int (default: 1)
        the number of dataset names to return
    dimensions: list of str or str (default: ['NumberOfInstances',
    'NumberOfFeatures'])
        a list of dataset characteristics to include in similarity calculation.
        Dimensions must correspond to columns of datasets/all_summary_stats.csv.
        If 'all', uses all numeric columns.
    """
    if isinstance(X, np.ndarray):
        if y == None:
            ValueError('the target (y) must be specified if a np array '
                    'is passed.')
        df = pd.DataFrame({**{'x_'+str(i):x for i,x in enumerate(X.transpose)}
                           **{'target':y}})
    elif isinstance(X, pd.DataFrame):
        df = X
        
    return fetch_nearest_dataset_names(df, task, n, dimensions)

def fetch_nearest_dataset_names(df, task, n, dimensions):
    """Returns names of most similar datasets to df, in order of similarity. 

    Parameters
    ----------
    df: pandas Dataframe 
        a dataframe of n_samples x n_features+1 with a target column labeled
        'target'
    task: str 
        specify classification or regression for summary stat generation. 
    n: int (default: 1)
        the number of dataset names to return
    dimensions: list of str or str (default: ['NumberOfInstances',
    'NumberOfFeatures'])
        a list of dataset characteristics to include in similarity calculation.
        Dimensions must correspond to columns of datasets/all_summary_stats.csv.
        If 'all', uses all numeric columns.

    Returns
    -------
    dataset_names: an n-element list of dataset names in order of most similar 
        to least similar.
    """

    # load pmlb summary stats
    path = pathlib.Path(__file__).parent / "all_summary_stats.tsv"
    pmlb_stats = pd.read_csv(path, sep = '\t')
    # restrict to same task
    pmlb_stats = pmlb_stats.loc[pmlb_stats.task==task]
    all_names = pmlb_stats['dataset'].values
    # restrict to floating point data in stats
    pmlb_stats = pmlb_stats.apply(
            lambda x: pd.to_numeric(x,errors='coerce')).dropna(axis=1,how='all')

    if dimensions=='all':
        dimensions = list(pmlb_stats.columns)
    else:
        pmlb_stats = pmlb_stats[dimensions]  
        assert(all([d in pmlb_stats.columns for d in dimensions]))

    dataset_stats_tmp = get_dataset_stats(df)
    dataset_stats_tmp['yaml_task'] = task
    dataset_stats = generate_summarystats('dataset', dataset_stats_tmp, 
            write_summary=False)
    dataset_stats = dataset_stats[dimensions]


    # #categorical and #continuous features columns
    ss = StandardScaler()
    pmlb_stats_norm = ss.fit_transform(pmlb_stats) 

    # find nearest neighbors
    nn = NearestNeighbors(n_neighbors=n).fit(pmlb_stats_norm)
    distances, ds = nn.kneighbors(ss.transform(dataset_stats), n_neighbors=n, 
                                            return_distance=True)
    # print([(name, dist) for name, dist in zip(all_names[ds.flatten()],
    #     distances.flatten())])
    dataset_names = all_names[ds.flatten()]

    return dataset_names

def get_reviewed_datasets(dataset_names, local_cache_dir = 'datasets/'):
    reviewed_datasets = []

    for dataset_name in dataset_names:
        if local_cache_dir != None:
            meta_path = pathlib.Path(f'{local_cache_dir}{dataset_name}/metadata.yaml')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    header = f.readline()
        else:
            meta_url = '{GITHUB_URL}/{DATASET_NAME}/metadata.yaml'.format(
                GITHUB_URL=GITHUB_URL,
                DATASET_NAME=dataset_name
            )
            header = requests.get(meta_url).text.splitlines()[0] + '\n'

        if header != '# Reviewed by [your name here]\n':
            reviewed_datasets.append(dataset_name)
            
    return sorted(reviewed_datasets)

def select_datasets(obs_min = None, obs_max = None, feat_min = None, feat_max = None, class_min = None, class_max = None, endpt = None, max_imbalance = None, task = None):
    """Filters existing datasets by given parameters, and returns a list of their names.

    Parameters
    ----------
    obs_min: int (default: None)
        The minimum acceptable number of observations/instances in the dataset
    obs_Max: int (default: None)
        The maximum acceptable number of observations/instances in the dataset
    feat_min: int (default: None)
        The minimum acceptable number of features in the dataset
    feat_max: int (default: None)
        The maximum acceptable number of features in the dataset
    class_min: int (default: None)
        The minimum acceptable number of classes in the dataset
    class_max: int (default: None)
        The maximum acceptable number of classes in the dataset
    max_imbalance: float (default: None)
        Maximum acceptable imbalance value for the dataset
    endpt: str (default: None)
        Whether the dataset endpoint type should be discrete, continuous, categorical, or binary
    task: str (default: None)
        Whether the dataset is suited for classification or regression problems
    Returns
    ----------
    list (str): 
        list of names of datasets within filters. Will return an empty list if no datasets match.


    """

    path = pathlib.Path(__file__).parent / "all_summary_stats.tsv"
    tempdf = pd.read_csv(path, sep = '\t')
    if obs_min is not None:
        tempdf = tempdf.loc[tempdf['n_instances'] >= obs_min]
    if obs_max is not None:
        tempdf = tempdf.loc[tempdf['n_instances'] <= obs_max]
    if feat_min is not None:
        tempdf = tempdf.loc[tempdf['n_features'] >= feat_min]
    if feat_max is not None:
        tempdf = tempdf.loc[tempdf['n_features'] <= feat_max]
    if class_min is not None:
        tempdf = tempdf.loc[tempdf['n_classes'] >= class_min]
    if class_max is not None:
        tempdf = tempdf.loc[tempdf['n_classes'] <= class_max]
    if max_imbalance is not None:
        tempdf = tempdf.loc[tempdf['imbalance'] < max_imbalance]
    if endpt is not None:
        tempdf = tempdf.loc[tempdf['endpoint_type'] == endpt]
    if task is not None:
        tempdf = tempdf.loc[tempdf['task'] == task]
    return list(tempdf['dataset'].values)
