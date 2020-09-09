# -*- coding: utf-8 -*-

"""
PMLB was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - William La Cava (lacava@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Trang Le (ttle@pennmedicine.upenn.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pathlib
import yaml
import pandas as pd
from collections import Counter
from .pmlb import fetch_data, dataset_names, get_updated_datasets
from .dataset_lists import datasets_with_metadata
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(module)s: %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TARGET_NAME = 'target'

# these are words that yaml doesn't like you to use
protected_feature_names = ['y','Y','yes','Yes','YES','n','N','no','No','NO',
                           'true','True','TRUE','false','False','FALSE',
                           'on','On','ON','off','Off','OFF']

def compute_imbalance(data):
    """ Computes imbalance metric for a given dataset.
    Imbalance metric is equal to 0 when a dataset is perfectly balanced
    (i.e. number of in each class is exact).
    :param data : pandas.DataFrame
        A dataset in a panda's data frame
    :returns int
        A value of imbalance metric, where zero means that the dataset is
        perfectly balanced and the higher the value, the more imbalanced the
        dataset.
    """
    if not data:
        return 0
    #imb - shows measure of inbalance within a dataset
    imb = 0
    num_classes=float(len(Counter(data)))
    for x in Counter(data).values():
        p_x = float(x)/len(data)
        if p_x > 0:
            imb += (p_x - 1/num_classes)*(p_x - 1/num_classes)
    #worst case scenario: all but 1 examplars in 1st class, the remaining one
    #in 2nd class
    worst_case=(num_classes-1)*pow(1/num_classes,2) + pow(1-1/num_classes,2)
    return (num_classes,imb/worst_case)

def count_features_type(types, include_binary=False):
    """ Counts two or three different types of features
    (binary (optional), categorical, continuous).
    :param types: list of types from get_type
    :returns a tuple (binary (optional), categorical, continuous)
    """
    if include_binary:
        return (
                types.count('binary'),
                types.count('categorical'),
                types.count('continuous')
                )
    else:
        return (
                types.count('categorical'),
                types.count('continuous')
                )

def get_type(x, include_binary=False):
    x.dropna(inplace=True)
    if x.dtype=='float64':
        return 'continuous'
    elif x.dtype=='int64':
        if include_binary:
            if x.nunique() == 2:
                return 'binary'
        return 'categorical'
    else:
        raise ValueError("Error getting type")

def get_dataset_stats(df):
    feat_names = [col for col in df.columns if col!=TARGET_NAME]
    types = [get_type(df[col], include_binary=True) for col in feat_names]
    feat = count_features_type(types, include_binary=True)
    endpoint = get_type(df[TARGET_NAME])
    mse = compute_imbalance(df[TARGET_NAME].tolist())
    task = 'regression' if endpoint == 'continuous' else 'classification'

    return {
        'n_instances': len(df),
        'n_features': len(df.columns)-1,
        'feat_names': feat_names,
        'types': types,
        'feat': feat,
        'endpoint': endpoint,
        'task': task,
        'mse': mse
    }


def generate_metadata(df, dataset_name, dataset_stats, overwrite_existing=True,
                         local_cache_dir=None):
    """Generates desription for a given dataset in its metadata.yaml file in a
    dataset local_cache_dir file.

    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    """

    metadata_template = '''\
{header_to_print}
dataset: {dataset_name}
description: {none_yet}
source: {none_yet}
publication: {none_yet}
task: {task}
keywords:
  -
  -
target:
  type: {endpoint}
  description: {none_yet}
  code: {none_yet}
features:
{all_features}\
'''
    feature_template = '''\
  - name: {feat_name}
    type: {feat_type}
'''
    feat_extra_template = '''\
    description:
    code:
    transform:
'''
    feat_extra_first = '''\
    description: # optional but recommended, what the feature measures/indicates, unit
    code: # optional, coding information, e.g., Control = 0, Case = 1
    transform: # optional, any transformation performed on the feature, e.g., log scaled
'''
    
    none_yet = ('None yet. See our contributing guide to help us add one.')
    header_to_print = '# Reviewed by [your name here]'
    assert (local_cache_dir != None)
    meta_path = pathlib.Path(f'{local_cache_dir}{dataset_name}/metadata.yaml')
    if meta_path.exists():
        if (not overwrite_existing):
            logger.warning(f'Not writing {dataset_name}/metadata.yaml ; '
                            'File exists (use overwrite_existing=True to override.\n')
            return None

        print(f'WARNING: {meta_path} exists. Overwriting...')

    print('Generating metadata for', dataset_name)

    all_features = ''
    first = True
    for feature, feature_type in zip(dataset_stats['feat_names'], dataset_stats['types']):
        if feature in protected_feature_names:
            feature = f'"{feature}"'
        all_features += feature_template.format(
            feat_name=feature,
            feat_type=feature_type
        )
        if first:
            all_features += feat_extra_first
            first = False
        else:
            all_features += feat_extra_template

    metadata = metadata_template.format(
        header_to_print=header_to_print,
        dataset_name=dataset_name,
        none_yet=none_yet,
        endpoint=dataset_stats['endpoint'],
        task=dataset_stats['task'],
        all_features=all_features
    )
    
    try:
        meta_path.write_text(metadata)
    except IOError as err:
        print(err)

def generate_summarystats(dataset_name, dataset_stats, local_cache_dir=None,
                          update_all=False):
    """Generates summary stats for a given dataset in its summary_stats.csv
    file in a dataset local_cache_dir file.
    TODO: link dataset_desribe from PennAI to this for generating stats.
    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    :param update_all: bool
        Whether new summary statistics should be written out to directory.
    """
    print('generating summary stats for', dataset_name)

    feat = dataset_stats['feat']
    mse = dataset_stats['mse']

    stats_df = pd.DataFrame({
        'dataset':dataset_name,
        'n_instances':dataset_stats['n_instances'],
        'n_features':dataset_stats['n_features'],
        'n_binary_features':feat[0],
        'n_categorical_features':feat[1],
        'n_continuous_features':feat[2],
        'endpoint_type':dataset_stats['endpoint'],
        'n_classes':mse[0],
        'imbalance':mse[1],
        'task':dataset_stats['yaml_task']
        }, index=[0])

    if update_all:
        assert (local_cache_dir != None)
        stats_df.to_csv(pathlib.Path(f'{local_cache_dir}{dataset_name}/summary_stats.tsv'),
                    index=False, sep='\t')

    return stats_df

def generate_all_summaries(local_cache_dir='datasets/'):
    frames = []
    for f in sorted(pathlib.Path(local_cache_dir).glob('*/summary_stats.tsv')):
        frames.append(pd.read_csv(f, sep = '\t'))
    pd.concat(frames).to_csv('pmlb/all_summary_stats.tsv', index=False, sep='\t')

def update_metadata_summary(dataset_name, datasets_with_metadata, overwrite=False, local_cache_dir=None, update_all=False):
    df = fetch_data(dataset_name, local_cache_dir=local_cache_dir, dropna=False)
    dataset_stats = get_dataset_stats(df)
    if dataset_name not in datasets_with_metadata:
        generate_metadata(df, dataset_name, dataset_stats, overwrite, local_cache_dir)
    
    with open(pathlib.Path(f'{local_cache_dir}{dataset_name}/metadata.yaml')) as f:
        meta_dict = yaml.load(f, Loader=yaml.FullLoader)

    dataset_stats['yaml_task'] = meta_dict['task']

    generate_summarystats(dataset_name, dataset_stats, local_cache_dir, update_all)

def write_readme(dataset, local_cache_dir='datasets/'):
    readme_template = '''\
# {dataset}

[**Pandas Profiling Report**](https://epistasislab.github.io/penn-ml-benchmarks/profile/{dataset}.html)

[Metadata](metadata.yaml) | [Summary Statistics](summary_stats.tsv)
'''
    """Writes a readme file for a dataset."""
    print(dataset)
    path = pathlib.Path(f'{local_cache_dir}{dataset}/README.md')

    if path.exists():
        print(f'WARNING: {path} exists. Overwriting...')
    readme = readme_template.format(dataset=dataset)
    path.write_text(readme)

if __name__ =='__main__':
    # assuming this is run from the repo root directory
    local_dir = 'datasets/'

    # overwrite = True
    # for d in dataset_names:
    #     print(d, '...')
    #     update_metadata_summary(
    #         d, datasets_with_metadata,
    #         overwrite=overwrite,
    #         local_cache_dir=local_dir)

    # which datasets have changed for this commit
    updated_sets = get_updated_datasets()
    updated_datasets = updated_sets['changed_datasets']
    updated_metadatas = updated_sets['changed_metadatas']

    for dataset_name in updated_datasets:
        print(f'Adding readme, metadata and summary stats for {dataset_name}...')
        # update dataset specific readme files
        write_readme(dataset_name)
        # add metadata and summary_stats
        update_metadata_summary(
            dataset_name, datasets_with_metadata,
            overwrite=False,
            local_cache_dir=local_dir,
            update_all=True)

    for dataset_name in updated_metadatas:
        print(f'Updating summary stats for {dataset_name}...')
        datasets_with_metadata.append(dataset_name)
        # update summary_stats from updated metadata
        update_metadata_summary(
            dataset_name, datasets_with_metadata,
            overwrite=False,
            local_cache_dir=local_dir,
            update_all=True)

    # update summary_stats from updated metadata
    generate_all_summaries(local_cache_dir=local_dir)
