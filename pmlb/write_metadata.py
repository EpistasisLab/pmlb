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

import glob, os
import csv
import pandas as pd
from collections import Counter
from pmlb import fetch_data
from .dataset_lists import (classification_dataset_names,
                            regression_dataset_names)
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

def imbalance_metrics(data):
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
    if x.dtype=='float64':
        return 'continuous'
    elif x.dtype=='int64':
        if include_binary:
            if x.nunique() == 2:
                return 'binary'
        return 'categorical'
    else:
        raise ValueError("Error getting type")

def generate_description(df, dataset_name, task, overwrite_existing=True,
                         local_cache_dir=None):
    """Generates desription for a given dataset in its metadata.yaml file in a
    dataset local_cache_dir file.

    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    """

    print('generating metadata for',dataset_name)
    header_to_print = '#Generated automatically by pmlb/write_metadata.py\n'
    assert (local_cache_dir!=None)
    metadata_filename = os.path.join(local_cache_dir,
        dataset_name, 'metadata.yaml')
    if (not overwrite_existing) and os.path.isfile(metadata_filename):
        logger.warning('Not writing '+ dataset_name + '/metadata.yaml ; '
                   + 'File exists (use overwrite_existing=True to override\n')
    metadata_file = open(metadata_filename, 'r')
    header = metadata_file.readline()
    if header != header_to_print:

        logger.warning('Not writing '+ dataset_name + '.yaml ; '
                       + 'It has a customized metadata file\n')
        return

    metadata_file.close()
    metadata_file = open(metadata_filename, 'w')
    try:
        fnames = [col for col in df.columns if col!=TARGET_NAME]
        #determine all required values
        df_X = df.drop(TARGET_NAME,axis=1)
        types = [get_type(df_X[col]) for col in df_X.columns]
        endpoint=get_type(df[TARGET_NAME], include_binary=True)
        #proceed with writing
        none_yet = ('None yet. '
                    'See our contributing guide to help us add one.')
        metadata_file.write(header_to_print)
        # required, dataset name
        metadata_file.write('dataset: {}\n'.format(dataset_name))
        # required, dataset description
        metadata_file.write('description: {}\n'.format(none_yet))
        # required, link to the source from where dataset was retrieved
        metadata_file.write('source: {}\n'.format(none_yet))
        # optional, study that generated the dataset (doi, pmid, pmcid,
        # or url)
        metadata_file.write('publication: {}\n'.format(none_yet))
        # required, classification or regression
        metadata_file.write('task: {}\n'.format(task))
        metadata_file.write('target:\n')
        metadata_file.write('  type: {}\n'.format(endpoint))
        # required, describe the endpoint/outcome (and unit if exists)
        metadata_file.write('  description: {}\n'.format(none_yet))
        # optional but recommended, coding information,
        # e.g., 'Control' = 0, 'Case' = 1
        metadata_file.write('  code: {}\n'.format(none_yet))
        metadata_file.write('features: # list of features in the '
                'dataset\n')
        first = True
        for feature,feature_type in zip(fnames, types):
            if feature in protected_feature_names:
                feature = '"'+feature+'"'
            metadata_file.write('  - name: {}\n'.format(feature))
            metadata_file.write('    type: {}\n'.format(feature_type))
            if first:
                metadata_file.write('    description: null # optional but '
                        'recommended, what the feature measures/indicates, '
                        'unit\n')
                metadata_file.write('    code: null # optional, coding '
                        'information, e.g., Control = 0, Case = 1\n')
                metadata_file.write('    transform: ~ # optional, any '
                'transformation performed on the feature, e.g., log '
                'scaled\n')
                first = False

    except IOError as err:
        print(err)
    finally:
        metadata_file.close()

def generate_summarystats(df, dataset_name, task, local_cache_dir=None):
    """Generates summary stats for a given dataset in its summary_stats.csv
    file in a dataset local_cache_dir file.
    TODO: link dataset_desribe from PennAI to this for generating stats.
    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    """
    print('generating summary stats for',dataset_name)
    df_X = df.drop('target',axis=1)
    types = [get_type(df_X[col], include_binary=True) for col in df_X.columns]
    feat=count_features_type(types, include_binary=True)
    mse=imbalance_metrics(df[TARGET_NAME].tolist())

    stats_df = pd.DataFrame({
        '#instances':len(df),
        '#features':len(df.columns)-1,
        '#binary_features':feat[0],
        '#categorical_features':feat[1],
        '#continuous_features':feat[2],
        'Endpoint_type':get_type(df[TARGET_NAME], include_binary=True),
        '#Classes':mse[0],
        'Imbalance_metric':mse[1],
        },index=[0])

    assert (local_cache_dir!=None)
    stats_df.to_csv(os.path.join(local_cache_dir,dataset_name,
        'summary_stats.csv'))

if __name__ =='__main__':

    # assuming this is run from the repo root directory
    local_dir = 'datasets/'
    overwrite = True

    for d in classification_dataset_names:
        print(d,'...')
        df = fetch_data(d, local_cache_dir=local_dir)
        generate_description(df, d,'classification',
                overwrite_existing=overwrite,
                local_cache_dir=local_dir)
        generate_summarystats(df, d,'classification',local_cache_dir=local_dir)
    for d in regression_dataset_names:
        print(d,'...')
        df = fetch_data(d, local_cache_dir=local_dir)
        generate_description(df, d,'regression',
                overwrite_existing=overwrite,
                local_cache_dir=local_dir)
        generate_summarystats(df, d,'regression', local_cache_dir=local_dir)
