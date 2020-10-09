import pandas as pd
import pathlib
import subprocess
from collections import Counter

TARGET_NAME = 'target'
# these are words that yaml doesn't like you to use
protected_feature_names = ['y','Y','yes','Yes','YES','n','N','no','No','NO',
                           'true','True','TRUE','false','False','FALSE',
                           'on','On','ON','off','Off','OFF']

def get_feature_type(x, include_binary=False):
    x.dropna(inplace=True)
    if not check_if_all_integers(x):
        return 'continuous'
    else:
        if x.nunique() > 10:
            return 'continuous'
        if include_binary:
            if x.nunique() == 2:
                return 'binary'
        return 'categorical'

def get_target_type(x, include_binary=False):
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

def check_if_all_integers(x):
    "check a pandas.Series is made of all integers."
    return all(float(i).is_integer() for i in x.unique())
    
def generate_summarystats(dataset_name, dataset_stats, local_cache_dir=None,
                          write_summary=False):
    """Generates summary stats for a given dataset in its summary_stats.csv
    file in a dataset local_cache_dir file.
    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    :param write_summary: bool
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

    if write_summary:
        assert (local_cache_dir != None)
        stats_df.to_csv(pathlib.Path(f'{local_cache_dir}{dataset_name}'
            '/summary_stats.tsv'), index=False, sep='\t')

    return stats_df

def get_dataset_stats(df):
    feat_names = [col for col in df.columns if col!=TARGET_NAME]
    types = [get_feature_type(df[col], include_binary=True) for col in feat_names]
    feat = count_features_type(types, include_binary=True)
    endpoint = get_target_type(df[TARGET_NAME])
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

def generate_all_summaries(local_cache_dir='datasets/'):
    frames = []
    for f in sorted(pathlib.Path(local_cache_dir).glob('*/summary_stats.tsv')):
        frames.append(pd.read_csv(f, sep = '\t'))
    pd.concat(frames).to_csv('pmlb/all_summary_stats.tsv', index=False, sep='\t')

def generate_metadata(df, dataset_name, dataset_stats, overwrite_existing=True,
                         local_cache_dir=None):
    """Generates description for a given dataset in its metadata.yaml file in a
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

def write_readme(dataset, local_cache_dir='datasets/'):
    readme_template = '''\
# {dataset}

[**Pandas Profiling Report**](https://epistasislab.github.io/pmlb/profile/{dataset}.html)

[Metadata](metadata.yaml) | [Summary Statistics](summary_stats.tsv)

'''
    """Writes a readme file for a dataset."""
    print(dataset)
    path = pathlib.Path(f'{local_cache_dir}{dataset}/README.md')

    if path.exists():
        print(f'WARNING: {path} exists. Overwriting...')
    readme = readme_template.format(dataset=dataset)
    path.write_text(readme)

def last_commit_message() -> str:
    """
    Get commit message from last commit, excluding merge commits
    """
    command = "git log --no-merges -1 --pretty=%B".split()
    message = subprocess.check_output(command, universal_newlines=True)
    return message
