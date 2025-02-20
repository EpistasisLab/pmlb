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
import os
import logging

from .pmlb import fetch_data, get_updated_datasets, get_reviewed_datasets
from .support_funcs import (
    generate_summarystats, get_dataset_stats,
    generate_all_summaries, write_readme,
    last_commit_message, generate_metadata
)
from .dataset_lists import dataset_names

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(module)s: %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def update_metadata_summary(dataset_name, reviewed_datasets, overwrite=True, 
                            local_cache_dir=None, write_summary=False):
    df = fetch_data(dataset_name, local_cache_dir=local_cache_dir, dropna=False)
    dataset_stats = get_dataset_stats(df)
    if dataset_name not in reviewed_datasets:
        generate_metadata(df, dataset_name, dataset_stats, overwrite, local_cache_dir)

    with open(pathlib.Path(f'{local_cache_dir}{dataset_name}/metadata.yaml')) as f:
        meta_dict = yaml.load(f, Loader=yaml.FullLoader)

    dataset_stats['yaml_task'] = meta_dict['task']

    generate_summarystats(dataset_name, dataset_stats, local_cache_dir, write_summary)

def datasets_to_update() -> list:
    """
    Return datasets to regenerate profiles for 
    """
    if '[update_all_datasets]' in last_commit_message():
        print('"update_all_datasets=true" >> $GITHUB_ENV')
        return {
            'changed_datasets': dataset_names,
            'changed_metadatas': dataset_names
        }
    if 'update_all_datasets' in os.environ:
        return {
            'changed_datasets': dataset_names,
            'changed_metadatas': dataset_names
        }
    updated_sets = get_updated_datasets()
    return updated_sets

if __name__ =='__main__':
    # assuming this is run from the repo root directory
    local_dir = 'datasets/'

    # which datasets have changed for this commit
    updated_sets = datasets_to_update()
    updated_datasets = updated_sets['changed_datasets']
    updated_metadatas = updated_sets['changed_metadatas']
    reviewed_datasets = get_reviewed_datasets(dataset_names)
    generate_all_summaries(local_cache_dir=local_dir)

    for dataset_name in updated_datasets:
        print(f'Adding readme, metadata and summary stats for {dataset_name}...')
        # update dataset specific readme files
        write_readme(dataset_name)
        # add metadata and summary_stats
        update_metadata_summary(
            dataset_name, reviewed_datasets,
            overwrite=True,
            local_cache_dir=local_dir,
            write_summary=True)

    for dataset_name in updated_metadatas:
        print(f'Updating summary stats for {dataset_name}...')
        # update summary_stats from updated metadata
        update_metadata_summary(
            dataset_name, reviewed_datasets,
            overwrite=True,
            local_cache_dir=local_dir,
            write_summary=True)

    # update summary_stats from updated metadata
    generate_all_summaries(local_cache_dir=local_dir)
