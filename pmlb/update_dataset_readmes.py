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

from pmlb import fetch_data, get_updated_datasets
from .dataset_lists import (classification_dataset_names,
                            regression_dataset_names)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import operator
from glob import glob
import os
import pathlib

TARGET_NAME = 'target'

protected_feature_names = ['y','Y','yes','Yes','YES','n','N','no','No','NO',
                           'true','True','TRUE','false','False','FALSE',
                           'on','On','ON','off','Off','OFF']

def typify(x):
    """Tries to typecast argument to a numeric type."""
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x

readme_template = '''\
# {dataset}

[**Pandas Profiling Report**](https://epistasislab.github.io/penn-ml-benchmarks/profile/{dataset}.html)

[Metadata](metadata.yaml) | [Summary Statistics](summary_stats.csv)

'''

def write_readme(dataset, problem_type):
    """Writes a readme file for a dataset."""
    print(dataset)
    path = pathlib.Path(f'datasets/{dataset}/README.md')
    # summary_stats = pd.read_csv(path.with_name('summary_stats.csv'))
    if path.exists():
        print(f'WARNING: {path} exists. Overwriting...')
    readme = readme_template.format(
        dataset=dataset,
    )
    path.write_text(readme)

if __name__ =='__main__':

    local_dir = 'datasets/'
    names = {
    	'regression': regression_dataset_names,
    	'classification': classification_dataset_names
        #     'regression': regression_dataset_names[:5],
        #     'classification': classification_dataset_names[:5]
    }

    # figure out which datasets have changed for this commit
    # updated_datasets = get_updated_datasets()
    updated_datasets = regression_dataset_names + classification_dataset_names

    ##
    # update dataset specific readme files
    ##

    for problem_type in ['classification','regression']:
        for dataset in names[problem_type]:
            if dataset not in updated_datasets:
                # don't update if the dataset has not changed
                continue

            write_readme(dataset, problem_type) #, df)
