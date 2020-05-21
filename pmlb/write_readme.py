# -*- coding: utf-8 -*-

"""
PMLB was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - William La Cava (lacava@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Trang Le (ttle@pennmedicine.upenn.edu)
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

import glob, os
import csv
import pandas as pd
from collections import Counter
import numpy as np
import pdb
from pmlb import fetch_data
from .dataset_lists import classification_dataset_names, regression_dataset_names

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

def write_readme(dataset, problem_type, df):
    """Writes a readme file for a dataset."""

    filename = 'datasets/'+dataset+'/README.md'
    summary_stats = pd.read_csv(filename.split('README.md')[0]
                                +'summary_stats.csv')
    print('summary_stats:',summary_stats)
    if os.path.isfile(filename):
        print('WARNING:',filename,'exists. Overwriting...') 
    with open(filename, 'w') as f: 
        f.write('# %s\n\n' % dataset)
        f.write('[Metadata](metadata.yaml) |'
                ' [Summary Statistics](summary_stats.csv)\n\n')
        f.write('## Summary\n\n')
        f.write('**task**: %s\n\n' % problem_type)
        f.write('**instances**: %s\n\n' % summary_stats['#instances'].values[0])
        f.write('**features**: %s\n\n' % summary_stats['#features'].values[0])
        if problem_type == 'classification':
            f.write('**number of classes**: {}\n\n'.format(
                summary_stats['#features'].values[0]))
        f.write('## Summary Plots\n\n')
        f.write('![Labels](label.svg)\n\n')
        f.write('![Corr](corr.svg)\n\n')

        # make markdown table
        f.write('## Data Summary\n\n')
        summary = df.describe()
        print('summary:',summary)
        f.write('|\tvariable')
        for stat in summary.index:
            f.write('\t|\t{}'.format(stat))
        f.write('|\n')
        for i in np.arange(len(summary)+1):
            f.write('| --- ')
        f.write('|\n')
        for col in summary.columns:
            f.write('|\t'+col)
            for stat in summary.index:
                s = typify(summary.loc[stat,col])
                if type(s) == int:
                    f.write('\t|\t{:d}'.format(s))
                else:
                    f.write('\t|\t{:6.2f}'.format(s))
            f.write('\n')

if __name__ =='__main__':

    # assuming this is run from the repo root directory
    local_dir='datasets/'

    for d in classification_dataset_names:
    # for d in ['adult']:
        print(d,'...')
        df = fetch_data(d, local_cache_dir=local_dir)
        write_readme(d, 'classification', df)
    for d in regression_dataset_names:
    # for d in ['1027_ESL']:
        print(d,'...')
        df = fetch_data(d, local_cache_dir=local_dir)
        write_readme(d, 'regression', df)

