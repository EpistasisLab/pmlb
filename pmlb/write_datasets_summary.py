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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import os    

if __name__ =='__main__':
    # write summary of datasets

    frames = []
    for f in glob('datasets/*/summary_stats.csv'):
        df = pd.read_csv(f)
        df['dataset'] = f.split('datasets/')[-1].split('/')[0]
        if df['Endpoint_type'].values[0]=='continuous':
            df['problem_type'] = 'regression'  
        else:
            df['problem_type'] = 'classification'
        frames.append(df)
    df_summary = pd.concat(frames)
    
    cols = list(df_summary)
    cols.insert(0, cols.pop(cols.index('dataset')))
    (
        df_summary
        .loc[:, cols] # move dataset column to front
        .drop(columns = ['#integer_features', '#float_features'])
        .to_csv('datasets/all_summary_stats.csv', index=False)
    )

    nclass_datasets = len(df_summary.loc[df_summary.problem_type=='classification'])
    nreg_datasets = len(df_summary.loc[df_summary.problem_type=='regression'])

    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot("#instances", "#features", data=df_summary,
                     edgecolor='w', ax=ax, hue='problem_type')
    ax.set_yscale('log')
    ax.set_xscale('log')
    h, l = ax.get_legend_handles_labels()
    print('labels:',l)
    ax.legend(
    plt.title('Dataset Sizes')
    plt.savefig('datasets/dataset_sizes.svg',dpi=300)

    # write readme

    filename = 'datasets/README.md'

    if os.path.isfile(filename):
        print('WARNING:',filename,'exists. Overwriting...')
    with open(filename, 'w') as f:
        f.write('# PMLB Datasets\n\n')

        f.write('Classification datasets: {}\n\n'.format(nclass_datasets))
        f.write('Regression datasets: {}\n\n'.format(nreg_datasets))
        
        f.write('## Summary Plots\n\n')
        f.write('![Dataset_Sizes](dataset_sizes.svg)\n\n')


