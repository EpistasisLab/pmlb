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
    for f in glob('datasets/*/summary_stats.tsv'):
        frames.append(pd.read_csv(f, sep = '\t'))
    df_summary = pd.concat(frames)
    
    df_summary.to_csv('pmlb/all_summary_stats.tsv', index=False, sep='\t')

    # nclass_datasets = len(df_summary.loc[df_summary.task=='classification'])
    # nreg_datasets = len(df_summary.loc[df_summary.task=='regression'])

    # sns.set(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 6))
    # sns.scatterplot("n_instances", "n_features", data=df_summary,
    #                  edgecolor='w', ax=ax, hue='task')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # h, l = ax.get_legend_handles_labels()
    # print('labels:',l)
    # ax.legend(
    # plt.title('Dataset Sizes')
    # plt.savefig('datasets/dataset_sizes.svg',dpi=300)
