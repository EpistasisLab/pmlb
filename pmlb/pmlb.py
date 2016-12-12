# -*- coding: utf-8 -*-

"""
Copyright (c) 2016 Randal S. Olson

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

dataset_names = [
    'GAMETES_Epistasis_2-Way_1000atts_0.4H_EDM-1_EDM-1_1',
    'GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1',
    'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1',
    'GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1',
    'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001',
    'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM-2_001',
    'Hill_Valley_with_noise',
    'Hill_Valley_without_noise',
    'adult',
    'agaricus-lepiota',
    'allbp',
    'allhyper',
    'allhypo',
    'allrep',
    'analcatdata_aids',
    'analcatdata_asbestos',
    'analcatdata_authorship',
    'analcatdata_bankruptcy',
    'analcatdata_boxing1',
    'analcatdata_boxing2',
    'analcatdata_creditscore',
    'analcatdata_cyyoung8092',
    'analcatdata_cyyoung9302',
    'analcatdata_dmft',
    'analcatdata_fraud',
    'analcatdata_germangss',
    'analcatdata_happiness',
    'analcatdata_japansolvent',
    'analcatdata_lawsuit',
    'ann-thyroid',
    'appendicitis',
    'australian',
    'auto',
    'backache',
    'balance-scale',
    'banana',
    'biomed',
    'breast',
    'breast-cancer',
    'breast-cancer-wisconsin',
    'breast-w',
    'buggyCrx',
    'bupa',
    'calendarDOW',
    'car',
    'car-evaluation',
    'cars',
    'cars1',
    'chess',
    'churn',
    'clean1',
    'clean2',
    'cleve',
    'cleveland',
    'cleveland-nominal',
    'cloud',
    'cmc',
    'coil2000',
    'colic',
    'collins',
    'confidence',
    'connect-4',
    'contraceptive',
    'corral',
    'credit-a',
    'credit-g',
    'crx',
    'dermatology',
    'diabetes',
    'dis',
    'dna',
    'ecoli',
    'fars',
    'flags',
    'flare',
    'german',
    'glass',
    'glass2',
    'haberman',
    'hayes-roth',
    'heart-c',
    'heart-h',
    'heart-statlog',
    'hepatitis',
    'horse-colic',
    'house-votes-84',
    'hungarian',
    'hypothyroid',
    'ionosphere',
    'iris',
    'irish',
    'kddcup',
    'kr-vs-kp',
    'krkopt',
    'labor',
    'led24',
    'led7',
    'letter',
    'liver-disorder',
    'lupus',
    'lymphography',
    'magic',
    'mfeat-factors',
    'mfeat-fourier',
    'mfeat-karhunen',
    'mfeat-morphological',
    'mfeat-pixel',
    'mfeat-zernike',
    'mnist',
    'mofn-3-7-10',
    'molecular-biology_promoters',
    'monk1',
    'monk2',
    'monk3',
    'movement_libras',
    'mushroom',
    'mux6',
    'new-thyroid',
    'nursery',
    'optdigits',
    'page-blocks',
    'parity5',
    'parity5+5',
    'pendigits',
    'phoneme',
    'pima',
    'poker',
    'postoperative-patient-data',
    'prnn_crabs',
    'prnn_fglass',
    'prnn_synth',
    'profb',
    'promoters',
    'ring',
    'saheart',
    'satimage',
    'schizo',
    'segmentation',
    'shuttle',
    'sleep',
    'solar-flare_1',
    'solar-flare_2',
    'sonar',
    'soybean',
    'spambase',
    'spect',
    'spectf',
    'splice',
    'tae',
    'texture',
    'threeOf9',
    'tic-tac-toe',
    'titanic',
    'tokyo1',
    'twonorm',
    'vehicle',
    'vote',
    'vowel',
    'waveform-21',
    'waveform-40',
    'wdbc',
    'wine-quality-red',
    'wine-quality-white',
    'wine-recognition',
    'xd6',
    'yeast'
]

def fetch_data(dataset_name, return_X_y=False, local_cache_dir=None):
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

    Returns
    ----------
    dataset: pd.DataFrame or (array-like, array-like)
        if return_X_y == False: A pandas DataFrame containing the fetched data set.
        if return_X_y == True: A tuple of NumPy arrays containing (features, labels)

    """
    if dataset_name not in dataset_names:
        raise ValueError('Data set not found in PMLB.')

    dataset_url = 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/{DATASET_NAME}/{DATASET_NAME}.csv.gz'.format(DATASET_NAME=dataset_name)

    if local_cache_dir is None:
        dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
    else:
        dataset_path = os.path.join(local_cache_dir, dataset_name) + '.csv.gz'

        # Use the local cache if the file already exists there
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, sep='\t', compression='gzip')
        # Download the data to the local cache if it is not already there
        else:
            dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
            dataset.to_csv(dataset_path, sep='\t', compression='gzip', index=False)

    if return_X_y:
        X = dataset.drop('class', axis=1).values
        y = dataset['class'].values
        return (X, y)
    else:
        return dataset
