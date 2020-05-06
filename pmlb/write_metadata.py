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

import glob, os
import csv
import pandas as pd
from collections import Counter
from pmlb import fetch_data, dataset_names
import pdb
def imbalance_metrics(data):
    """ Computes imbalance metric for a given dataset. 
    Imbalance metric is equal to 0 when a dataset is perfectly balanced (i.e. number of in each class is exact).
    :param data : pandas.DataFrame 
        A dataset in a panda's data frame
    :returns int 
        A value of imbalance metric, where zero means that the dataset is perfectly balanced and the higher the value, the more imbalanced the dataset.
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
    #worst case scenario: all but 1 examplars in 1st class, the remaining one in 2nd class
    worst_case=(num_classes-1)*pow(1/num_classes,2) + pow(1-1/num_classes,2)
    return (num_classes,imb/worst_case)



def determine_endpoint_type(features):
    """ Determines the type of an endpoint
    :param features: pandas.DataFrame
        A dataset in a panda's data frame
    :returns string
        string with a name of a dataset
    """    
    counter={k.name: v for k, v in features.columns.to_series().groupby(features.dtypes).groups.items()}
    if(len(features.groupby('class').apply(list))==2):
        return('binary')
    if ('float64' in counter):
        return ('float')
    return ('integer')
    #binary=len(set(features.loc[:, (features<=1).all(axis=0)].columns.values) \
    #           & set(features.loc[:, (features>=0).all(axis=0)].columns.values) \
    #           & set(counter['int64']))
    

def count_features_type(features):
    """ Counts three different types of features (float, integer, binary).
    :param features: pandas.DataFrame
        A dataset in a panda's data frame
    :returns a tuple (binary, integer, float)
    """    
    counter={k.name: v for k, v in features.columns.to_series().groupby(features.dtypes)}
    binary=0
    if ('int64' in counter):
        binary=len(set(features.loc[:, (features<=1).all(axis=0)].columns.values) 
                & set(features.loc[:, (features>=0).all(axis=0)].columns.values) 
                & set(counter['int64']))    
    return (binary,len(counter['int64'])-binary if 'int64' in counter else 0,len(counter['float64']) if 'float64' in counter else 0)

def get_types(df):
    types=[]
#    pdb.set_trace()
    for cols in df.columns:
        if df[cols].dtype=='float64':
            types.append('continous')
        elif df[cols].dtype=='int64':
            if len(set(df[cols].values)) <= 2:
                types.append('binary')
            else:
                types.append('discrete')
    return types
        
def generate_description(dataset_name, local_cache_dir=None):
    """Generates desription for a given dataset in its README.md file in a dataset local_cache_dir file.
    
    :param dataset_name: str
        The name of the data set to load from PMLB.
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    """
    
    assert (local_cache_dir!=None)
    readme_file = open(os.path.join(local_cache_dir,'datasets',dataset_name,'README.md'), 'wt')
    try:
        df = fetch_data(dataset_name)
        fnames = [col for col in df.columns if col!='class']
        #determine all required values
        types = get_types(df.ix[:, df.columns != 'class'])
        feat=count_features_type(df.ix[:, df.columns != 'class'])
        endpoint=determine_endpoint_type(df.ix[:, df.columns == 'class'])
        mse=imbalance_metrics(df['class'].tolist())
        #proceed with writing
        readme_file.write('# %s\n\n' % dataset_name)
        readme_file.write('## Summary Stats\n\n')
        readme_file.write('#instances: %s\n\n' % str(len(df.axes[0])))
        readme_file.write("#features: %s\n\n" % str(len(df.axes[1])-1))
        readme_file.write("  #binary_features: %s\n\n" % feat[0])
        readme_file.write("  #integer_features: %s\n\n" % feat[1])
        readme_file.write("  #float_features: %s\n\n" % feat[2])
        readme_file.write("Endpoint type: %s\n\n" % endpoint)
        readme_file.write("#Classes: %s\n\n" % int(mse[0]))
        readme_file.write("Imbalance metric: %s\n\n" % mse[1])
        readme_file.write('## Feature Types\n\n %s\n\n' % '\n\n'.join([f + ':' + t for f,t in
                                                              zip(fnames,types)]))

    except IOError as err:
        print(err)
    finally:
        readme_file.close()



def generate_readmes(local_cache_dir=None):
    """Generates a summary report for all dataset in PMLB
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
    """
    for dataset in dataset_names:
        print("Dataset:", dataset)
        generate_description(dataset,local_cache_dir)

def generate_pmlb_summary(local_cache_dir=None):
    """Generates a summary report for all dataset in PMLB
    :param local_cache_dir: str (required)
        The directory on your local machine to store the data files.
    """
    report_filename = open(os.path.join(local_cache_dir, 'report.csv'), 'wt')
    assert (local_cache_dir!=None)
    try:
        writer = csv.writer(report_filename, delimiter='\t')
        writer.writerow(['Dataset','#instances','#features','#binary_features','#integer_features','#float_features',\
                     'Endpoint_type','#classes','Imbalance_metric'])

        for dataset in dataset_names:
            df=fetch_data(dataset)
            print( "Dataset:", dataset)
            assert 'class' in df.columns, "no class column"
            #removing class column
            print( "SIZE: "+ str(len(df.axes[0]))+ " x " + str(len(df.axes[1])-1))
            feat=count_features_type(df.ix[:, df.columns != 'class'])
            endpoint=determine_endpoint_type(df.ix[:, df.columns == 'class'])
            mse=imbalance_metrics(df['class'].tolist())
            #writer.writerow([file,str(len(df.axes[0])),str(len(df.axes[1])-1),feat[0],feat[1],feat[2],endpoint,mse[0],mse[1],mse[2]])
            writer.writerow([dataset,str(len(df.axes[0])),str(len(df.axes[1])-1),feat[0],feat[1],feat[2],endpoint,int(mse[0]),mse[1]])
    finally:
        report_filename.close()

if __name__ =='__main__':
    local_dir = '../'
    
    for d in dataset_names:
        print(d,'...')
        generate_description(d,local_cache_dir=local_dir)
