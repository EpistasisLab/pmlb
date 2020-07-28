from pmlb import dataset_names
import pathlib

'''Writes the Browse datasets page.'''

if __name__ =='__main__':
    browse_md = '''\
# Browse datasets

Please click on a dataset to access its [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/) report.
*Note*: if a dataset has more than 20 predictors, only 20 of them are selected to show in the report.
'''
    for dataset in dataset_names:
        path = pathlib.Path(f'docs_sources/browse-datasets.md')

        browse_md = browse_md + (f'\n- [{dataset}](../profile/{dataset}.html)')
        path.write_text(browse_md)