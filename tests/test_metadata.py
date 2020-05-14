import glob
import yaml
import pandas as pd
import pdb
yaml_files = glob.glob('datasets/*/*/*.yaml')


def test_all_yaml_files():
    "Check basic information in yaml files."
    for yf in yaml_files:
        folder_name = yf.split("/")[-2]

        with open(yf) as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        # check if folder name is correct
        assert metadata['dataset'] == folder_name
        assert metadata['task'] in ["classification", "regression"]


def test_dataset():
    "Check basic information in dataset files."
    for yf in yaml_files:
        folder_name = yf.split("/")[-2]
        file_name = yf.split('metadata.yaml')[0]+folder_name+'.tsv.gz'
        with open(yf) as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        dataset = pd.read_csv(file_name, sep='\t', compression='gzip')
        cols = set(list(dataset.columns))

        exported_cols = set([str(s['name']) for s in metadata['features']]
                +['target'])

        if exported_cols != cols:
            print('filename:',file_name)
            print('metadata:',metadata)
            print('exported_cols:',exported_cols)
            print('cols:',cols)
            print('set diff:',cols.difference(exported_cols))

        assert exported_cols == cols
