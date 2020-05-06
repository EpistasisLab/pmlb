import glob
import yaml
import pandas as pd
yaml_files = glob.glob('datasets/*/*.yaml')


def test_all_yaml_files():
    "Check basic information in yaml files."
    for yf in yaml_files:
        folder_name = yf.split("/")[1]
        file_name = "datasets/{0}/{0}.tsv.gz".format(folder_name)

        with open(yf) as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        # check if folder name is correct
        assert metadata['dataset'] == folder_name
        assert metadata['task'] in ["classification", "regression"]


def test_dataset():
    "Check basic information in dataset files."
    for yf in yaml_files:
        folder_name = yf.split("/")[1]
        file_name = "datasets/{0}/{0}.tsv.gz".format(folder_name)
        with open(yf) as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        dataset = pd.read_csv(file_name, sep='\t', compression='gzip')
        cols = sorted(list(dataset.columns))

        features = [s['name'] for s in metadata['features']]
        exported_cols = sorted(features + ['target'])

        assert exported_cols == cols
