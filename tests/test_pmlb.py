import pandas as pd
import numpy as np
from pmlb import fetch_data
from pmlb.pmlb import get_dataset_url, get_updated_datasets
from nose.tools import assert_raises
from tempfile import mkdtemp
from shutil import rmtree
from os import path
GITHUB_URL = ("https://github.com/EpistasisLab/pmlb/"
"raw/master/datasets")
suffix = '.tsv.gz'

def test_fetch_data_1():
    """Test fetch_data can fetch data from GitHub."""

    mushroom = fetch_data('mushroom')
    assert not mushroom.empty
    assert not mushroom.isnull().values.any()


def test_fetch_data_2():
    """Test fetch_data can fetch data from local cache."""

    mushroom = fetch_data('mushroom', local_cache_dir="datasets/")
    assert not mushroom.empty

def test_fetch_data_3():
    """Test fetch_data can not fetch data with incorrect dataset name."""

    assert_raises(ValueError, fetch_data, "musroom")

def test_fetch_data_4():
    """Test fetch_data can not fetch data from local cache
     with incorrect dataset name."""

    assert_raises(ValueError, fetch_data, "musroom", local_cache_dir="datasets/")

def test_fetch_data_5():
    """Test fetch_data can fetch data from local cache
     but the dataset is not available in local cache"""

    cachedir = mkdtemp()
    dataset_name = 'mushroom'
    mushroom = fetch_data(dataset_name, local_cache_dir=cachedir)
    out_cache_data = path.join(cachedir, dataset_name,
                                dataset_name+'.tsv.gz')
    assert not mushroom.empty
    assert path.isfile(out_cache_data)
    rmtree(cachedir)

def test_fetch_data_6():
    """Test fetch_data can fetch data from GitHub with return_X_y."""
    X, y = fetch_data('mushroom', return_X_y=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_get_dataset_url_1():
    """Test get_dataset_url can fetch data from GitHub."""
    dataset_name = 'mushroom'
    dataset_url = get_dataset_url(GITHUB_URL,
                                    dataset_name, suffix)
    expected_url = ("https://github.com/EpistasisLab/pmlb"
    "/raw/master/datasets/mushroom/mushroom.tsv.gz")

    assert dataset_url == expected_url

def test_get_dataset_url_2():
    """Test get_dataset_url can not fetch data from GitHub with
    incorrect dataset name."""
    dataset_name = 'mushrom'
    assert_raises(ValueError, get_dataset_url,
                                GITHUB_URL,
                                dataset_name,
                                suffix)

def test_get_updated_datasets():
    """Test get_updated_datasets can run without error."""
    updated_datasets = get_updated_datasets()
    print(updated_datasets)
