# Penn Machine Learning Benchmarks

This repository contains the code and data for a large, curated set of benchmark datasets for evaluating and comparing supervised machine learning algorithms.
These data sets cover a broad range of applications, and include binary/multi-class classification problems and regression problems, as well as combinations of categorical, ordinal, and continuous features.

Please go to our [home page](https://epistasislab.github.io/pmlb/) to interactively browse the datasets, vignette, and contribution guide!

## Breaking changes in PMLB 1.0

*This repository has been restructured, and several dataset names have been changed!*

If you have an older version of PMLB, we highly recommend you upgrade it to v1.0 for updated URLs and names of datasets:

```
pip install pmlb --upgrade
```

## Datasets

Datasets are tracked with Git Large File Storage (LFS).
If you would like to clone the entire repository, please [install and set up Git LFS](https://git-lfs.github.com/) for your user account. 
Alternatively, you can download the `.zip` file from GitHub.

All data sets are stored in a common format:

* First row is the column names
* Each following row corresponds to one row of the data
* The target column is named `target`
* All columns are tab (`\t`) separated
* All files are compressed with `gzip` to conserve space

![Dataset_Sizes](datasets/dataset_sizes.svg)

The [complete table](pmlb/all_summary_stats.tsv) of dataset characteristics is also available for download.
Please note, in our documentation, a feature is considered:
* "binary" if it is of type integer and has 2 unique values (equivalent to pandas profiling's "boolean")
* "categorical" if it is of type integer and has *more than* 2 unique values (equivalent to pandas profiling's "categorical")
* "continuous" if it is of type float (equivalent to pandas profiling's "numeric").

## Python wrapper

For easy access to the benchmark data sets, we have provided a Python wrapper named `pmlb`. The wrapper can be installed on Python via `pip`:

```
pip install pmlb
```

and used in Python scripts as follows:

```python
from pmlb import fetch_data

# Returns a pandas DataFrame
adult_data = fetch_data('adult')
print(adult_data.describe())
```

The `fetch_data` function has two additional parameters:
* `return_X_y` (True/False): Whether to return the data in scikit-learn format, with the features and labels stored in separate NumPy arrays.
* `local_cache_dir` (string): The directory on your local machine to store the data files so you don't have to fetch them over the web again. By default, the wrapper does not use a local cache directory.

For example:

```python
from pmlb import fetch_data

# Returns NumPy arrays
adult_X, adult_y = fetch_data('adult', return_X_y=True, local_cache_dir='./')
print(adult_X)
print(adult_y)
```

You can also list all of the available data sets as follows:

```python
from pmlb import dataset_names

print(dataset_names)
```

Or if you only want a list of available classification or regression datasets:

```python
from pmlb import classification_dataset_names, regression_dataset_names

print(classification_dataset_names)
print('')
print(regression_dataset_names)
```

## Example usage: Compare two classification algorithms with PMLB

PMLB is designed to make it easy to benchmark machine learning algorithms against each other. Below is a Python code snippet showing the most basic way to use PMLB to compare two algorithms.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []

for classification_dataset in classification_dataset_names:
    X, y = fetch_data(classification_dataset, return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    logit = LogisticRegression()
    gnb = GaussianNB()

    logit.fit(train_X, train_y)
    gnb.fit(train_X, train_y)

    logit_test_scores.append(logit.score(test_X, test_y))
    gnb_test_scores.append(gnb.score(test_X, test_y))

sb.boxplot(data=[logit_test_scores, gnb_test_scores], notch=True)
plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB'])
plt.ylabel('Test Accuracy')
```

## Contributing

See our [Contributing Guide](https://epistasislab.github.io/pmlb/contributing.html). 
We're looking for help with documentation, and also appreciate new dataset and functionality contributions.

## Citing PMLB

If you use PMLB in a scientific publication, please consider citing one of the following papers:

Joseph D. Romano, Le, Trang T., William La Cava, John T. Gregg, Daniel J. Goldberg, Praneel Chakraborty, Natasha L. Ray, Daniel Himmelstein, Weixuan Fu, and Jason H. Moore.
[PMLB v1.0: an open source dataset collection for benchmarking machine learning methods](https://arxiv.org/abs/2012.00058).
_arXiv preprint arXiv:2012.00058_ (2020).

```bibtex
@article{romano2021pmlb,
  title={PMLB v1.0: an open source dataset collection for benchmarking machine learning methods},
  author={Romano, Joseph D and Le, Trang T and La Cava, William and Gregg, John T and Goldberg, Daniel J and Chakraborty, Praneel and Ray, Natasha L and Himmelstein, Daniel and Fu, Weixuan and Moore, Jason H},
  journal={arXiv preprint arXiv:2012.00058v2},
  year={2021}
}
```

Randal S. Olson, William La Cava, Patryk Orzechowski, Ryan J. Urbanowicz, and Jason H. Moore (2017). [PMLB: a large benchmark suite for machine learning evaluation and comparison](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-017-0154-4). *BioData Mining* **10**, page 36.

BibTeX entry:

```bibtex
@article{Olson2017PMLB,
    author="Olson, Randal S. and La Cava, William and Orzechowski, Patryk and Urbanowicz, Ryan J. and Moore, Jason H.",
    title="PMLB: a large benchmark suite for machine learning evaluation and comparison",
    journal="BioData Mining",
    year="2017",
    month="Dec",
    day="11",
    volume="10",
    number="1",
    pages="36",
    issn="1756-0381",
    doi="10.1186/s13040-017-0154-4",
    url="https://doi.org/10.1186/s13040-017-0154-4"
}
```

## Support for PMLB

PMLB was developed in the [Computational Genetics Lab](http://epistasis.org/) at the [University of Pennsylvania](https://www.upenn.edu/) with funding from the [NIH](http://www.nih.gov/) under grant AI117694, LM010098 and LM012601. We are incredibly grateful for the support of the NIH and the University of Pennsylvania during the development of this project.
