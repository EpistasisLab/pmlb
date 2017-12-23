# Penn Machine Learning Benchmarks

This repository contains the code and data for a large, curated set of benchmarks for evaluating supervised machine learning algorithms. These data sets cover a broad range of applications, and include binary and multi-class problems, as well as combinations of categorical, ordinal, and continuous features. There are no missing values in these data sets.

Check the `datasets` directory for information about the individual data sets.

## Data set format

All data sets are stored in a common format:

* First row is the column names
* Each following row corresponds to one row of the data
* The target column is named `class`
* All columns are tab (`\t`) separated
* All files are compressed with `gzip` to conserve space

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

You can also list all of the available data sets as follows:

```python
from pmlb import dataset_names

print(dataset_names)
```

## Citing PMLB

If you use PMLB in a scientific publication, please consider citing the following paper:

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

PMLB was developed in the [Computational Genetics Lab](http://epistasis.org/) at the [University of Pennsylvania](https://www.upenn.edu/) with funding from the [NIH](http://www.nih.gov/) under grant R01 AI117694. We are incredibly grateful for the support of the NIH and the University of Pennsylvania during the development of this project.
