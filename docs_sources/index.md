# Penn Machine Learning Benchmarks

This repository contains the code and data for a large, curated set of benchmark datasets for evaluating and comparing supervised machine learning algorithms. These data sets cover a broad range of applications, and include binary/multi-class classification problems and regression problems, as well as combinations of categorical, ordinal, and continuous features. There are no missing values in these data sets.

Check the `datasets` directory for information about the individual data sets.

## Datasets Summary

<center>
<img src="https://raw.githubusercontent.com/EpistasisLab/penn-ml-benchmarks/PMLB2.0/datasets/dataset_sizes.svg" />
</center>

## Data set format

All data sets are stored in a common format:

* First row is the column names
* Each following row corresponds to one row of the data
* The target column is named `target`
* All columns are tab (`\t`) separated
* All files are compressed with `gzip` to conserve space


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
