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
