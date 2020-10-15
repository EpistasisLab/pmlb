---
title: 'PMLB v1.0: an open source dataset collection for benchmarking machine learning methods'
tags:
  - machine learning
  - python
  - rstats
  - benchmark
authors:
  - name: Trang T. Le
    orcid: 0000-0003-3737-6565
    affiliation: 1
  - name: William La Cava
    orcid: 0000-0002-1332-2960
    affiliation: 1
  - name: Joseph D. Romano
    orcid: 0000-0002-7999-4399
    affiliation: "1, 2"
  - name: John T. Greg
    orcid: 0000-0002-2619-3440
    affiliation: 1
  - name: Daniel J. Goldberg
    orcid: 0000-0003-4173-9867
    affiliation: 3
  - name: Praneel Chakraborty
    orcid: 0000-0001-9586-0721
    affiliations: "4, 5"
  - name: Natasha L. Ray
    orcid: 0000-0001-6883-4624
    affiliation: 6
  - name: Daniel Himmelstein
    orcid: 0000-0002-3012-7446
    affiliations: "7, 8"
  - name: Weixuan Fu
    orcid: 0000-0002-6434-5468
    affiliation: 9
  - name: Jason H. Moore^[corresponding author]
    orcid: 0000-0002-5015-1099
    affiliation: 9
affiliations:
 - name: Department of Biostatistics, Epidemiology and Informatics, University of Pennsylvania, Philadelphia, PA 19104
   index: 1
 - name: Institution Name
   index: 2
 - name: Department of Computer Science & Engineering, Washington University in St. Louis, St. Louis, MO 63130
   index: 3
 - name: School of Arts and Sciences, University of Pennsylvania, Philadelphia, PA 19104
   index: 4
 - name: Wharton School, University of Pennsylvania, Philadelphia, PA 19104
   index: 5
 - name: Princeton Day School, Princeton, NJ 08540
   index: 6
 - name: Related Sciences
   index: 7
 - name: Department of Systems Pharmacology & Translational Therapeutics, University of Pennsylvania, Philadelphia, PA 19104
   index: 8
 - name: Institute for Biomedical Informatics, University of Pennsylvania, Philadelphia, PA 19104
   index: 9

date: 17 October 2020
bibliography: paper.bib

---

# Summary

PMLB (Penn Machine Learning Benchmark) is an open source data repository containing a curated collection of datasets for evaluating and comparing machine learning (ML) algorithms.
Compiled from a broad range of existing ML benchmark collections, PMLB synthesizes and standardizes hundreds of publicly available datasets from diverse sources such as the [UCI ML repository](http://archive.ics.uci.edu/ml) and [OpenML](www.openml.org) [@Vanschoren2014], enabling systematic assessment of different ML methods.
These datasets cover a range of applications, from binary/multi-class classification to regression problems with combinations of categorical and continuous features.
PMLB has both a Python interface (`pmlb`) and an R interface (`pmlbr`), both with detailed documentation that allows the user to access cleaned and formatted datasets using a single function call (`fetch_data`).
PMLB also provides a comprehensive description of each dataset and advanced functions to explore the dataset space such as `nearest_datasets` and `filter_datasets`, which allow for smoother user experience and handling of data.
The resource is designed to facilitate open source contributions in the form of datasets as well as improvements to curation.

# Statement of need

Benchmarking is a standard practice to illustrate the strengths and weaknesses of algorithms with regards to different problem characteristics.
In ML, benchmarking often involves assessing the performance of specific ML models &mdash; namely, how well they predict labels for new samples (supervised learning) or how well they organize and/or represent data with no pre-existing labels (unsupervised learning).
The extent to which ML methods achieve these aims is typically evaluated over a group of benchmark datasets [@Stallkamp2012; @Caruana2006].
PMLB was designed to provide a suite of such datasets with uniform formatting, as well as the framework for conducting automatic evaluation of the different algorithms.

The original release of PMLB (v0.2) [@Olson2017] received positive feedback from the ML community, reflecting the pressing need for a collection of standardized datasets to evaluate models without intensive preprocessing and dataset curation.
As the repository becomes more widely used, community members have requested new features such as additional information about the datasets, as well as new functions to select datasets given specific criteria.
In this paper, we review the original functionality and present new enhancements that facilitate a fluid interaction with the repository, both from the perspective of database contributors and end-users.

# Differentiating attributes

## New datasets with rich metadata

Since its previous major release, v0.2 [@Olson2017], we have made substantial improvements in the collection of new datasets as well as other helpful supporting features.
PMLB now has a new repository structure that includes benchmark datasets for regression problems (\autoref{fig:home-chart}).
To fulfill [requests made by several users](https://github.com/EpistasisLab/pmlb/issues/13), each dataset also includes a `metadata.yaml` file that contains general descriptive information about the dataset itself (an example can be viewed [here](https://github.com/EpistasisLab/pmlb/blob/master/datasets/molecular_biology_promoters/metadata.yaml)).
Specifically, for each dataset, the metadata file includes a web address to the original source of the dataset, a text description of the dataset's purpose, the publication associated with the dataset generation, the type of learning problem it was designed for (i.e., classification or regression), keywords (e.g., "simulation", "ecological", "bioinformatics"), and a description of individual features and their coding schema (e.g., ‘non-promoter’= 0,  ‘promoter’= 1).
Metadata files are supported by a standardized format that is formalized using JSON-Schema (version `draft-07`) [@Pezoa2016] &mdash; upcoming releases of PMLB will include automated validation of datasets and metadata files to further improve ease of contribution and data accuracy.

![Characteristics of datasets in the PMLB collection.\label{fig:home-chart}](pmlb-home-chart.png)

A number of open source contributors have been invaluable in providing manually-curated metadata.
In addition, contributors' careful examination have led to important bug fixes, such as a [correction to the target column](https://github.com/EpistasisLab/pmlb/issues/54) in the [bupa](https://github.com/EpistasisLab/pmlb/tree/master/datasets/bupa) dataset.

## User-friendly interfaces

On PMLB's [home page](https://epistasislab.github.io/pmlb/), users can now browse, sort, filter, and search datasets from a lookup table of datasets with summary statistics (\autoref{fig:home-tab}).
To select datasets with numerical values for specific metadata characteristics (e.g., number of observations, number of features, class balance, etc.), one can type ranges in the box at the bottom of each numeric column in the format `low ... high`.
For example, if the user wants to view all classification datasets with 80 to 100 observations, they would select `classification` at the bottom of the `Task` column, and type `80 ... 100` at the bottom of the `n_observations` column.
The `CSV` button allows the user to download the table's contents with any active filters applied.

![Dataset summary statistics table, with advanced searching, filtering, and sorting features.\label{fig:home-tab}](pmlb-home-tab.png)

On the website, we have also published a concise [contribution guide](https://epistasislab.github.io/pmlb/contributing.html) with step-by-step instructions on how to add new datasets, submit edits for existing datasets, or improve the provided Python or R code.
When a new dataset is added, summary statistics (e.g., number of observations, number of classes, etc.) are automatically computed, a profiling report is generated (see below), a corresponding metadata template is added to the dataset folder, and PMLB's list of available dataset names is updated.
Other checks included in the continuous integration workflow help reduce the amount of work required from both contributors and code reviewers.

In addition to the Python interface for PMLB, we have included an [R library](https://github.com/EpistasisLab/pmlbr) that originated from a [separate repository](https://github.com/makeyourownmaker/pmlblite) that is currently unmaintained.
However, because its source code was released under the [GNU General Public License, version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html), we were able to adapt the code to make it compatible with the new repository structure in this release and offer additional functionality.
The R library also includes a number of detailed "vignette" documents to help new users learn how to use the software.

PMLB now includes original data rows with missing data (i.e., NA). 
The new version of PMLB also allows the user to select datasets most similar to one of their own using the `nearest_datasets` function. 
Here, the similarity between datasets is configurable to any number of metadata characteristics (e.g., number of samples, number of features, number of target classes, etc.).
This functionality is helpful for users who wish to find PMLB datasets with similar characteristics to their own in order to test or optimize methods (e.g., hyperparameter tuning) for their desired problem without the risk of over-fitting to their dataset. 

API reference guides that detail all user-facing functions and variables in PMLB's [Python](https://epistasislab.github.io/pmlb/python-ref.html) and [R](https://epistasislab.github.io/pmlb/r-ref.html) libraries is included on the PMLB website.

## Pandas profiling reports 

For each dataset, we use [`pandas-profiling`](https://pandas-profiling.github.io/pandas-profiling/) to generate summary statistic reports.
In addition to the descriptive statistics provided by the commonly-used `pandas.describe` (Python) [@McKinney2010] or `skimr::skim` (R) functions, `pandas-profiling` gives a more extensive exploration of the dataset, including correlation structure within the dataset and flagging of duplicate samples.
Browsing a report allows users and contributors to easily assess dataset quality and make any necessary changes.
For example, if a feature is flagged by `pandas-profiling` as having a single value replicated in all samples, it is likely that this feature is uninformative for ML analysis and should be removed from the dataset.

The profiling reports can be accessed by clicking on the dataset name in the interactive data table or the data point in the interactive chart on the PMLB website.
Alternatively, all reports can be viewed on the repository's [gh-pages](https://github.com/EpistasisLab/pmlb/tree/gh-pages/profile) branch, or generated manually by users on their local computing resources.

## Space efficiency

We have significantly reduced the size of the PMLB source repository by using [Git Large File Storage (LFS)](https://git-lfs.github.com/) to efficiently track changes in large database source files [@PerezRiverol2016].
Users who would like to interact with the entire repository (including the complete database sources) locally can do so by either [installing Git LFS](https://git-lfs.github.com/) and cloning the PMLB repository, or by downloading a ZIP archive of [the repository](https://github.com/EpistasisLab/pmlb) from GitHub in a web browser.

# References
