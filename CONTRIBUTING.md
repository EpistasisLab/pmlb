# Introduction

### Thanks!

First off, thank you for considering contributing to PMLB.
We want this to be the easiest resource to use for benchmarking machine learning algorithms on many datasets.
This is a community effort, and we rely on help from users like you.


### Why you should read this

Making a really easy-to-use benchmark resource also means being diligent about how contributions are made.
Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project.
In return, we will reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

### Types of contributions

The main contribution our project needs at the moment is help identifying, sourcing and documenting the datasets that currently don't have that information.
We would also consider dataset contributions that meet the format specifications of PMLB.
We're open to other ideas (improving documentation, writing tutorials, etc.) that you may want to make.  

# Ground Rules
### Be kind.
We will, too.

### Responsibilities
 * For sourcing/documentation of existing datasets, make sure your pull request follows our [source guidelines](#contributing-source-information)
 * For new datasets, make sure your pull request follows our [new dataset guidelines](#contributing-a-new-dataset)
 * Make sure your changes pass the tests. To check, please run the following (note, you must have the `nose` package installed within your dev environment for this to work):
    ```
    nosetests -s -v
    ```
 * Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
 * Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/) as an example.

# Your First Contribution
Help people who are new to your project understand where they can be most helpful.
This is also a good time to let people know if you follow a label convention for flagging beginner issues.

Unsure where to begin contributing to PMLB? You can start by looking through issues for help-wanted tags.

If you haven't contributed to open source code before, check out these friendly tutorials:
 - http://makeapullrequest.com/
 - http://www.firsttimersonly.com/
 - [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

Those guides should tell you everything you need to start out!

# Getting started
### How to submit a contribution

1. Create your own fork of this repository
2. Make the changes in your fork
3. If you think the project would benefit from these changes:
    * Make sure you have followed the guidelines above.
    * Submit a pull request.

# How to report a bug

When filing an issue, please make sure to answer these five questions:

1. What version of PMLB are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

# Contributing Source Information

Relevant issues: [#13](https://github.com/EpistasisLab/penn-ml-benchmarks/issues/13), [#22](https://github.com/EpistasisLab/penn-ml-benchmarks/issues/22).

As part of our [PMLB 2.0 Project](https://github.com/EpistasisLab/penn-ml-benchmarks/projects/1), each dataset is getting revamped with a README file, a metadata.yaml file, and a summary statistics file.
We need help doing this for each dataset!

### How to submit a contribution

1. Verify the source for the dataset.
    - Often the place to start is an internet search of the dataset name. 
    Most datasets can be found in [OpenML](https://www.openml.org/), [the UC Irvine ML repository](http://archive.ics.uci.edu/ml/index.php), or [Kaggle](www.kaggle.com). 
    - Ideally, you will be able to verify the source is correct by downloading the source dataset, applying some simple transformations like normalization, and doing a checksum that validates the two datasets are now equivalent. Something like below.

    ```python
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df_pmlb = fetch_data('example')
    df_source = pd.read_csv('potential_source_dataset.csv')

    # apply some changes to df_source
    df_source['x1'] = LabelEncoder().fit_transform(df_source['x1'])
    
    import hashlib
    rowhashes_pmlb = hash_pandas_object(df_pmlb).values 
    hash_pmlb = hashlib.sha256(rowhashes_pmlb).hexdigest()
    rowhashes_source = hash_pandas_object(df_source).values 
    hash_source = hashlib.sha256(rowhashes_source).hexdigest()
    
    # verify hashes match
    print(hash_pmlb == hash_source)
    ```

2. Update the information on the dataset's metadata.yaml file. 
Refer to the [metadata template file](metadata_template.yaml) or [wine_quality_red](datasets/wine_quality_red/metadata.yaml) as an example.
3. Issue a pull request for your changes. In the pull request, document how you verified the source of the dataset, for example, by performing a checksum on the data. Include any information to help us independently check that what you have added is accurate.

# Contributing a new dataset

New datasets should follow these guidelines:

 - Each sample/observation forms a row of the dataset.
 - Each feature/variable forms a column of the dataset.
 - The dependent variable, i.e., outcome/target, should be labelled `'target'`.
 - If the task is classification, the dependent variable must be encoded with numeric, contiguous labels in [0, 1, .. k], where there are k classes in the data.
 - Column headers are feature/variable names and `'target'`.
 - Any `'sample_id'` or `'row_id'` column should be *excluded*.
 - The data should be tab-delimited and in `.tsv.gz` format.
 - The dataset should be in the correct folder; i.e., for a classification dataset, `penn-ml-benchmarks/datasets/classification/your_dataset/`
 - A metadata.yaml file must be provided with all required fields filled in. Please follow the template guidelines.
 - The dataset should not exceed 50 MB.  

Note that any pull requests for new dataset contributions will not be accepted if these guidelines are not met.

