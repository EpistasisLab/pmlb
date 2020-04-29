# Introduction

### Thanks!

First off, thank you for considering contributing to PMLB. 
We want this to be the easiest resource to use for benchmarking machine learning algorithms on many datasets. 
This is a community effort, and we rely on help from users like you.


### Why you should read this

Making a really easy-to-use benchmark resource mealso also ans being diligent about how contributions are made. 
Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. 
In return, we will reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

### Types of contributions

The main contribution our project needs at the moment is help identifying, sourcing and documenting the datasets that currently don't have that information. 
We would also consider dataset contributions that meet the format specifications of PMLB. 
We're open to other ideas (improving documentation, writing tutorials, etc.) that you may want to make.  

# Ground Rules
### Be kind
We will too. 

 Responsibilities
 * For sourcing/documentation of existing datasets, make sure your pull request follows our guidelines: *add link* 
 * For new datasets, make sure your pull request follows our new dataset guidelines: *add link*
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

1. Create your own fork of the code
2. Do the changes in your fork
3. If you like the change and think the project could use it:
    * Be sure you have followed the guidelines above.
    * Send a pull request.

# How to report a bug

When filing an issue, make sure to answer these five questions:

1. What version of PMLB are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

# Contributing Source Information

Relevant issues: #13, #22

As part of our [PMLB 2.0 Project](https://github.com/EpistasisLab/penn-ml-benchmarks/projects/1), each dataset is getting revamped with a README file, a metadata.yaml file, and a summary statistics file. 
We need help doing this for each dataset! 

### How to submit a contribution

1. Verify the source for the dataset. 
2. Fill in the missing information on the dataset's associated metadata.yaml file. 
3. Issue a pull request for your changes. In the pull request, document how you verified the source of the dataset, for example by performing a checksum on the data. Include any information to help us independently check that what you have added is accurate. 

# Contributing a new dataset

New datasets should follow these guidelines:

 - Samples of the dataset should be in rows, and features in columns, with a header file. 
 - The data should be tab-delimited and in `.tsv.gz` format. 
 - The dataset should be in the correct folder; i.e. for a classification dataset, `penn-ml-benchmarks/datasets/classification/your_dataset/'
 - The dependent variable, i.e. outcome/target, should be labelled `target'. 
 - A metadata.yaml file must be provided with all fields filled in. Please follow the template guideliness. 
 - The dataset should not exceed 50 MB.  

Note that any pull requests for new dataset contributions will not be accepted if these guidelines are not demonstrated to be met. 
