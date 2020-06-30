## Using PMLB in Python scripts

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
