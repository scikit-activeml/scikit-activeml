# Developer Guide

**Scikit-ActiveML** is a library that implements the most important query strategies of active learning. It is built upon the well-known machine learning framework [scikit-learn](https://scikit-learn.org/stable/).

## Introduction

### Getting Help

If you have any questions, please reach out to other developers via the following channels:

- [Github Issues](https://github.com/scikit-activeml/scikit-activeml/issues)

### Roadmap

Our Roadmap is summarized in the issue [Upcoming Features](https://github.com/scikit-activeml/scikit-activeml/issues/145).

## Get Started

Before you can contribute to this project, you might execute the following steps.

### Setup Development Environment

There are several ways to create a local Python environment, such as [virtualenv](https://www.google.com/search?client=safari&rls=en&q=virtualenv&ie=UTF-8&oe=UTF-8), [pipenv](https://pipenv.pypa.io/en/latest/), [miniconda](https://docs.conda.io/en/latest/miniconda.html), etc. One possible workflow is to install `miniconda` and use it to create a Python environment. And use `pip` to install packages in this environment.

#### Example With miniconda

Create a new Python environment named scikit-activeml:

```bash
conda create -n scikit-activeml
```

To be sure that the correct env is active:

```bash
conda activate scikit-activeml
```

Then install `pip`:

```bash
conda install pip
```

### Install Dependencies

Now we can install some required dependencies for scikit-activeml, which are defined in the `requirements.txt` file.

```bash
# Make sure your scikit-activeml python env is active!
cd <project-root>
pip install -r requirements.txt
pip install -r requirements_extra.txt
```

After the pip installation was successful, we have to install `pandoc` and `ghostscript` if it is not already installed.

#### Example with MacOS (Homebrew)

```bash
brew install pandoc
```

```bash
brew install ghostscript
```

## Contributing Code

### General Coding Conventions

As this library conforms to the convention of [scikit-learn](https://scikit-learn.org/stable/developers/develop.html#coding-guidelines), the code should conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/) Style Guide for Python Code. For linting, the use of [flake8](https://flake8.pycqa.org/en/latest/) is recommended.

### Example for C3 (Code Contribution Cycle) and Pull Requests

1. Fork the repository and clone your fork to your local machine:

```bash
git clone https://github.com/scikit-activeml/scikit-activeml.git
```

1. Create a branch for the changes from the `master` branch:

> Make sure that you create your branch from `master`.

> #TODO: Wollen wir hier the labels benutzen?

```bash
# Example: feature/187-adds-docker
# Example: bug-fix/187-fixes-docker-issue
# Example: documentation/187-adds-classifier-section
git checkout -b <feature, bug-fix, documentation>/<issue-no>-<short description>
```

3. After you have finished implementing the feature, make sure that all the tests pass. The tests can be run as

```bash
$ pytest
```

Make sure, you covered all lines by tests.

```bash
$ pytest --cov=./skactiveml
```

4. Commit and push the changes.

> TODO: Bitte beschreiben, wie das als Pull request bei uns auftaucht.

> TODO: Bitte hier noch schreiben, dass wir über alle contributions froh sind. Kurzer Satz dazu reicht.

```bash
$ git add modified_files
$ git commit -m "<commit-message>"
$ git push
```

## Query Strategies

> TODO-ALL@MMUEJDE: Einleitetext mit Übersicht der allgmeinen Struktur (Klassen, Methoden) und kurze Erklärung, dass jedes Szenario der Einfachheit wegen gesondert beschrieben wird. Evtl. einbringen eines UML Diagramms (mit/ohne Parameter/Attribute je nach Übersichtlichkeit).

### Pool-based Query Strategies

> TODO-ALL@MMUEJDE: Bitte Vorschlag machen, wie man die einzelnen Beschreibungen visuell besser darstellen kann (z.B. Einleitetext mit Sätzen. Methoden als Anstriche und Parameter/Attribute als Unterpunkte, evtl Unterüberschriften-einheitlich für jedes Szenario).

- All query strategies are stored in a file `skactiveml/pool/_query_strategy.py`

- Every class inherits from `SingleAnnotatorPoolBasedQueryStrategy`

- The class must implement the `__init__` function for initialization and a `query` function

- For typical class parameters we use standard names:

  - `prior` (Prior probabilities for the distribution of probabilistic strategies)

  - `random_state` (Number or np.random.RandomState like sklearn)

  - `method` (String for classes that implement multiple methods)

  - `cost_matrix` (Cost matrix defining the cost of predicting instances wrong)

- Typical parameters of the query function are:

| Parameter | Description |
| --- | --- |
| `X_cand` | Set of candidate instances, inherited from `SingleAnnotatorPoolBasedQueryStrategy` |
| `clf` | The classifier used by the strategy |
| `X` | Set of labeled and unlabeled instances |
| `y` | (unknown) labels of `X` |
| `sample_weight` | Weights of training samples in `X` |
| `sample_weight_cand` | Weights of samples in `X_cand` |
| `batch_size` | Number of instances for batch querying, inherited from `SingleAnnotatorPoolBasedQueryStrategy` |
| `return_utilities` | Inherited from `SingleAnnotatorPoolBasedQueryStrategy` |

- `X_cand` (Set of candidate instances, inherited from `SingleAnnotatorPoolBasedQueryStrategy`)
- `clf` (The classifier used by the strategy)
- `X` (Set of labeled and unlabeled instances)
- `y` ((unknown) labels of `X`)
- `sample_weight` (Weights of training samples in `X`)
- `sample_weight_cand` (Weights of samples in `X_cand`)
- `batch_size` (Number of instances for batch querying, inherited from `SingleAnnotatorPoolBasedQueryStrategy`)
- `return_utilities` (Inherited from `SingleAnnotatorPoolBasedQueryStrategy`)

- The `query` function returns:

  - `query_indices` (Indices of the best instances)
  - `utilities` (Utilities of all candidate instances, only if `return_utilities` is `True`)

- General advice for the query code:

  - use `self._validate_data` function (Is implemented in superclass)
  - check the input `X` and `y` only once
  - fit classifier if it is not yet fitted (May use `fit_if_not_fitted` form utils)
  - calculate utilities (In an extra function)
  - use `simple_batch` function from utils for return value

- All query strategies are tested by a general unittest (`test_pool.py`):
  - Querying of every method is tested with standard configurations with 0, 1, and 5 initial labels.
  - For every class `ExampleQueryStrategy` that inherits from `SingleAnnotPoolBasedQueryStrategy` (stored in `_example.py`), it is automatically tested if there exists a file `test/test_example.py`. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleQueryStrategy(unittest.TestCase)`
  - Every parameter in `__init__()` will be tested if it is written the same as a class variable.
  - Every parameter arg in `__init__()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_init_param_arg()`.
  - Every parameter arg in `query()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_query_param_arg()`.
  - Standard parameters `random_state`, `X_cand`, `batch_size` and `return_utilities` are tested and do not have to be tested in the specific tests.

### Stream-based Query Strategies

- All query strategies are stored in a file `skactivml/stream/*.py`
- Every query strategy inherits from `SingleAnnotatorStreamBasedQueryStrategy`
- Every query strategy has either an internal budget handling or an outsourced budget_manager
- The class must implement the following functions:
  - `init`: function for initialization
  - `query`: identify the instances whose labels to select
  - `update`: adapting the budget monitoring according to the queried labels
- For typical class parameters we use standard names:
  - `random_state`: integer that acts as random seed or `np.random.RandomState` like sklearn
  - `budget`: % of labels that the strategy is allowed to query
  - `budget_manager`: enforces the budget constraint
- Parameters of the query function are (similar to pool):
  - `X_cand` (Set of candidate instances, inherited from `SingleAnnotatorStreamBasedQueryStrategy`)
  - `clf` (The classifier used by the strategy)
  - `X` (Set of labeled and unlabeled instances)
  - `y` (labels of `X` (it may be set to `MISSING_LABEL` if `y` is unknown))
  - `sample_weight` (weights for each instance in `X` or `None` if all are equally weighted)
  - `return_utilities` (inherited from SingleAnnotatorStreamBasedQueryStrategy)
- The `query` function returns:
  - `queried_indices` (indices of the best instances from `X_Cand`)
  - `utilities` (utilities of all candidate instances, only if `return_utilities` is `True`)
- General advice for `query` code:
  - the `query` function must not change the internal state of the `query` strategy (`budget` and `random_state` included) to allow for assessing multiple instances with the same state. Update the the internal state in the `update()` function
  - use `self._validate_data` function (is implemented in superclass)
  - check the input `X` and `y` only once
  - fit classifier if it is not yet fitted (may use `fit_if_not_fitted` from `utils`)
- Typical parameters of the update function are:
  - `X_cand` (Set of candidate instances, inherited from `SingleAnnotatorStreamBasedQueryStrategy`)
  - `queried_indices` (typically the return value of `query`)
  - `budget_manager_param_dict`: provides additional parameters to the `update` function of the `budget_manager` (only include if a `budget_manager` is used)
- general advice for `update` code:
  - use `self._validate_data` in case the strategy is used without using the `query` method (if parameters need to be initialized before the update)
  - If a `budget_manager` is used forward the update call to the `budget_manager.update` method
- All stream query strategies are tested by a general unittest (`stream/tests/test_stream.py`):
  - For every class `ExampleQueryStrategy` that inherits from `SingleAnnotStreamBasedQueryStrategy` (stored in `_example.py`), it is automatically tested if there exists a file `test/test_example.py`. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleQueryStrategy` and inherit from `unittest.TestCase`
  - Every parameter in `init()` will be tested if it is written the same as a class variable.
  - Every parameter arg in `init()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_init_param_arg()`.
  - Every parameter arg in `query()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_query_param_arg()`.

#### `budget_manager` for stream-based query strategies

- All budget managers are stored in `skactivml/stream/budget_manager/\*.py`
- The class must implement the following functions:
  - `__init__`: function for initialization
  - `query_by_utilities`: identify which instances to query based on the assessed utility
  - `update`: adapting the budget monitoring according to the queried labels
- The update function of the budget manager has the same functionality as the query strategy update
- For typical class parameters we use standard names:
  - `budget`: % of labels that the strategy is allowed to query
  - `random_state`: integer that acts as random seed or `np.random.RandomState` like sklearn
- Typical parameters of the `query_by_utilities` function are:
  - `utilities` (The `utilities` of `X_cand` calculated by the query strategy, inherited from `BudgetManager`)
- General advice for working with a `budget_manager`:
  - If a `budget_manager` is used, the `_validate_data` of the query strategy needs to be adapted accordingly:
    - If only a `budget` is given use the default `budget_manager` with the given budget
    - If only a `budget_manager` is given use the `budget_manager`
    - If both are not given use the default `budget_manager` with the default budget
    - If both are given and the budget differs from `budget_manager.budget` throw an error
- All budget managers are tested by a general unittest (`stream/budget_manager/tests/test_budget_manager.py`):
  - For every class `ExampleBudgetManager` that inherits from `BudgetManager` (stored in `_example.py`), it is automatically tested if there exists a file `test/test_example.py`. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleBudgetManager` and inheriting from `unittest.TestCase`
  - Every parameter in `__init__()` will be tested if it is written the same as a class variable.
  - Every parameter `arg` in `__init__()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_init_param_arg()`.
  - Every parameter `arg` in `query_by_utility()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_query_by_utility` `_param_arg()`.

### Multi-Annotator Pool-based Query Strategies

- All query strategies are stored in a file `skactiveml/pool/multi/_query_strategy.py`

- Every class inherits from `MultiAnnotatorPoolBasedQueryStrategy`

- The class must implement the following functions:

  - `__init__`: function for initialization of hyperparameters
  - `query`: identify the instance annotator pairs whose labels to select

- For typical class parameters we use standard names:

  - `random_state` (number or `np.random.RandomState` like sklearn)

- Typical parameters of the `query` function are:

  - `X_cand` (Sequence of candidate instances to be queried, inherited from `MultiAnnotatorPoolBasedQueryStrategy`)
  - `A_cand` (Boolean mask further specifying which annotator can be queried for which candidate instance, inherited from `MultiAnnotatorPoolBasedQueryStrategy`)
  - `clf` (The classifier used by the strategy)
  - `X` (Sequence of labeled and unlabeled instances)
  - `y` ((unknown) Labels of `X` for each annotator)
  - `sample_weight` (Weights of the prediction of a sample from an annotator (used for predictions of labels))
  - `A_perf` (Performance of an annotators for a given sample, usually the accuracy (used for estimating the best annotator to query for a given candidate sample))
  - `batch_size` (Number of instances for batch querying, inherited from `MultiAnnotatorPoolBasedQueryStrategy`)
  - `return_utilities` (Inherited from `MultiAnnotatorPoolBasedQueryStrategy`)

- The query function returns:

  - `query_indices` (Indices of the best candidate instance annotator pair)
  - `utilities` (Utilities of all candidate instances annotator pairs, only if `return_utilities` is `True`)

- General advice for the query code:

- use `self._validate_data function` (is implemented in superclass)
- check the input `X` and `y` only once
- fit classifier if it is not yet fitted (may use `fit_if_not_fitted` form `utils`)
- if the strategy combines a single annotator query strategy with a performance estimate

  - define an aggregation function
  - evaluate the performance for each annotator sample pair
  - use the `MultiAnnotWrapper`

- if the strategy is a `greedy` method regarding the utilities
  - calculate utilities (in an extra function)
  - use `simple_batch` function from utils for return value

## Classifiers

- Standard classifier implementations are part of the subpackage `skactiveml.classifier` and classifiers learning from multiple annotators are implemented in its subpackage `skactiveml.classifier.multi`.

- Every class of a classifier inherits from `skactiveml.base.SkactivemlClassifier`.

- The class of a classifier must implement the `__init__` method for initialization, a `fit` method for training, and a `predict_proba` method predicting class membership probabilities for samples. A `predict` method is already implemented in the superclass by using the outputs of the `predict_proba` method. Additionally, a `score` method is implemented by the superclass to evaluate the accuracy of a fitted classifier.

- A commonly used subclass of `skactiveml.base.SkactivemlClassifier` is the sk`activeml.base.ClassFrequencyEstimator`, which requires an implementation of the method `predict_freq`, which can be interpreted as prior parameters of a Dirichlet distribution over the class membership probabilities of a sample.

- Typical `__init__` parameters are:

  - `classes`: Holds the label for each class. If `None`, the classes are determined during the fit.
  - `missing_label`: Value to represent a missing label.
  - `cost_matrix`: Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class `classes[j]` for a sample of class `classes[i]`. Can be only set, if classes is not `None`.
  - `random_state`: Ensures reproducibility (cf. scikit-learn).
  - `class_prior`: A `skactiveml.base.ClassFrequencyEstimator` requires additionally this parameter as prior observations of the class frequency estimates.

- Required `fit` parameters are:

  - `X`: Is a matrix of feature values representing the samples.
  - `y`: Contains the class labels of the training samples. Missing labels are represented through the attribute 'missing_label'. Usually, `y` is a column array except for multi-annotator classifiers which expect a matrix with columns containing the class labels provided by a specific annotator.
  - `sample_weight`: Contains the weights of the training samples' class labels. It must have the same shape as `y`.

- `fit` method returns:

  - `self`: The fitted classifier object.

- General advice for the `fit` code:

  - Use `self._validate_data` method (is implemented in superclass) to check standard parameters of `__init__` and `fit` method.
  - If `self.n_features_` is None, no samples were provided as training data. In this case, the classifier should still be fitted but only for the purpose to make random predictions, i.e., outputting uniform class membership probabilities when calling `predict_proba`.
  - Ensure that the classifier can handle missing labels.

- Required `predict_proba`, `predict_freq`, and `predict` parameters are: X: Is a matrix of feature values representing the samples, for which the classifier will make predictions.

- `predict_proba` method returns:

  - `P`: The estimated class membership probabilities per sample.

- General advice for the `predict_proba` code:

  - Check parameter `X` regarding its shape, i.e., use superclass method `self._check_n_features` to ensure a correct number of features.
  - Check that the classifier has been fitted.
  - If the classifier is a `skactiveml.base.ClassFrequencyEstimator`, this method is already implemented in the superclass.
  - If no samples or class labels were provided during the previous call of the `fit` method, uniform class membership probabilities are to be outputted.

- `predict_freq` method returns:

  - `F`: The estimated class frequency estimates (excluding the prior observations).

- General advice for the `predict_freq` code:

  - Check parameter X regarding its shape, i.e., use superclass method `self._check_n_features` to ensure a correct number of features.
  - Check that the classifier has been fitted.
  - If no samples or class labels were provided during the previous call of the `fit` method, a matrix of zeros is to be outputted.

- `predict` method returns:

  - `y_pred`: The estimated class label of each sample.

- General advice for the `predict` code:

  - Usually, this method is already implemented by the superclass through calling the `predict_proba` method.
  - If the superclass method is overwritten, ensure that it can handle imbalanced costs and missing labels.
  - If no samples or class labels were provided during the previous call of the `fit` method, random class label predictions are to be outputted.

- Required `score` parameters are:

  - `X`: Is a matrix of feature values representing the samples, for which the classifier will make predictions.
  - `y`: Contains the true label of each sample.
  - `sample_weight`: Defines the importance of each sample when computing the accuracy of the classifier.

- `score` method returns:

  - `score`: Mean accuracy of `self.predict(X)` regarding `y`.

- General advice for the `score` code:

  - Usually, this method is already implemented by the superclass.
  - If the superclass method is overwritten, ensure that it checks the parameters and that the classifier has been fitted.

- All classifiers are tested by a general unittest (`skactiveml/classifier/tests/test_classifier.py`):
  - For every class `ExampleClassifier` that inherits from `skactiveml.base.SkactivemlClassifier` (stored in `_example_classifier.py`), it is automatically tested if there exists a file `tests/test_example_classifier.py`. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleClassifier` and inherit from `unittest.TestCase`.
  - For each parameter of an implemented method, there must be a test method called `test_methodname_parametername` in the Python file `_example_classifier.py`. It is to check whether invalid parameters are handled correctly.
  - For each implemented method, there must be a test method called `test_methodname` in the Python file `_example_classifier.py`. It is to check whether the method works as intended.

## Annotators Models

- Annotator models are marked by implementing the interface `skactiveml.base.AnnotMixing`. These models can estimate the performances of annotators for given samples.

- Every class of a classifier inherits from `skactiveml.base.SkactivemlClassifier`.

- The class of an annotator model must implement the `predict_annot_perf` method estimating the performances per sample of each annotator as proxies of the provided annotation's qualities.

- Required `predict_annot_perf` parameters are:

  - `X`: Is a matrix of feature values representing the samples.

- `predict_annot_perf` method returns:

  - `P_annot`: The estimated performances per sample-annotator pair.

- General advice for the `predict_annot_perf` code:
  - Check parameter `X` regarding its shape.
  - Check that the annotator model has been fitted.
  - If no samples or class labels were provided during the previous call of the `fit` method, the maximum value of annotator performance should be outputted for each sample-annotator pair.

## Testing and improving test coverage

> TODO: Implement me

- Codecov
- How to run tests
- Code conventions
- Test conventions
  - See Coding Conventions

## Documentation (User guide and Developer guide)

### Guidelines for writing documentation

In `Scikit-ActiveML`, the [guidelines](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation) for writing the documentation are adopted from [scikit-learn](https://scikit-learn.org/stable/).

### Building the documentation

> TODO: Build the user guide and the developer guide?

## Issue Tracking

> TODO: Github templates in .github?

We use [Github Issues](https://github.com/scikit-activeml/scikit-activeml/issues) as our issue tracker. If you think you have found a bug in `Scikit-ActiveML`, you can report it to the issue tracker. Documentation bugs can also be reported there.

### Checking If A Bug Already Exists

The first step before filing an issue report is to see whether the problem has already been reported. Checking if the problem is an existing issue will:

- help you see if the problem has already been resolved or has been fixed for the next release
- save time for you and the developers
- help you learn what needs to be done to fix it
- determine if additional information, such as how to replicate the issue, is needed

To see if the issue already exists, search the issue database (`bug` label) using the search box on the top of the issue tracker page.

### Reporting an issue

Use the following labels to report an issue:

| Label           | Usecase                              |
| --------------- | ------------------------------------ |
| `documentation` | Improvement or additions to document |
| `enhancement`   | New feature                          |
| `guideline`     | # TODO                               |
| `question`      | # TODO                               |
| `bug`           | Something isn't working              |
| `upcoming`      | # TODO                               |
| `classifier`    | # TODO                               |
| `regressor`     | # TODO                               |
| `stream`        | # TODO                               |
| `pool`          | # TODO                               |
| `test`          | # TODO                               |
