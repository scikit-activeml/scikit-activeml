# Developer Guide

> TODO-MMUEJDE: Please use capitalized captions.

`scikit-activeml` is a library that implements the most important query strategies of active learning. It is built upon the well-known machine learning framework scikit-learn.

## Introduction

### Getting Help

If you have any questions, please reach out to other developers via the following channels:

- [Github Issues](https://github.com/scikit-activeml/scikit-activeml/issues)

### Roadmap

Our Roadmap is summarized in the issue [Upcoming Features](https://github.com/scikit-activeml/scikit-activeml/issues/145).

## Get Started

Before you can contribute to this project, you might execute the following steps.

### Setup Development Environment

> TODO-MMUEJDE: Add links

There are several ways to create a local Python environment, such as virtualenv[], pipenv[], miniconda[], etc. One possible workflow is to install miniconda and use it to create a Python environment. And use pip to install packages in this environment.

#### Example With miniconda

Create a new python environment named scikit-activeml:

```bash
conda create -n scikit-activeml
```

To be sure that the correct env is active:

```bash
conda activate scikit-activeml
```

Then install the pip:

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

> TODO-MMUEJDE: Ghostscript unten ergänzen und alles vervollständigen (Links zu Packages ist ggf ausreichend. Beispiele sind zu divers)

#### Example: Macintosh (Homebrew)

```bash
brew install pandoc
```

#### Example: Linux (Homebrew)

> TODO-MMUEJDE: Add cmd

```bash
# Implement me
```

#### Example: Windows (Homebrew)

> TODO-MMUEJDE: Add cmd

```bash
# Implement me
```

## Contributing Code

### General Coding Conventions

> TODO-MMUEJDE: Einfach ein Verweis auf Sklearn oder wie unten? (Letzter Stand: Einfach ein Verweis auf Sklearn) | Verweis auf sklearn ist ausreichend. L

As this library conforms to the convention of scikit-learn, the code should conform to PEP 8 Style Guide for Python Code. For linting, the use of flake8 is recommended.

> TODO-MMUEJDE: Ab hier nur Verweis auf sklearn reicht. Mit Link zu den guides für sklearn und pep8

- File organization
- Comments?
- Naming Conventions
  - Files
  - Classes
  - Functions
  - Vars
  - Maybe known python naming conventions?
- Commit Messages
  - https://chris.beams.io/posts/git-commit/
- etc.

### Example for C3 (Code Contribution Cycle) and Pull Requests

> TODO-MMUEJDE: Brauchen wir sowas? | Gute Idee. 

1. Fork the repository and clone your fork to your local machine:

```bash
git clone https://github.com/USERNAME/FORKED_REPOSITORY.git
```

2. Create a feature branch for the changes from the `master` branch:

```bash
git checkout -b <feature, bug-fix, hotfix, etc...>/new-feature dev
```

> Make sure that you create your branch from `master`.

3. After you have finished implementing the feature, make sure that all the tests pass. The tests can be run as

```bash
$ pytest
```

Make sure, you covered all lines by tests.

```bash
$ pytest --cov=./skactiveml
```

4. Commit and push the changes.

> TODO-MMUEJDE: Bitte beschreiben, wie das als Pull request bei uns auftaucht.

> TODO-MMUEJDE: Bitte hier noch schreiben, dass wir über alle contributions froh sind. Kurzer Satz dazu reicht.

```bash
$ git add modified_files
$ git commit -m 'commit message explaning the changes briefly'
$ git push origin new-feature
```

## Query Strategies

> TODO-MMUEJDE: Übernommen aus [Github Ticket](https://github.com/scikit-activeml/scikit-activeml/issues/186#issuecomment-981907257).

> TODO-MMUEJDE: Review

> TODO-ALL@MMUEJDE: Einleitetext mit Übersicht der allgmeinen Struktur (Klassen, Methoden) und kurze Erklärung, dass jedes Szenario der Einfachheit wegen gesondert beschrieben wird. Evtl. einbringen eines UML Diagramms (mit/ohne Parameter/Attribute je nach Übersichtlichkeit).

### Pool-based Query Strategies

> TODO-ALL@MMUEJDE: Bitte Vorschlag machen, wie man die einzelnen Beschreibungen visuell besser darstellen kann (z.B. Einleitetext mit Sätzen. Methoden als Anstriche und Parameter/Attribute als Unterpunkte, evtl Unterüberschriften-einheitlich für jedes Szenario).

- All query strategies are stored in a file skactiveml/pool/_query_strategy.py
- Every class inherits from `SingleAnnotatorPoolBasedQueryStrategy`
- The class must implement the `__init__` finction for initialization and a `query` function
- For typical class parameters we use standard names:
  - `prior` (Prior probabilities for the distribution of probabilistic strategies)
  - `random_state` (number or np.random.RandomState like sklearn)
  - `method` (string for classes that implement multiple methods)
  - `cost_matrix` (Cost matrix defining the cost of predicting instances wrong)

- Typical parameters of the query function are:
  - `X_cand` (Set of candidate instances, inherited from `SingleAnnotatorPoolBasedQueryStrategy`)
  - `clf` (The classifier used by the strategy)
  - `X` (Set of labeled and unlabeled instances)
  - `y` ((unknown) labels of `X`)
  - `sample_weight` (Weights of training samples in `X`)
  - `sample_weight_cand` (Weights of samples in `X_cand`)
  - `batch_size` (Number of instances for batch querying, inherited from `SingleAnnotatorPoolBasedQueryStrategy`)
  - `return_utilities` (inherited from `SingleAnnotatorPoolBasedQueryStrategy`)

- The `query` function returns:
  - `query_indices` (indices of the best instances)
  - `utilities` (utilities of all candidate instances, only if `return_utilities` is True)
  
- General advice for the query code:
  - use `self._validate_data` function (is implemented in superclass)
  - check the input `X` and `y` only once
  - fit classifier if it is not yet fitted (may use `fit_if_not_fitted` form utils)
  - calculate utilities (in an extra function)
  - use `simple_batch` function from utils for return value

- All query strategies are tested by a general unittest (test_pool.py):
  - Querying of every method is tested with standard configurations with 0, 1, and 5 initial labels.
  - For every class `ExampleQueryStrategy` that inherits from `SingleAnnotPoolBasedQueryStrategy` (stored in _example.py), it is automatically tested if there exists a file test/test_example.py. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleQueryStrategy(unittest.TestCase)`
  - Every parameter in `__init__()` will be tested if it is written the same as a class variable.
  - Every parameter arg in `__init__()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_init_param_arg()`.
  - Every parameter arg in `query()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_query_param_arg()`.
  - Standard parameters `random_state`, `X_cand`, `batch_size` and `return_utilities` are tested and do not have to be tested in the specific tests.

## Estimators

> TODO-MMUEJDE: Kein Content

## Testing and improving test coverage

> TODO-MMUEJDE: Implement me

- Codecov
- How to run tests
- Code conventions
- Test conventions
  - See Coding Conventions

## Documentation (User guide and Developer guide)

> TODO-MMUEJDE: Review

### Guidelines for writing documentation

> TODO-MMUEJDE: Sollen wir bestimmte Regeln aus [here](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation) übernehmen? | ALL: einfach nur zitieren und erwähnen, dass wir uns daran halten

### Building the documentation

> TODO-MMUEJDE: Implement me

## Issue Tracking

We use [Github Issues](https://github.com/scikit-activeml/scikit-activeml/issues) as our issue tracker.

If you think you have found a bug in scikit-activeml, you can report it to the issue tracker. Documentation bugs can also be reported there.

### Checking If A Bug Already Exists

The first step before filing an issue report is to see whether the problem has already been reported.

Checking if the problem is an existing issue will:

- help you see if the problem has already been resolved or has been fixed for the next release
- save time for you and the developers
- help you learn what needs to be done to fix it
- determine if additional information, such as how to replicate the issue, is needed

To see if the issue already exists, search the issue database (`bug` label) using the search box on the top of the issue tracker page.

### Reporting an issue

> TODO-MMUEJDE: Muss was beachtet werden? | ALL: mach mal einen konkreten entwurf. Wir schauen dann. Sieh auch unsere labels.

- Use the following labels
  - documentation: If you ...
  - bug: if ...
  - cosmetics: if ...
  - feature: if ...
  - nice-to-have: if ...
  - ...
- Post error message
- python version
- dependency versions
- ...
