import json
import os
import string

import numpy as np

from skactiveml import pool, classifier, utils#, stream TODO uncomment for stream
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import is_unlabeled, MISSING_LABEL, plot_2d_dataset
from skactiveml.classifier import SklearnClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")



def generate_api_reference_rst(path):
    with open(path, 'w') as file:
        file.write('API Reference\n')
        file.write('=============\n')
        file.write('\n')
        file.write('.. toctree::\n')
        file.write('\n')
        file.write('This is an overview of the API.\n')
        file.write('\n')
        file.write('.. currentmodule:: skactiveml\n')
        file.write('\n')

        file.write('Pool:\n')
        file.write('-----\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        file.write('   :nosignatures:\n')
        file.write('   :toctree: generated/api/\n')
        file.write('   :template: class.rst\n')
        file.write('\n')
        for qs_name in pool.__all__:
            file.write('   pool.{}\n'.format(qs_name))
        file.write('\n')

        file.write('Classifier:\n')
        file.write('-----------\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        file.write('   :nosignatures:\n')
        file.write('   :toctree: generated/api/\n')
        file.write('   :template: class.rst\n')
        file.write('\n')
        for qs_name in classifier.__all__:
            file.write('   classifier.{}\n'.format(qs_name))
        file.write('\n')

        file.write('Utils:\n')
        file.write('------\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        file.write('   :nosignatures:\n')
        file.write('   :toctree: generated/api/\n')
        file.write('   :template: class.rst\n')  # TODO change template?
        file.write('\n')
        for qs_name in utils.__all__:
            file.write('   utils.{}\n'.format(qs_name))
        file.write('\n')


def generate_stratagy_summary_rst(path):
    with open(path, 'w') as file:
        file.write('Strategy Summary\n')
        file.write('================\n')
        file.write('\n')
        file.write('.. toctree::\n')
        file.write('\n')
        file.write('This is a summary of all implemented AL strategies.\n')
        file.write('\n')
        file.write('Pool Strategies:\n')
        file.write('----------------\n')
        file.write('\n')
        file.write(table_from_array(get_table_data(pool),
                                    title='',
                                    wights='20 20 20 20'))
        file.write('\n')
        # TODO uncomment for stream
        # file.write('Stream Strategies:\n')
        # file.write('------------------\n')
        # file.write('\n')
        # file.write(table_from_array(get_table_data(stream),
        #                            title='',
        #                            wights='20 20 20 20'))
        file.write('\n')



def table_from_array(a, title, wights, header_rows=1):
    a = np.asarray(a)
    table = '.. list-table:: {}\n   :widths: {}\n   :header-rows: {}\n\n' \
            ''.format(title, wights, header_rows)
    for column in a:
        table += '   *'
        for row in column:
            table += ' - ' + str(row) + '\n    '
        table = table[0: -4]
    return table


def get_table_data(package):
    data = np.array([['Strategy', 'Methods', 'Examples', 'Reference']])
    query_strategies = {}
    for qs_name in package.__all__:
        query_strategies[qs_name] = getattr(package, qs_name)
    for qs_name, strat in query_strategies.items():
        metods_text = ''
        if hasattr(strat, '_methods'):
            for m in strat._methods:
                metods_text += m + ', '
            metods_text = metods_text[0:-2]
        strategy_text = ':doc:`{} </generated/api/{}.{}>`' \
                        ''.format(qs_name, package.__name__, qs_name)
        example_text = 'Example {}'.format(qs_name)
        ref_text = 'Reference {}'.format(qs_name)
        data = np.append(data, [[strategy_text, metods_text, example_text, ref_text]],
                         axis=0)

    return data


def generate_examples(path, package):
    json_path = "{}\\examples.json".format(os.path.dirname(package.__file__))
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    for data in json_data:
        example_path = path + '\\' +\
                       package.__name__ + "." + data["class"] + '.rst'
        generate_example_rst(example_path, data)

    return
    query_strategies = {}
    for qs_name in package.__all__:
        query_strategies[qs_name] = getattr(package, qs_name)
    for qs_name, strat in query_strategies.items():
        pass



def generate_example_rst(path, data):
    with open(path, 'w') as file:
        code_blocks = []
        for block in data["blocks"]:
            if block.startswith("title"):
                block_str = format_title(data[block])
            elif block.startswith("text"):
                block_str = format_text(data[block])
            elif block.startswith("code"):
                code_blocks.append(data[block])
                block_str = format_code(data[block])
            elif block.startswith("example"):
                block_str = format_example(data["init_params"],
                                           data["query_params"])
            elif block.startswith("plot"):
                block_str = format_plot(code_blocks,
                                        data["init_params"],
                                        data["query_params"])
            elif block.startswith("refs"):
                block_str = format_refs(data[block])

            file.write(block_str)

        return
        file.write('{}\n'.format(title))
        file.write('=====================================================\n')
        file.write('\n')
        file.write('.. toctree::\n')
        file.write('\n')
        file.write('The examplery cycles for different strategies are explained here.\n')
        file.write('\n')
        file.write('.. code-block:: python\n')
        file.write('\n')
        file.write('    X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)\n')
        file.write('    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n')
        file.write('    clf = SklearnClassifier(LogisticRegression(),  classes=np.unique(y_true))\n')
        file.write('    qs = {}({})\n'.format(qs_name, params_init))
        file.write('    n_cycles = 20\n')
        file.write('    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n')
        file.write('    for c in range(n_cycles):\n')
        file.write('            unlbld_idx = np.where(is_unlabeled(y))[0]\n')
        file.write('            X_cand = X[unlbld_idx]\n')
        file.write('            query_idx = unlbld_idx[qs.query(X_cand, X, y, {})]\n'.format(dict_to_str(query_params)))
        file.write('            y[query_idx] = y_true[query_idx]\n')
        file.write('            clf.fit(X, y)\n')
        file.write('            X_cand = X[unlbld_idx]\n')


def format_title(title):  # TODO Atal
    block_str = title + "\n" + "=".ljust(len(title),"=") + "\n"
    return block_str

def format_text(text):  # TODO Atal
    block_str = text + "\n" + "\n"
    return block_str


def format_code(code):  # TODO Atal
    code = code.splitlines()
    new_len = len(code[0])+ 3
    new_len2 = len(code[1]) + 3
    block_str = ".. code-block:: python\n" + "\n" + code[0].rjust(new_len) + \
                "\n" + code[1].rjust(new_len2) + "\n" + "\n"
    return block_str




def format_example(init_params, query_params): # TODO Atal

    """

    Parameters
    ----------
    init_params
    query_params

    Returns
    -------
    """
    block_str = ""
    block_str += ".. code-block:: python\n"
    block_str += "\n"
    block_str += "    X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)\n"
    block_str += "    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n"
    block_str += "    clf = SklearnClassifier(LogisticRegression(),  classes=np.unique(y_true))\n"
    block_str += "    qs = ({})\n".format(dict_to_str(init_params))
    block_str += "    n_cycles = 20\n"
    block_str += "    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n"
    block_str += "    for c in range(n_cycles):\n"
    block_str += "            unlbld_idx = np.where(is_unlabeled(y))[0]\n"
    block_str += "            X_cand = X[unlbld_idx]\n"
    block_str += "            query_idx = unlbld_idx[qs.query(X_cand, X, y, {})]\n".format(dict_to_str(query_params))
    block_str += "            y[query_idx] = y_true[query_idx]\n"
    block_str += "            clf.fit(X, y)\n"
    block_str += "            X_cand = X[unlbld_idx]\n"

    return block_str


def format_plot(code_blocks, init_params, query_params):
    block_str = ".. plot:: pyplots/ellipses.py\n"
    block_str += "   :include - source:\n"
    block_str += "\n"
    return block_str


def format_refs(ref):  # TODO Atal
    block_str = ".. code-block:: python\n" + "\n"\
                "   :cite:p:`settles2009active`\n" + \
                "\n" + \
                "   .. bibliography::\n"
    return block_str


def dict_to_str(d, idx=None):
    """Converts a dictionary into a string.
    Parameters
    ----------
    d : dict
        The dictionary to be converted.
        Shape: {key1:value1,...} or {key1:[value1, value2,...],...} or a
        combination of both.
    idx: : dict, optional
        If a key has multiple values, idx[key] chooses the used value for the
        specific key. If idx is not given, the first value in the list is
        always used. It is not necessary to specify all keys from d.
        shape: {key1:int1,...}

    Returns
    -------
    String : dict_ as String. Shape: 'key1=value[idx[key1], key2...' or
             'key1=value1, key2=value2...' or a combination of both.
    """
    dd_str = ""
    for key, value in d.items():
        if not isinstance(value, list):
            value = [value]
        if idx is not None and key in idx.keys():
            value = value[idx[key]]
        else:
            value = value[0]
        dd_str += str(key) + "=" + value + ", "
    return dd_str[0:-2]

