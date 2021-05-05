import errno
import json
import os
import string

import numpy as np

from skactiveml import pool, classifier, utils#, stream TODO uncomment for stream
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import is_unlabeled, MISSING_LABEL, plot_2d_dataset, \
    call_func, is_labeled
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

def generate_examples(example_path, package, json_path):
    for filename in os.listdir(json_path):
        with open(json_path +"\\"+ filename) as file:
            json_data = json.load(file)
            dir_path = example_path + '\\' +\
                package.__name__.split('.')[1] + "\\"
            generate_example_rst(json_data["class"] + '.rst', dir_path, json_data)
    return
    query_strategies = {}
    for qs_name in package.__all__:
        query_strategies[qs_name] = getattr(package, qs_name)
    for qs_name, strat in query_strategies.items():
        pass



def generate_example_rst(filename,dir_path, data):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(dir_path + filename, 'w') as file:
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
                rel_path = \
                    "generated/examples/plot/plot_" + data["class"] + ".py"
                block_str = format_plot(code_blocks, data["class"],
                                        data["init_params"],
                                        data["query_params"],
                                        rel_path)
            elif block.startswith("refs"):
                block_str = format_refs(data[block])

            file.write(block_str)

        file.write("\n")

        return


def format_title(title):  # TODO Atal
    block_str = title + "\n"
    block_str += "".ljust(len(title), "-") + "\n"
    return block_str


def format_text(text):  # TODO Atal
    block_str = text + "\n" + "\n"
    return block_str


def format_code(code):  # TODO Atal
    block_str = ".. code-block:: python\n" + "\n"
    code = code.split("\n")
    for line in code:
        block_str += "   " + line + "\n"

    block_str += "\n"
    return block_str


def format_example(init_params, query_params):
    """

    Parameters
    ----------
    init_params
    query_params

    Returns
    -------

    """
    block_str = ""
    return block_str


def format_plot(code_blocks, qs_name, init_params, query_params, rel_path):
    directory, _ = os.path.split(rel_path)
    try:
        os.makedirs(os.path.abspath(directory))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(os.path.abspath(rel_path), "w") as file:
        file.write("import numpy as np\n")
        file.write("from matplotlib import pyplot as plt, animation\n")
        file.write("from sklearn.datasets import make_classification\n")
        file.write("from skactiveml.classifier import SklearnClassifier\n")
        file.write("from skactiveml.utils import MISSING_LABEL, is_unlabeled, plot_2d_dataset\n")
        file.write("from sklearn.linear_model import LogisticRegression\n")
        file.write("\n")
        for cb in code_blocks:
            file.write(cb + "\n")
        file.write('fig, ax = plt.subplots()\n')
        file.write('artists = []\n')
        file.write('X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)\n')
        file.write('y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n')
        file.write('clf = SklearnClassifier(LogisticRegression())\n')
        file.write('qs = {}({})\n'.format(qs_name, dict_to_str(init_params)))
        file.write('n_cycles = 20\n')
        file.write('for c in range(n_cycles):\n')
        file.write('    unlbld_idx = np.where(is_unlabeled(y))[0]\n')
        file.write('    X_cand = X[unlbld_idx]\n')
        query_params_str = ""
        if query_params_str != "":
            query_params_str = ", " + dict_to_str(query_params)
        file.write('    query_idx = unlbld_idx[qs.query(X_cand, X, y{})]\n'.format(query_params_str))
        file.write('    if c in [1]:\n')
        file.write('        artists.append([plot_2d_dataset(X, y, y_true, clf, qs)])\n'.format(query_params_str))
        file.write('    y[query_idx] = y_true[query_idx]\n')
        file.write('    clf.fit(X, y)\n')
        file.write('\n')
        file.write('ani = animation.ArtistAnimation(fig, artists, blit=True)\n')
        #file.write('plt.show()\n')

    block_str = ".. plot:: " + rel_path + "\n"
    block_str += "   :include-source:\n"
    block_str += "\n"
    return block_str


def format_refs(refs):  # TODO Atal
    if not refs:
        return ""
    block_str = "References:\n" \
                "===========\n" \
                ".. bibliography::\n" \
                "   :filter: key in {"
    for ref in refs:
        block_str += "'{}', ".format(ref.lower())

    block_str = block_str[0:-2] + "}"
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

