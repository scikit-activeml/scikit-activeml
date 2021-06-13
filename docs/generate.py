import errno
import inspect
import json
import os
import string

import numpy as np

from skactiveml import pool, classifier, utils  # , stream TODO uncomment for stream
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


def generate_stratagy_summary_rst(path, methods={}, refs={}):
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
        file.write(table_from_array(get_table_data(pool, methods, refs, "generated/api/"),  # TODO
                                    title='',
                                    wights='20 20 20'))
        file.write('\n')
        # TODO uncomment for stream
        # file.write('Stream Strategies:\n')
        # file.write('------------------\n')
        # file.write('\n')
        # file.write(table_from_array(get_table_data(stream),
        #                            title='',
        #                            wights='20 20 20 20'))
        file.write('\n')
        file.write('.. footbibliography::')
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


def get_table_data(package, methods, refs, rel_path_api):
    data = np.array([['Strategy', 'Methods', 'Reference']])
    query_strategies = {}
    package_name = package.__name__.split('.')[1]
    for qs_name in package.__all__:
        query_strategies[qs_name] = getattr(package, qs_name)
    for qs_name, strat in query_strategies.items():
        metods_text = ''
        ref_text = ''
        if qs_name in methods.keys():
            for m in methods[qs_name]:
                # TODO path
                metods_text += ':doc:`{} <{}>`, '.format(m, 'generated/sphinx_gallery_examples/'+package_name+'/plot_'+qs_name+'_'+m)
        else:
            metods_text += ':doc:`{} <{}>`, '.format('default', 'generated/sphinx_gallery_examples/' + package_name + '/plot_' + qs_name)
        metods_text = metods_text[0:-2]
        strategy_text = ':doc:`{} <{}{}.{}>`' \
                        ''.format(qs_name, rel_path_api, package.__name__, qs_name)
        if qs_name in refs.keys():
            for ref in refs[qs_name]:
                ref_text += ':footcite:t:`{}` :, '.format(ref)
            ref_text = ref_text[0:-2]
        data = np.append(data, [[strategy_text, metods_text, ref_text]],
                         axis=0)

    return data


def generate_examples(example_path, package, json_path):
    dir_path = example_path + '\\' + \
        package.__name__.split('.')[1] + "\\"
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(example_path + '\\README.rst', 'w') as file:
        file.write('Example Gallery\n')
        file.write('===============')

    methods = {}
    refs = {}
    for filename in os.listdir(json_path):
        with open(json_path + '\\' + filename) as file:
            json_data = json.load(file)
            methods_list = []
            for data in json_data:
                plot_filename = 'plot_' + data["class"]
                if "method" in data.keys():
                    if data['class'] in methods.keys():
                        methods[data['class']].append(data["method"])
                    else:
                        methods[data['class']] = [data["method"]]
                    methods_list.append(data["method"])
                    plot_filename += "_" + data["method"]
                if "refs" in data.keys():
                    if data['class'] in refs.keys():
                        for r in data['refs']:
                            if r not in refs[data['class']]:
                                refs[data['class']].append(r)
                    else:
                        refs[data['class']] =  data['refs']
                generate_example_script(plot_filename + '.py',
                                        dir_path, data)
    return methods, refs
    query_strategies = {}
    for qs_name in package.__all__:
        query_strategies[qs_name] = getattr(package, qs_name)
    for qs_name, strat in query_strategies.items():
        pass


def generate_example_script(filename, dir_path, data):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    first_title = True
    with open(dir_path + filename, 'w') as file:
        code_blocks = []
        for block in data["blocks"]:
            if block.startswith("title"):
                block_str = format_title(data[block], first_title)
                first_title = False
            elif block.startswith("text"):
                block_str = format_text(data[block])
            elif block.startswith("code"):
                code_blocks.append(data[block])
                block_str = format_code(data[block])
            elif block.startswith("example"):
                block_str = format_example(data["init_params"],
                                           data["query_params"])
            elif block.startswith("plot"):
                if "clf" in data.keys():
                    clf = data["clf"]
                else:
                    clf = None
                block_str = format_plot(data["class"],
                                        data["init_params"],
                                        data["query_params"],
                                        clf=clf)
            elif block.startswith("refs"):
                block_str = format_refs(data[block])

            file.write(block_str)

        file.write("\n")

        return


def format_title(title, first_title=False):
    if first_title:
        block_str = '"""\n' \
                    '' + title + '\n' \
                    ''.ljust(len(title)+1, '=') + '\n' \
                    '"""\n' \
                    '\n'
    else:
        block_str = '# %%\n'\
                    '# ' + title + '\n'
        block_str += '# '.ljust(len(title)+1, '=') + '\n\n'
    return block_str


def format_text(text):  # TODO Atal
    block_str = '# %%\n'
    for line in text.split('\n'):
        block_str += '# ' + line + '\n'
    return block_str + '\n'


def format_code(code):  # TODO Atal
    block_str = ''
    for line in code.split('\n'):
        block_str += line + '\n'
    return block_str + '\n'


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


def format_plot(qs_name, init_params, query_params, clf=None):

    block_str = ('import numpy as np\n')
    block_str += ('from matplotlib import pyplot as plt, animation\n')
    block_str += ('from sklearn.datasets import make_classification\n')
    block_str += ('from skactiveml.classifier import SklearnClassifier, PWC\n')
    block_str += ('from skactiveml.utils import MISSING_LABEL, is_unlabeled, plot_2d_dataset\n')
    block_str += ('\n')
    block_str += ('fig, ax = plt.subplots()\n')
    block_str += ('X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)\n')
    block_str += ('y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n')
    if clf is None:
        clf = 'PWC(classes=[0, 1])'
    block_str += ('clf = ' + clf + '\n')
    if 'clf' not in init_params.keys() and 'clf' in inspect.signature(getattr(pool, qs_name).__init__).parameters:
        init_params['clf'] = 'clf'
    block_str += ('qs = {}({})\n'.format(qs_name, dict_to_str(init_params)))
    block_str += ('\n')
    block_str += ('artists = []\n')
    block_str += ('n_cycles = 20\n')
    block_str += ('for c in range(n_cycles):\n')
    block_str += ('    unlbld_idx = np.where(is_unlabeled(y))[0]\n')
    block_str += ('    X_cand = X[unlbld_idx]\n')
    query_params_str = ''
    if query_params_str != '':
        query_params_str = ', ' + dict_to_str(query_params)
    block_str += ('    query_idx = unlbld_idx[qs.query(X_cand, X, y{})]\n'.format(query_params_str))
    block_str += ('    if c in [1, 2, 3, 5, 10, 15, 19]:\n')
    block_str += ('        temp = np.array(ax.collections)\n'.format(query_params_str))
    block_str += ('        collections = plot_2d_dataset(X, y, y_true, clf, qs, ax)\n'.format(query_params_str))
    block_str += ('        artists.append([x for x in collections if (x not in temp)])\n'.format(query_params_str))
    block_str += ('    y[query_idx] = y_true[query_idx]\n')
    block_str += ('    clf.fit(X, y)\n')
    block_str += ('\n')
    block_str += ('ani = animation.ArtistAnimation(fig, artists, blit=True)\n')

    return block_str + '\n'


def format_refs(refs):  # TODO Atal
    if not refs:
        return ''
    block_str = '# %%\n' \
                '# References:\n' \
                '# ===========\n' \
                '# .. footbibliography::\n' \
                '#    :filter: key in {'
    for ref in refs:
        block_str += '"{}", '.format(ref.lower())
    block_str = block_str[0:-2] + '}\n'

    return block_str + '\n'


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
