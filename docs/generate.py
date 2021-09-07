import inspect
import json
import os

import numpy as np

from skactiveml import pool, classifier, utils  # , stream TODO stream
import warnings

warnings.filterwarnings("ignore")


def generate_api_reference_rst(gen_path):
    """Creates the api_reference.rst file in the specified gen_path.

    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    """
    path = os.path.join(os.path.basename(gen_path), 'api_reference.rst')
    gen_path = os.path.join(os.path.basename(gen_path), 'api')
    os.makedirs(os.path.abspath(gen_path), exist_ok=True)
    with open(path, 'w') as file:
        file.write('.. _sphx_glr_api_reference:\n')
        file.write('\n')
        file.write('API Reference\n')
        file.write('=============\n')
        file.write('\n')
        file.write('.. toctree::\n')
        file.write('\n')
        file.write('This is an overview of the API.\n')
        file.write('\n')
        file.write('.. module:: skactiveml\n')
        file.write('\n')

        file.write('Pool:\n')
        file.write('-----\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        file.write('   :nosignatures:\n')
        file.write('   :toctree: api\n')
        file.write('   :template: class.rst\n')
        file.write('\n')
        for item in pool.__all__:
            file.write(f'   pool.{item}\n')
        file.write('\n')
        # TODO stream

        file.write('Classifier:\n')
        file.write('-----------\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        # file.write('   :nosignatures:\n')
        file.write('   :recursive:\n')
        file.write('   :toctree: api\n')
        file.write('   :template: class.rst\n')
        file.write('\n')
        for item in classifier.__all__:
            file.write(f'   classifier.{item}\n')
        file.write('\n')

        file.write('Utils:\n')
        file.write('------\n')
        file.write('\n')
        file.write('.. autosummary::\n')
        file.write('   :nosignatures:\n')
        file.write('   :toctree: api\n')
        file.write('   :template: base.rst\n')
        file.write('\n')
        for item in utils.__all__:
            if item == 'MISSING_LABEL': continue
            file.write(f'   utils.{item}\n')
        file.write('\n')


def generate_strategy_overview_rst(gen_path, examples_data={}):
    """Creates the strategy_overview.rst file in the specified path.
    
    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    examples_data : dict
        The additional data, needed to create the strategy overview. Shape:
        {'strategy_1' : [
            ['method_1', ['reference_1', ...], {'tab_1': 'category', ...}],
            ['method_2', ...],
            ...],
        'strategy_2' : [...],
        ...
        }
    """
    # Generate an array which contains the data for the tables. TODO stream
    api_path = os.path.join(os.path.basename(gen_path), 'api').replace('\\', '/')
    data = get_table_data(pool, examples_data, api_path)

    # create directory if it does not exist.
    os.makedirs(os.path.join(gen_path, 'strategy_overview'), exist_ok=True)

    # Generate file
    with open(
            os.path.join(gen_path, 'strategy_overview.rst'),
            'w') as file:
        file.write('Strategy Overview\n')
        file.write('=================\n')
        file.write('\n')
        file.write('In the following you\'ll find summaries of all implemented'
                   ' "Query Strategies", based on the categorization of '
                   'different papers.\n')
        file.write('\n')
        file.write('.. toctree::\n')
        file.write('   :maxdepth: 1\n')
        file.write('\n')
        for tab in data.keys():
            file.write(f'   strategy_overview/strategy_overview-{tab.replace(" ", "_")}\n')

    # Iterate over the tabs.
    for tab, cats in data.items():
        tab_space = tab.replace('_', ' ')
        tab_ul = tab.replace(' ', '_')
        path = os.path.join(gen_path,
                            'strategy_overview',
                            f'strategy_overview-{tab_ul}.rst')
        with open(path, 'w') as file:
            title = f'By {tab_space}\n'
            file.write(title)
            file.write(''.ljust(len(title) + 1, '=') + '\n')
            file.write('\n')
            file.write('.. toctree::\n')
            file.write('\n')
            file.write(f'This is a overview of all implemented AL strategies. '
                       f'The strategies are categorized, according to "'
                       f'{tab_space}". To categorize them by a different paper'
                       f', use the links below.\n')
            file.write('\n')
            links_str = ''
            for t in data.keys():
                t_space = t.replace('_', ' ')
                t_ul = t.replace(' ', '_')
                if t != tab:
                    links_str += f':doc:`{t_space} ' \
                                 f'<strategy_overview-{t_ul}>`,\n'
            file.write(links_str[0:-2] + '\n')
            file.write('\n')
            file.write('Pool Strategies:\n')
            file.write('----------------\n')
            file.write('\n')

            # Iterate over the categories in the current tab.
            for cat in sorted(cats):
                if cat != 'Others':
                    file.write(table_from_array(cats[cat], title=cat,
                                                header_rows=1, indent=0))
            if 'Others' in cats.keys():
                # 'Others' is the fallback, if no category is specified
                # in the json file
                file.write(table_from_array(cats['Others'], title='Others',
                                            header_rows=1, indent=0))
            file.write('.. footbibliography::')
            # TODO stream
            # file.write('\n')
            # file.write('Stream Strategies:\n')
            # file.write('------------------\n')
            # file.write('\n')
            file.write('\n')


def table_from_array(a, title='', widths=None, header_rows=0, indent=0):
    """Generates a rst-table and returns it as a string.

    Parameters
    ----------
    a : array-like, shape=(columns, rows)
        Contains the data for the table.
    title : string, optional (default='')
        The title of the table.
    widths : string, optional (default=None)
        A list of relative column widths separated with comma or space
        or 'auto'.
    header_rows : int, optional (default=0)
        The number of rows to use in the table header.
    indent : int, optional (default=0)
        Number of spaces as indent in each line

    Returns
    -------
    string : reStructuredText list-table as String.
    """
    a = np.asarray(a)
    indents = ''.ljust(indent, ' ')
    table = f'{indents}.. list-table:: {title}\n' \
            f'{indents}   :header-rows: {header_rows}\n'
    if widths is None:
        table += '\n'
    elif widths == 'auto':
        table += f'{indents}   :widths: auto\n\n'
    else:
        table += f'{indents}   :widths: {widths}\n\n'
    for row in a:
        table += f'{indents}   *'
        for column in row:
            table += ' - ' + str(column) + f'\n{indents}    '
        table = table[0: -4 - indent]
    return table + '\n'


def get_table_data(package, additional_data, rel_api_path):
    """Creates a np.array holding the data for the strategy overview.

    Parameters
    ----------
    package : module
        The '__init__' module of the package from which to get the data.
    additional_data : dict
        The additional data, needed to create the strategy overview. Shape:
        {'strategy_1' : [
            ['method_1', ['reference_1', ...], {'tab_1': 'category', ...}],
            ['method_2', ...],
            ...],
        'strategy_2' : [...],
        ...
        }
    rel_api_path : string
        The relative path of the directory in which the api reference rst files
        are saved.

    Returns
    -------
    np.array : Holds the data for the strategy overview, including the
        header row(Strategy | Method | Reference).

    """
    head_line = ['Strategy', 'Method', 'Reference']
    tabs = {}  # (Tab, Category, Collum, Row)

    # extract the tabs from 'examples_data'.
    for strat in additional_data.values():
        for method in strat:
            for tab in method[2].keys():
                tabs[tab] = {}

    temp = np.empty((0, 3))
    package_name = package.__name__.split('.')[1]
    # iterate over the query strategies in the package.
    for qs_name in sorted(package.__all__):
        strategy_text = \
            f':doc:`{qs_name} </{rel_api_path}/{package.__name__}.{qs_name}>`'
        if qs_name in additional_data.keys():
            # it exists an example for qs_name
            for m in range(len(additional_data[qs_name])):
                ref_text = ''
                method = additional_data[qs_name][m][0]
                refs = additional_data[qs_name][m][1]

                methods_text = \
                    f':doc:`{method} </generated/sphinx_gallery_examples/' \
                    f'{package_name}/plot_{qs_name}_{method}>`'
                for ref in refs:
                    ref_text += f':footcite:t:`{ref}`, '
                ref_text = ref_text[0:-2]
                temp = np.append(temp,
                                 [[strategy_text, methods_text, ref_text]],
                                 axis=0)
                for tab, cats in tabs.items():
                    if tab in additional_data[qs_name][m][2].keys():
                        cat = additional_data[qs_name][m][2][tab]
                        if cat == '':
                            cat = 'Others'
                        if cat not in cats.keys():
                            tabs[tab][cat] = np.array([head_line])
                    else:
                        cat = 'Others'
                    tabs[tab][cat] = np.append(tabs[tab][cat], [
                        [strategy_text, methods_text, ref_text]], axis=0)
        else:
            # it exists no example for qs_name
            methods_text = 'default'
            ref_text = ''
            temp = np.append(temp, [[strategy_text, methods_text, ref_text]],
                             axis=0)
            for tab, cats in tabs.items():
                if 'Others' not in cats.keys():
                    tabs[tab]['Others'] = np.array([head_line])
                tabs[tab]['Others'] = np.append(cats['Others'], [
                    [strategy_text, methods_text, ref_text]], axis=0)

    return tabs


def generate_examples(gen_path, package, json_path):
    """
    Creates all example scripts for the specified package and returns the data
    needed to create the strategy overview.

    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    package : module
        The '__init__' module of the package for which the examples should be
        created.
    json_path : string
        The path of the directory where to find the json example files for the
        specified package.

    Returns
    -------
    dict : Holds the data needed to create the strategy overview. Shape:
        {'strategy_1' : [
            ['method_1', ['reference_1', ...], {'tab_1': 'category', ...}],
            ['method_2', ...],
            ...],
        'strategy_2' : [...],
        ...
        }
    """
    dir_path_package = os.path.join(gen_path,
                                    'examples', package.__name__.split('.')[1],
                                    '')

    # create directory if it does not exist.
    os.makedirs(dir_path_package, exist_ok=True)

    # create README.rst files needed for 'sphinx-gallery'
    with open(os.path.join(gen_path, 'examples', 'README.rst'), 'w') as file:
        file.write('Example Gallery\n')
        file.write('===============')
    with open(os.path.join(dir_path_package, 'README.rst'), 'w') as file:
        title = f'{package.__name__.split(".")[1]} Example Gallery'.title()
        file.write(title + '\n')
        file.write(''.ljust(len(title), '='))

    additional_data = {}
    # iterate over jason example files
    for filename in os.listdir(json_path):
        if filename not in ["_uncertainty.json", "_alce.json"]:
            continue
        with open(os.path.join(json_path, filename)) as file:
            # iterate over the examples in the json file
            for data in json.load(file):
                plot_filename = 'plot_' + data["class"]
                # add the data to the 'examples_data' variable
                # needed to create the strategy overview
                if "method" in data.keys():
                    method = data['method']
                else:
                    method = 'default'
                plot_filename += "_" + method

                if not data['class'] in additional_data.keys():
                    additional_data[data['class']] = []

                if 'categories' in data.keys():
                    additional_data[data['class']].append(
                        [method, data['refs'], data['categories']])
                else:
                    additional_data[data['class']].append(
                        [method, data['refs'], {}])

                # create the example python script
                generate_example_script(plot_filename + '.py',
                                        dir_path_package, data, package)
    return additional_data


def generate_example_script(filename, dir_path, data, package):
    """
    Generates a python example file needed, for the 'sphinx-gallery' extension.

    Parameters
    ----------
    filename : string
        The name of the python example file
    dir_path : string
        The directory path in which to save the python example file.
    data : dict
        The data from the jason example file for the example.
    package : module
        The '__init__' module of the package for which the examples should be
        created.
    """
    # create directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    # Validation of 'data'.
    if data['class'] not in package.__all__:
        raise ValueError(f'"{data["class"]}" is not in "{package}.__all__".')

    first_title = True
    # Create the file.
    with open(os.path.join(dir_path, filename), 'w') as file:
        code_blocks = []
        # Iterate over the 'blocks' and generate the corresponding strings
        # expected from sphinx-gallery.
        for block in data["sequence"]:
            if block.startswith("title"):
                block_str = format_title(data[block], first_title)
                first_title = False
            elif block.startswith("subtitle"):
                block_str = format_title(data[block], False)
            elif block.startswith("text"):
                block_str = format_text(data[block])
            elif block.startswith("code"):
                code_blocks.append(data[block])
                block_str = format_code(data[block])
            elif block.startswith("plot"):
                if "clf" in data.keys():
                    clf = data["clf"]
                else:
                    clf = None
                qs = getattr(package, data["class"])
                block_str = format_plot(qs, data["init_params"],
                                        data["query_params"], clf=clf)
            elif block.startswith("refs"):
                block_str = format_refs(data[block])

            # Write the corresponding string to the python script.
            file.write(block_str)

        file.write("\n")


def format_title(title, first_title):
    """
    Generates the string for a title of the example page, formatted for a
    'sphinx-gallery' example script.

    Parameters
    ----------
    title : string
        The title string.
    first_title : boolean
        Indicates whether the title is the first title of the example script.

    Returns
    -------
    string : The formatted string for the example script.
    """
    if first_title:
        block_str = '"""\n' \
                    '' + title + '\n' \
                                 ''.ljust(len(title) + 1, '=') + '\n' \
                                                                 '"""\n'
    else:
        block_str = '# %%\n' \
                    '# .. rubric:: ' + title + ':\n'

    return block_str + '\n'


def format_subtitle(title):
    """
    Generates the string for a subtitle of the example page, formatted for a
    'sphinx-gallery' example script.

    Parameters
    ----------
    title : string
        The subtitle string.

    Returns
    -------
    string : The formatted string for the example script.
    """
    block_str = '# %%\n' \
                '# ' + title + '\n' \
                               '# '.ljust(len(title) + 1, '-') + '\n\n'
    return block_str


def format_text(text):
    """
    Generates the string for a paragraph of the example page, formatted for a
    'sphinx-gallery' example script.

    Parameters
    ----------
    text : string
        The paragraph text.

    Returns
    -------
    string : The formatted string for the example script.
    """
    block_str = '# %%\n'
    for line in text.split('\n'):
        block_str += '# ' + line + '\n'
    return block_str + '\n'


def format_code(code):
    """
    Generates the string for a code block of the example page, formatted for a
    'sphinx-gallery' example script.

    Parameters
    ----------
    code : string
        The python source code to be formatted.

    Returns
    -------
    string : The formatted string for the example script.
    """
    block_str = ''
    for line in code:
        block_str += line + '\n'
    return block_str + '\n'


def format_plot(qs, init_params, query_params, clf=None):
    """
    Generates the string for the plotting section of the example page,
    formatted for a 'sphinx-gallery' example script.

    Parameters
    ----------
    qs : QueryStrategy
        The query strategy.
    init_params : dict
        Should include the parameters used to initialize the query strategy.
    query_params : dict
        Should include the parameters used to call the 'query' method of the
        query strategy.
    clf : string, optional (default=None)
        An initialisation of the classifier used to initialize the query
        strategy, if no clf is specified in 'init_params'.

    Returns
    -------
    string : The formatted string for the example script.
    """
    block_str = f'from skactiveml.pool import {qs.__name__}\n'
    block_str += 'import numpy as np\n'
    block_str += 'from matplotlib import pyplot as plt, animation\n'
    block_str += 'from sklearn.datasets import make_classification\n'
    # Decide which classifier to use, if clf is None.
    if clf is None:
        if 'clf' not in init_params.keys():
            clf = 'PWC(classes=[0, 1], random_state=random_state)'
            block_str += 'from skactiveml.classifier import PWC\n'
        else:
            clf = init_params['clf']
            init_params['clf'] = 'clf'
    block_str += 'from skactiveml.utils import MISSING_LABEL, is_unlabeled, ' \
                 'plot_2d_dataset\n'  # TODO use the new plotting function

    block_str += '\n'
    block_str += 'random_state = 0\n'
    block_str += '\n'
    block_str += '# Build a dataset.\n'
    block_str += 'X, y_true = make_classification(n_features=2, ' \
                 'n_redundant=0, random_state=random_state)\n'
    block_str += 'y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)\n'

    block_str += '# Initialise the classifier.\n'
    block_str += ('clf = ' + clf + '\n')

    if 'clf' not in init_params.keys() and 'clf' in \
            inspect.signature(qs.__init__).parameters:
        init_params['clf'] = 'clf'
    # Set the random state if it is not set in the json file.
    if 'random_state' not in init_params.keys() and 'random_state' in \
            inspect.signature(qs.__init__).parameters:
        init_params['random_state'] = 'random_state'
    # Initialise the query strategy.
    block_str += f'# Initialise the query strategy.\n'
    block_str += f'qs = {qs.__name__}({dict_to_str(init_params)})\n'
    block_str += f'\n'
    block_str += f'# Preparation for plotting.\n'
    block_str += f'fig, ax = plt.subplots()\n'
    block_str += f'artists = []\n'
    block_str += f'\n'
    block_str += f'# The active learning cycle:\n'
    block_str += f'n_cycles = 20\n'
    block_str += f'for c in range(n_cycles):\n'
    block_str += f'    # Set X_cand to the unlabeled instances.\n'
    block_str += f'    unlbld_idx = np.where(is_unlabeled(y))[0]\n'
    block_str += f'    X_cand = X[unlbld_idx]\n'
    # Call the query method with the expected parameters.
    query_params_str = ''
    if 'X' in inspect.signature(qs.query).parameters:
        query_params_str += ', X'
    if 'y' in inspect.signature(qs.query).parameters:
        query_params_str += ', y'
    params = dict_to_str(query_params)
    if query_params_str != '' and params != '':
        query_params_str += ', '
    query_params_str += params
    block_str += f'    # Query the next instance/s.\n'
    block_str += f'    query_idx = unlbld_idx[qs.query(X_cand' \
                 f'{query_params_str})]\n'
    block_str += f'    # Plot the labeled data.\n'
    block_str += f'    collections = np.array(ax.collections)\n'
    block_str += f'    coll = plot_2d_dataset(X, y, y_true, clf, ' \
                 f'qs, ax)\n'
    block_str += f'    artists.append([x for x in coll if (x not ' \
                 f'in collections)])\n'
    block_str += f'    # Label the queried instances.\n'
    block_str += f'    y[query_idx] = y_true[query_idx]\n'
    block_str += f'    # Fit the classifier.\n'
    block_str += f'    clf.fit(X, y)\n'
    block_str += f'\n'
    block_str += f'ani = animation.ArtistAnimation(fig, artists, blit=True)\n '

    return block_str + '\n'


def format_refs(refs: list):
    """
    Generates the string for a references of the example page, formatted for a
    'sphinx-gallery' example script.

    Parameters
    ----------
    refs : list of strings
        A list of references to bibliographic entries in the 'refs.bib' file.

    Returns
    -------
    string : The formatted string for the example script.
    """
    if not refs:
        return ''
    block_str = '# %%\n' \
                '# .. rubric:: References:\n' \
                '# \n' \
                '# The implementation of this strategy is based on '
    if len(refs) > 1:
        for i in range(len(refs) - 1):
            block_str += f':footcite:t:`{refs[i]}`, '
        block_str = block_str[0:-2]
        block_str += f' and :footcite:t:`{refs[-1]}'
    else:
        block_str += f':footcite:t:`{refs[0]}'
    block_str += '`.\n' \
                 '#\n'

    block_str += '# .. footbibliography::\n'

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
