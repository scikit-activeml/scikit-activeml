import inspect
import json
import os
from pybtex.database import parse_file

import numpy as np

import skactiveml
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
        file.write('.. _api_reference:\n')
        file.write('\n')
        file.write('=============\n')
        file.write('API Reference\n')
        file.write('=============\n')
        file.write('\n')
        file.write('This is an overview of the API.\n')
        file.write('\n')
        file.write('.. module:: skactiveml\n')
        file.write('\n')
        for item in skactiveml.__all__:
            if inspect.ismodule(getattr(skactiveml, item)):
                file.write(automodule(getattr(skactiveml, item)))


def automodule(module, level=0):
    rst_str = ''
    modules = []
    classes = []
    functions = []
    constants = []

    for item in module.__all__:
        if inspect.ismodule(getattr(module, item)):
            modules.append(item)
        if inspect.isclass(getattr(module, item)):
            classes.append(item)
        if inspect.isfunction(getattr(module, item)):
            functions.append(item)
        if isinstance(getattr(module, item), object) and item.isupper():
            constants.append(item)

    title = f':mod:`{module.__name__}`:'
    rst_str += title + '\n'
    rst_str += ''.ljust(len(title), '=') + '\n\n'

    rst_str += f'.. automodule:: {module.__name__}\n'
    rst_str += f'    :no-members:\n'
    rst_str += f'    :no-inherited-members:\n\n'

    rst_str += f'.. currentmodule:: {module.__name__.split(".")[0]}\n\n'
    if classes:
        rst_str += f'Classes\n'
        rst_str += f'-------\n\n'

        rst_str += f'.. autosummary::\n'
        rst_str += f'   :nosignatures:\n'
        rst_str += f'   :toctree: api\n'
        rst_str += f'   :template: class.rst\n\n'
        for item in classes:
            name = module.__name__
            if 'skactiveml.' in name:
                name = name.replace('skactiveml.', '')
            if name:
                name += '.'
            name += item
            rst_str += f'   {name}' + '\n'
        rst_str += '\n'

    if functions:
        rst_str += f'Functions\n'
        rst_str += f'---------\n\n'

        rst_str += f'.. autosummary::\n'
        rst_str += f'   :nosignatures:\n'
        rst_str += f'   :toctree: api\n'
        rst_str += f'   :template: function.rst\n\n'
        for item in functions:
            name = module.__name__
            if 'skactiveml.' in name:
                name = name.replace('skactiveml.', '')
            if name:
                name += '.'
            name += item
            rst_str += f'   {name}' + '\n'
        rst_str += '\n'

    for item in modules:
        rst_str += automodule(getattr(module, item), level=level + 1)

    return rst_str


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
    # Load bibtex database.
    bib_data = parse_file('refs.bib')

    # create directory if it does not exist.
    os.makedirs(os.path.join(gen_path, 'strategy_overview'), exist_ok=True)

    # Generate file
    with open(
            os.path.join(gen_path, 'strategy_overview.rst'),
            'w') as file:
        file.write('=================\n')
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
        for tab in examples_data.keys():
            file.write(f'   strategy_overview/strategy_overview-{tab}\n')

    # Iterate over the tabs.
    for tab, cats in examples_data.items():
        author = bib_data.entries[tab].persons["author"][0].last_names[0]
        path = os.path.join(gen_path,
                            'strategy_overview',
                            f'strategy_overview-{tab}.rst')
        with open(path, 'w') as file:
            title = f'Strategy Overview By {author}\n'
            file.write(title)
            file.write(''.ljust(len(title) + 1, '=') + '\n')
            file.write('\n')
            file.write(f'This is an overview of all implemented AL strategies. '
                       f'The strategies are categorized according to '
                       f':footcite:t:`{tab}`.\n')
            file.write('\n')
            file.write('Pool Strategies\n')
            file.write('---------------\n')
            file.write('\n')

            # Iterate over the categories in the current tab.
            for cat in sorted(cats):
                if cat != 'Others':
                    file.write(table_from_array(cats[cat], title=cat,
                                                section_level='~',
                                                header_rows=1, indent=0))
            if 'Others' in cats.keys():
                # 'Others' is the fallback, if no category is specified
                # in the json file
                file.write(table_from_array(cats['Others'], title='Others',
                                            section_level='~',
                                            header_rows=1, indent=0))
            file.write('References\n')
            file.write('----------\n')
            file.write('.. footbibliography::')
            # TODO stream
            # file.write('\n')
            # file.write('Stream Strategies:\n')
            # file.write('------------------\n')
            # file.write('\n')
            file.write('\n')


def table_from_array(a, title='', caption='', widths=None, header_rows=0,
                     section_level='-', indent=0):
    """Generates a rst-table and returns it as a string.

    Parameters
    ----------
    a : array-like, shape=(columns, rows)
        Contains the data for the table.
    title : string, optional (default='')
        The title of the table.
    caption : str, optional (default='')
        The caption of the table.
    section_level : str, optional (default='-')
        The rst section level of the title.
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
    table = title + '\n'
    table += ''.ljust(len(title), section_level) + '\n'
    table += f'{indents}.. list-table:: {caption}\n' \
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
    dict : Holds the data needed to create the strategy overview.
    """
    dir_path_package = os.path.join(gen_path, 'examples',
                                    package.__name__.split('.')[1], '')

    rel_api_path = os.path.join(
        os.path.basename(gen_path), 'api'
    ).replace('\\', '/')

    # create directory if it does not exist.
    os.makedirs(dir_path_package, exist_ok=True)

    # create README.rst files needed for 'sphinx-gallery'
    with open(os.path.join(gen_path, 'examples', 'README.rst'), 'w') as file:
        file.write('Examples\n')
        file.write('========')
    with open(os.path.join(dir_path_package, 'README.rst'), 'w') as file:
        title = f'{package.__name__.split(".")[1]} based'.title()
        file.write(title + '\n')
        file.write(''.ljust(len(title), '-'))

    head_line = ['Method', 'Base Class', 'Reference']
    examples_data = {}  # (Tab, Category, Collum, Row)
    table = np.ndarray(shape=(0, 4))
    # iterate over jason example files
    for filename in os.listdir(json_path):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(json_path, filename)) as file:
            # iterate over the examples in the json file
            for data in json.load(file):
                # Collect the data needed to generate the strategy overview.
                qs_name = data["class"]
                method = data['method']
                package_name = package.__name__.replace('skactiveml.', '')
                methods_text = \
                    f':doc:`{method} </generated/sphinx_gallery_examples/' \
                    f'{package_name}/plot_{qs_name}_' \
                    f'{method.replace(" ", "_")}>`'
                strategy_text = f':doc:`{qs_name} </{rel_api_path}/' \
                                f'{package.__name__}.{qs_name}>`'
                ref_text = ''
                for ref in data['refs']:
                    ref_text += f':footcite:t:`{ref}`, '
                ref_text = ref_text[0:-2]
                category = data['categories'] if 'categories' in data.keys() \
                    else {}
                table = np.append(
                    table,
                    [[methods_text, strategy_text, ref_text, category]],
                    axis=0
                )

                # create the example python script
                plot_filename = \
                    'plot_' + data["class"] + "_" + method.replace(' ', '_')
                generate_example_script(
                    filename=plot_filename + '.py',
                    dir_path=dir_path_package,
                    data=data,
                    package=package,
                    template_path=os.path.join(json_path, 'template.py')
                )

    # Sort the table alphabetically.
    table = sorted(table, key=lambda row: row[1])

    # Collect the different tabs and categories.
    for i, row in enumerate(table):
        for tab in row[3].keys():
            if tab not in examples_data.keys():
                examples_data[tab] = {}
            if row[3][tab] == '':
                table[i][3][tab] = 'Others'
            if row[3][tab] not in examples_data[tab].keys():
                examples_data[tab][row[3][tab]] = np.array([head_line])

    # Build the dict that holds the data.
    for row in table:
        for tab in examples_data:
            if tab in row[3].keys():
                cat = row[3][tab] if tab in row[3].keys() else 'Others'
            examples_data[tab][cat] = \
                np.append(examples_data[tab][cat], [row[:3]], axis=0)

    return examples_data


def generate_example_script(filename, dir_path, data, package, template_path):
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
    template_path : path-like
        The path to the template file.
    """
    # create directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    # Validation of 'data'.
    if data['class'] not in package.__all__:
        raise ValueError(f'"{data["class"]}" is not in "{package}.__all__".')
    data["qs"] = getattr(package, data["class"])

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
                block_str = format_plot(data, template_path)
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


def format_plot(data, template_path):
    """
    Generates the string for the plotting section of the example page,
    formatted for a 'sphinx-gallery' example script.

    Parameters
    ----------
    data : dict
        The data from the jason example file for the example.
    template_path : path-like
        The path to the template file.
    Returns
    -------
    string : The formatted string for the example script.
    """
    # Collect the expected parameters for the query method.
    if 'X' not in data['query_params'].keys() and \
            'X' in inspect.signature(data["qs"].query).parameters:
        data['query_params']['"X"'] = 'X'
    if 'y' not in data['query_params'].keys() and \
            'y' in inspect.signature(data["qs"].query).parameters:
        data['query_params']['"y"'] = 'y'

    block_str = ''
    with open(template_path, "r") as template:
        for line in template:
            if '#_' in line:
                if '#_ import' in line:
                    line = f'from skactiveml.pool import {data["qs"].__name__}\n'
                    if 'clf' not in data.keys() and \
                            'clf' not in data['init_params'].keys():
                        line += 'from skactiveml.classifier import PWC\n'
                elif 'init_clf' in line:
                    # Decide which classifier to use, if clf is None.
                    if 'clf' not in data.keys():
                        if 'clf' not in data['init_params'].keys():
                            clf = 'PWC(classes=[0, 1], random_state=random_state)'
                        else:
                            clf = data['init_params']['clf']
                    else:
                        clf = data['clf']
                    line = ('clf = ' + clf + '\n')
                elif 'init_qs' in line:
                    if 'clf' not in data['init_params'].keys() and 'clf' in \
                            inspect.signature(data["qs"].__init__).parameters:
                        data['init_params']['clf'] = 'clf'
                    # Set the random state if it is not set in the json file.
                    if 'random_state' not in data[
                        'init_params'].keys() and 'random_state' in \
                            inspect.signature(data["qs"].__init__).parameters:
                        data['init_params']['random_state'] = 'random_state'
                    # Initialise the query strategy.
                    line = f'qs = {data["qs"].__name__}' \
                           f'({dict_to_str(data["init_params"])})\n'
                elif '#_query_params' in line:
                    start = line.find('#_query_params') - 1
                    line = line[:start] + '{' + \
                           dict_to_str(data['query_params'], allocator=': ',
                                       key_as_string=True) + '}\n'
                elif '#_bp' in line:
                    try:
                        s = line.find('#_')
                        bp = line[s + 2:].split()[0]
                        prefix = line[:s]
                        line = ''
                        if type(data[bp]) != list:
                            data[bp] = [data[bp]]
                        for l in data[bp]:
                            line += prefix + l + '\n'
                    except KeyError:
                        pass

            block_str += line

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


def dict_to_str(d, idx=None, allocator='=', key_as_string=False):
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
    allocator: string, optional (Default='=')
        The allocator is used to separate the key and the value.

    Returns
    -------
    String : dict_ as String. Shape: 'key1=value[idx[key1]], key2...' or
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
        key = str(key)
        if key_as_string:
            if not ((key.startswith('"') and key.endswith('"')) or \
                    (key.startswith('\'') and key.endswith('\''))):
                key = '"' + key + '"'
        dd_str += str(key) + allocator + value + ", "
    return dd_str[0:-2]
