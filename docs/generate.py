import distutils.dir_util
import importlib
import inspect
import json
import os
import re
import shutil
import warnings

import numpy as np
from pybtex.database import parse_file

import skactiveml

for module in skactiveml.__all__:
    importlib.import_module("skactiveml." + module)

warnings.filterwarnings("ignore")


def generate_api_reference_rst(gen_path):
    """Creates the api_reference.rst file in the specified gen_path.

    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    """
    path = os.path.join(os.path.basename(gen_path), "api_reference.rst")
    gen_path = os.path.join(os.path.basename(gen_path), "api")
    os.makedirs(os.path.abspath(gen_path), exist_ok=True)
    with open(path, "w") as file:
        file.write(".. _api_reference:\n")
        file.write("\n")
        file.write("=============\n")
        file.write("API Reference\n")
        file.write("=============\n")
        file.write("\n")
        file.write("This is an overview of the API.\n")
        file.write("\n")
        file.write(".. module:: skactiveml\n")
        file.write("\n")
        for item in skactiveml.__all__:
            if inspect.ismodule(getattr(skactiveml, item)):
                file.write(automodule(getattr(skactiveml, item)))


def automodule(module, level=0):
    """
    This function generates the restructured text for the api reference and the
     specified module.

    Parameters
    ----------
    module : python module
        Module for which to create the api reference.
    level : int, default=0
        This parameter is used for the recursive call of this function.

    Returns
    -------
        String : The restructured text
    """
    rst_str = ""
    modules = []
    classes = []
    functions = []
    constants = []

    for item in module.__all__:
        try:
            importlib.import_module(module.__name__ + "." + item)
        except ModuleNotFoundError:
            pass
        if inspect.ismodule(getattr(module, item)):
            modules.append(item)
        if inspect.isclass(getattr(module, item)):
            classes.append(item)
        if inspect.isfunction(getattr(module, item)):
            functions.append(item)
        if isinstance(getattr(module, item), object) and item.isupper():
            constants.append(item)

    title = f":mod:`{module.__name__}`:"
    rst_str += title + "\n"
    rst_str += "".ljust(len(title), "=") + "\n\n"

    rst_str += f".. automodule:: {module.__name__}\n"
    rst_str += f"    :no-members:\n"
    rst_str += f"    :no-inherited-members:\n\n"

    rst_str += f'.. currentmodule:: {module.__name__.split(".")[0]}\n\n'
    if classes:
        rst_str += f"Classes\n"
        rst_str += f"-------\n\n"

        rst_str += f".. autosummary::\n"
        rst_str += f"   :nosignatures:\n"
        rst_str += f"   :toctree: api\n"
        rst_str += f"   :template: class.rst\n\n"
        for item in classes:
            name = module.__name__
            if "skactiveml." in name:
                name = name.replace("skactiveml.", "")
            if name:
                name += "."
            name += item
            rst_str += f"   {name}" + "\n"
        rst_str += "\n"

    if functions:
        rst_str += f"Functions\n"
        rst_str += f"---------\n\n"

        rst_str += f".. autosummary::\n"
        rst_str += f"   :nosignatures:\n"
        rst_str += f"   :toctree: api\n"
        rst_str += f"   :template: function.rst\n\n"
        for item in functions:
            name = module.__name__
            if "skactiveml." in name:
                name = name.replace("skactiveml.", "")
            if name:
                name += "."
            name += item
            rst_str += f"   {name}" + "\n"
        rst_str += "\n"

    for item in modules:
        rst_str += automodule(getattr(module, item), level=level + 1)

    return rst_str


def generate_strategy_overview_rst(gen_path, json_data):
    """Creates the strategy_overview.rst file in the specified path.

    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    json_data : dict
        The data of the examples directory stored in a dictionary.
    """

    strategy_table = json_data_to_strategy_table(json_data, gen_path)

    # Generate file
    with open(os.path.join(gen_path, "strategy_overview.rst"), "w") as file:
        file.write("#################\n")
        file.write("Strategy Overview\n")
        file.write("#################\n")
        file.write("\n")

        file.write(
            f"This is an overview of all implemented active learning "
            f"strategies.\n"
        )
        file.write("\n")
        file.write(
            f"You can use the following checkboxes to filter the "
            f"tables below.\n"
        )
        file.write("\n")
        file.write(
            ".. raw:: html\n"
            "\n"
            '   <input type="checkbox" class="input-tag" '
            'value="regression">\n'
            '   <label>Regression</label>\n'
            '   <input type="checkbox" class="input-tag" '
            'value="classification">\n '
            '   <label>Classification</label>\n'
            '   <input type="checkbox" class="input-tag" '
            'value="multi-annotator">\n '
            '   <label>Multi-Annotator</label>\n'
            '   <input type="checkbox" class="input-tag" '
            'value="single-annotator">\n '
            "   <label>Single-Annotator</label>\n"
        )
        file.write("\n")

        # Iterate over the sections.
        for section_name, cats in strategy_table.items():
            file.write(
                " ".join([s.capitalize() for s in section_name.split(os.sep)])
                + "\n"
            )
            file.write("".ljust(len(section_name), "-") + "\n")
            file.write("\n")

            # Iterate over the examples.
            file.write(format_sections(cats))
            file.write("\n")

        file.write("References\n")
        file.write("----------\n")
        file.write(".. footbibliography::")
        file.write("\n")


def json_data_to_strategy_table(json_data, gen_path):
    strategy_table = {}
    head_line = ["Method", "Base Class", "Tags", "Reference"]
    rel_api_path = os.path.join(os.path.basename(gen_path), "api").replace(
        "\\", "/"
    )

    for section_name, section_items in json_data.items():
        table = np.ndarray(shape=(0, 5))
        for data in section_items["data"]:
            # Collect the data needed to generate the strategy overview.
            qs_name = data["class"]
            method = data["method"]
            package = getattr(skactiveml, data["package"])
            package_name = package.__name__.replace("skactiveml.", "")
            methods_text = (
                f":doc:`{method} </generated/sphinx_gallery_examples/"
                f"{package_name}/plot-{qs_name}-"
                f'{method.replace(" ", "_")}>`'
            )
            strategy_text = (
                f":doc:`{qs_name} </{rel_api_path}/"
                f"{package.__name__}.{qs_name}>`"
            )
            tags = " ".join(data["tags"])
            ref_text = ""
            for ref in data["refs"]:
                ref_text += f":footcite:t:`{ref}`, "
            ref_text = ref_text[0:-2]
            category = (
                data["category"]
                if "category" in data.keys() and data["category"] != ""
                else "Others"
            )

            table = np.append(
                table,
                [[methods_text, strategy_text, tags, ref_text, category]],
                axis=0,
            )

        # Sort the table alphabetically.
        table = sorted(table, key=lambda row: row[1])

        # Build the strategy table.
        strategy_table[section_name] = {}
        for i, row in enumerate(table):
            category = row[-1]
            if category not in strategy_table[section_name].keys():
                strategy_table[section_name][category] = np.array([head_line])
            strategy_table[section_name][category] = np.append(
                strategy_table[section_name][category], [row[:-1]], axis=0
            )

    return strategy_table


def format_sections(cats, indent=0):
    string = ""

    # Iterate over the categories in the current paper.
    for cat in sorted(cats):
        if cat != "Others":
            string += f"{cat}\n"
            string += "".ljust(len(cat), "~") + "\n"
            string += table_data_to_rst_table(
                cats[cat], header_lines=1, indent=indent
            )
    if "Others" in cats.keys():
        # 'Others' is the fallback, if no category is specified
        # in the json file
        string += "Others\n"
        string += "~~~~~~\n"
        string += table_data_to_rst_table(
            cats["Others"], header_lines=1, indent=indent
        )

    return string


def table_data_to_rst_table(
    a, caption="", widths=None, header_lines=0, indent=0
):
    """Generates a rst-table and returns it as a string.

    Parameters
    ----------
    a : array-like, shape=(columns, rows)
        Contains the data for the table..
    caption : str, optional (default='')
        The caption of the table.
    widths : string, optional (default=None)
        A list of relative column widths separated with comma or space
        or 'auto'.
    header_lines : int, optional (default=0)
        The number of rows to use in the table header.
    indent : int, optional (default=0)
        Number of spaces as indent in each line

    Returns
    -------
    string : reStructuredText list-table as String.
    """
    a = np.asarray(a)
    indents = "".ljust(indent, " ")
    table = (
        f"{indents}.. list-table:: {caption}\n"
        f"{indents}   :header-rows: {header_lines}\n"
    )
    if widths is None:
        table += "\n"
    elif widths == "auto":
        table += f"{indents}   :widths: auto\n\n"
    else:
        table += f"{indents}   :widths: {widths}\n\n"
    for row in a:
        table += f"{indents}   *"
        for column in row:
            table += " - " + str(column) + f"\n{indents}    "
        table = table[0 : -4 - indent]
    return table + "\n"


def generate_examples(gen_path, json_path, recursive=True):
    """
    Creates all example scripts for the specified package and returns the data
    needed to create the strategy overview.

    Parameters
    ----------
    gen_path : string
        The path of the main directory in which all generated files should be
        created.
    json_path : string
        The path of the directory where to find the json example files for the
        specified package.
    recursive : bool, default=True
        If True, examples for sub-packagers are also created.

    Returns
    -------
    dict : Holds the data needed to create the strategy overview.
    """

    # create directory if it does not exist.
    os.makedirs(gen_path, exist_ok=True)

    json_data = dict()
    # iterate over json example files
    for (root, dirs, files) in os.walk(json_path, topdown=True):
        if root.endswith('__pycache__'): continue
        if "README.rst" not in files and "README.txt" not in files:
            raise FileNotFoundError(
                f"No README.rst or README.txt found in \n" f'"{root}"'
            )

        sub_dir_str = root.replace(json_path, "").strip(os.sep)
        dst = os.path.join(gen_path, sub_dir_str)
        os.makedirs(dst, exist_ok=True)
        # Iterate over all files in 'root'.
        for filename in files:
            if filename.endswith(".json"):
                with open(os.path.join(root, filename)) as file:
                    # iterate over the examples in the json file
                    for data in json.load(file):
                        sub_package_dict = json_data
                        package_structure = sub_dir_str.split(".")
                        for sp in package_structure:
                            if sp not in sub_package_dict.keys():
                                sub_package_dict[sp] = dict()
                            sub_package_dict = sub_package_dict[sp]
                        if "data" not in sub_package_dict.keys():
                            sub_package_dict["data"] = list()

                        sub_package_dict["data"].append(data)
                        # create the example python script
                        plot_filename = (
                            "plot-"
                            + data["class"]
                            + "-"
                            + data["method"].replace(" ", "_")
                        )
                        generate_example_script(
                            filename=plot_filename + ".py",
                            dir_path=dst,
                            data=data,
                            package=getattr(skactiveml, data["package"]),
                            template_path=os.path.abspath(data["template"]),
                        )
            elif not filename.startswith("template"):
                if filename.endswith(".py") or filename.endswith(".ipynb"):
                    src = os.path.join(root, filename)
                    example_string = format_plot({}, src)
                    with open(os.path.join(dst, filename), "w") as file:
                        file.write(example_string)
                else:
                    # Copy all other files except for templates.
                    src = os.path.join(root, filename)
                    shutil.copyfile(src, os.path.join(dst, filename))

        if not recursive:
            break

    return json_data


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
        The data from the json example file for the example.
    package : module
        The '__init__' module of the package for which the examples should be
        created.
    template_path : path-like
        The path to the template file.
    """
    # create directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    # Validation of 'data'.
    if data["class"] not in package.__all__:
        raise ValueError(f'"{data["class"]}" is not in "{package}.__all__".')

    first_title = True
    # Create the file.
    with open(os.path.join(dir_path, filename), "w") as file:
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
        block_str = (
            '"""\n'
            "" + title + "\n"
            "".ljust(len(title) + 1, "=") + "\n"
            '"""\n'
        )
    else:
        block_str = "# %%\n" "# .. rubric:: " + title + ":\n"

    return block_str + "\n"


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
    block_str = (
        "# %%\n" "# " + title + "\n" "# ".ljust(len(title) + 1, "-") + "\n\n"
    )
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
    block_str = "# %%\n"
    for line in text.split("\n"):
        block_str += "# " + line + "\n"
    return block_str + "\n"


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
    block_str = ""
    for line in code:
        block_str += line + "\n"
    return block_str + "\n"


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
    pattern = (
        r'"""\$[^"""]*"""|"\$[^"&^\n]*"|' + r"'''\$[^''']*'''|'\$[^'&^\n]*'"
    )
    pattern_group = (
        r'"""\$([^"""]*)"""|"\$([^"&^\n]*)"|'
        + r"'''\$([^''']*)'''|'\$([^'&^\n]*)'"
    )
    with open(template_path, "r") as template:
        template_str = template.read()

        findings = re.findall(pattern, template_str)

        for finding in findings:
            r = re.match(pattern_group, finding)
            s = filter(lambda x: x is not None, r.groups()).__next__()
            splits = s.split("|")
            key = splits[0]

            if "FULLEXAMPLES" not in os.environ:
                if key == "n_samples":
                    data[key] = "10"
                elif key == "n_cycles":
                    data[key] = "2"
                elif key == "res":
                    data[key] = "3"
            if key in data.keys():
                if isinstance(data[key], list):
                    new_str = ""
                    for line in data[key]:
                        new_str += line + "\n"
                    new_str = new_str[0:-1]
                else:
                    new_str = data[key]
            else:
                if len(splits) > 1:
                    new_str = splits[1]
                else:
                    new_str = ""

            template_str = template_str.replace(finding, new_str)
    return template_str + "\n"


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
        return ""
    block_str = (
        "# %%\n"
        "# .. rubric:: References:\n"
        "# \n"
        "# The implementation of this strategy is based on "
    )
    if len(refs) > 1:
        for i in range(len(refs) - 1):
            block_str += f":footcite:t:`{refs[i]}`, "
        block_str = block_str[0:-2]
        block_str += f" and :footcite:t:`{refs[-1]}"
    else:
        block_str += f":footcite:t:`{refs[0]}"
    block_str += "`.\n" "#\n"

    block_str += "# .. footbibliography::\n"

    return block_str + "\n"


def dict_to_str(d, idx=None, allocator="=", key_as_string=False):
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
            if not (
                (key.startswith('"') and key.endswith('"'))
                or (key.startswith("'") and key.endswith("'"))
            ):
                key = '"' + key + '"'
        dd_str += str(key) + allocator + value + ", "
    return dd_str[0:-2]


def generate_tutorials(src_path, dst_path):
    """Includes the tutorials folder from the git root, such that tutorials are
    included in the documentation. Effectively this function copies all
    contents from src_path to dst_path.
    Parameters
    ----------
    src_path: string
        The path where the notebooks are found.
    dst_path: string
        The path where the notebooks are saved, such that tutorials.rst can
        find them.
    """
    distutils.dir_util.copy_tree(src=src_path, dst=dst_path)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def export_legend(handles, labels, ax, path="legend.pdf", expand=None):
    if expand is None:
        expand = [-5, -5, 5, 5]

    ax.axis("off")
    legend = ax.legend(handles, labels,
                       loc=3,
                       framealpha=1,
                       frameon=True,
                       ncol=4,
                       mode="expand",
                       bbox_to_anchor=(0., 0., 1., 1.),
                       fontsize=8,
                       )

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.abspath(path), dpi="figure", bbox_inches=bbox)


def generate_stream_classification_legend(path):
    handles = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    labels = []

    labels.append("Sample Of Class 0")
    handles.append(ax.plot([], [], color="blue")[0])
    labels.append("Sample Of Class 1")
    handles.append(ax.plot([], [], color="red")[0])
    labels.append("Unlabled Sample")
    handles.append(ax.plot([], [], color="grey")[0])
    labels.append("Current Sample")
    handles.append(ax.plot([], [], color="grey", linewidth=3)[0])
    labels.append("Decision Boundary")
    handles.append(ax.plot([], [], color="black")[0])

    export_legend(handles, labels, ax, path=path)