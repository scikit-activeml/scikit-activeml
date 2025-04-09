import os

import packaging.version
import importlib
import inspect
import json
import os
import re
import shutil
import warnings
import copy

import numpy as np
from pybtex.database import parse_file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from joblib import Parallel, delayed
import git

import skactiveml

for module in skactiveml.__all__:
    importlib.import_module("skactiveml." + module)

warnings.filterwarnings("ignore")


def generate_api_reference_rst(gen_path):
    """Creates the api_reference.rst file in the specified gen_path.

    Parameters
    ----------
    gen_path : str
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
        str : The restructured text
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

    title = f":mod:`{module.__name__}`"
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
    gen_path : str
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
            f"strategies, which are often divided into three main "
            "categories based on the utilities they compute for sample "
            "selection:\n\n"
            "1. **Informativeness-based** strategies mostly select samples "
            "for which the model is most uncertain (e.g., via "
            "information-theoretic measures).\n\n"
            "2. **Representativeness-based** strategies select samples that "
            "capture the overall data distribution (e.g., via clustering or"
            "density estimation).\n\n"
            "3. **Hybrid** strategies combine both criteria to select "
            "samples that are informative and representative.\n\n"
        )
        file.write("\n")
        file.write(
            "Furthermore, we distinguish between **regression** and "
            "**classification** as supervised learning tasks, where labels can"
            "be provided by a **single annotator** or **multiple annotators**. "
            "You can use the checkboxes below to filter the query strategies "
            "based on these distinctions.\n"
        )
        file.write("\n")
        file.write(
            ".. raw:: html\n"
            "\n"
            '   <input type="checkbox" class="input-tag" '
            'value="regression">\n'
            "   <label>Regression</label>\n"
            '   <input type="checkbox" class="input-tag" '
            'value="classification">\n '
            "   <label>Classification</label>\n"
            '   <input type="checkbox" class="input-tag" '
            'value="multi-annotator">\n '
            "   <label>Multi-Annotator</label>\n"
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
        Contains the data for the table.
    caption : str, optional (default='')
        The caption of the table.
    widths : str, optional (default=None)
        A list of relative column widths separated with comma or space
        or 'auto'.
    header_lines : int, optional (default=0)
        The number of rows to use in the table header.
    indent : int, optional (default=0)
        Number of spaces as indent in each line.

    Returns
    -------
    str : reStructuredText list-table as String.
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


def generate_examples(
        gen_path,
        json_path,
        example_notebook_directory,
        recursive=True
):
    """
    Creates all example scripts for the specified package and returns the data
    needed to create the strategy overview.

    Parameters
    ----------
    gen_path : str
        The path of the main directory in which all generated files should be
        created.
    json_path : str
        The path of the directory where to find the json example files for the
        specified package.
    example_notebook_directory: str
        The path to the directory where the notebooks are saved.
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
    for root, dirs, files in os.walk(json_path, topdown=True):
        if root.endswith("__pycache__"):
            continue
        if "README.rst" not in files and "README.txt" not in files:
            raise FileNotFoundError(
                f"No README.rst or README.txt found in \n" f'"{root}"'
            )

        sub_dir_str = root.replace(json_path, "").strip(os.sep)
        dst = os.path.join(gen_path, sub_dir_str)
        os.makedirs(dst, exist_ok=True)
        # Iterate over all files in 'root'.
        num_cpus = -1
        if "num_cpus" in os.environ:
            num_cpus = int(os.environ["num_cpus"])
        if num_cpus != 1:
            json_data_lists = Parallel(n_jobs=num_cpus, backend="loky")(
                (
                    delayed(_generate_single_example)(
                        filename=filename,
                        root=root,
                        local_dir_path=sub_dir_str,
                        dst=dst,
                        notebook_directory=example_notebook_directory
                    )
                    for filename in files
                )
            )
        else:
            json_data_lists = []
            for filename in files:
                json_data_lists.append(
                    _generate_single_example(
                        filename=filename,
                        root=root,
                        local_dir_path=sub_dir_str,
                        dst=dst,
                        notebook_directory=example_notebook_directory
                    )
                )
        for json_data_list in json_data_lists:
            package_structure = sub_dir_str.split(os.sep)
            json_data_entry = json_data
            for sp in package_structure:
                if sp not in json_data_entry.keys():
                    json_data_entry[sp] = dict()
                json_data_entry = json_data_entry[sp]
            if "data" not in json_data_entry.keys():
                json_data_entry["data"] = list()
            json_data_entry["data"].extend(json_data_list)

        if not recursive:
            break
    return json_data


def _generate_single_example(
        filename,
        root,
        local_dir_path,
        dst,
        notebook_directory
        ):
    """_summary_

    Parameters
    ----------
    filename : str
        The path to the json file for which an example is generated.
    root : str
        The root directory where the json file is stored.
    local_dir_path : str
        The directory relative from the root directory.
    dst : str
        The root directory where the examples are saved.
    notebook_directory: str
        The path to the directory where the notebooks are saved.
    """
    data_list = []
    if filename.endswith(".json"):
        with open(os.path.join(root, filename)) as file:
            # iterate over the examples in the json file
            for data in json.load(file):
                data_list.append(data)
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
                    local_dir_path=local_dir_path,
                    data=data,
                    package=getattr(skactiveml, data["package"]),
                    template_path=os.path.abspath(data["template"]),
                    notebook_directory=notebook_directory
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
    return data_list


def generate_example_script(
        filename,
        dir_path,
        local_dir_path,
        data,
        package,
        template_path,
        notebook_directory,
        google_colab_link=None
):
    """
    Generates a python example file needed, for the 'sphinx-gallery' extension.

    Parameters
    ----------
    filename : str
        The name of the python example file
    dir_path : str
        The directory path in which to save the python example file.
    local_dir_path : str
        The directory relative from the root directory.
    data : dict
        The data from the json example file for the example.
    package : module
        The '__init__' module of the package for which the examples should be
        created.
    template_path : path-like
        The path to the template file.
    notebook_directory: str
        The path to the directory where the notebooks are saved.
    google_colab_link: str or None, default=None
        The link to google colab that can be used to open notebooks directly in
        google colab.
    """
    # create directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    # Validation of 'data'.
    if data["class"] not in package.__all__:
        raise ValueError(f'"{data["class"]}" is not in "{package}.__all__".')

    google_colab_link = check_google_colab_link(google_colab_link)

    notebook_filename = filename.replace('.py', '.ipynb')

    data["colab_link"] = "/".join([
        google_colab_link,
        notebook_directory,
        local_dir_path,
        notebook_filename
    ]

    )

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
    title : str
        The title string.
    first_title : boolean
        Indicates whether the title is the first title of the example script.

    Returns
    -------
    str : The formatted string for the example script.
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
    title : str
        The subtitle string.

    Returns
    -------
    str : The formatted string for the example script.
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
    text : str
        The paragraph text.

    Returns
    -------
    str : The formatted string for the example script.
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
    code : str
        The python source code to be formatted.

    Returns
    -------
    str : The formatted string for the example script.
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
        The data from the json example file for the example.
    template_path : path-like
        The path to the template file.
    Returns
    -------
    str : The formatted string for the example script.
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
                    data[key] = "50"
                elif key == "n_cycles":
                    data[key] = "5"
                elif key == "res":
                    data[key] = "8"
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
    str : The formatted string for the example script.
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
    allocator: str, optional (Default='=')
        The allocator is used to separate the key and the value.

    Returns
    -------
    str : dict_ as String. Shape: 'key1=value[idx[key1]], key2...' or
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


def generate_tutorials(src_path, dst_path, dst_path_colab):
    """Includes the tutorials folder from the git root, such that tutorials are
    included in the documentation. Effectively this function copies all
    contents from src_path to dst_path.
    Parameters
    ----------
    src_path: str
        The path where the notebooks are found.
    dst_path: str
        The path where the notebooks are saved, such that tutorials.rst can
        find them.
    dst_path_colab: str
        The path where the notebooks are saved, such that tutorials.rst can
        find them. This path is specially used to save the versions of the
        notebook that are linked to Google Colab.
    """
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    if os.path.exists(dst_path_colab):
        shutil.rmtree(dst_path_colab)
    shutil.copytree(src=src_path, dst=dst_path)
    shutil.copytree(src=src_path, dst=dst_path_colab)
    post_process_tutorials(
        dst_path,
        colab_notebook_path=dst_path_colab,
        show_installation_code=False
    )
    post_process_tutorials(
        dst_path_colab,
        colab_notebook_path=dst_path_colab,
        show_installation_code=True
    )


def post_process_tutorials(
        tutorials_path,
        colab_notebook_path,
        show_installation_code=False,
        google_colab_link=None
):
    """This function allows to post-process the tutorial notebooks. In
    particular, the placeholder (<colab_link>) within notebooks are replaced
    with the actual link to open this notebook within Google colab and the
    comments before pip and jupyter installation instructions are removed for
    the Google Colab versions.

    Parameters
    ----------
    tutorials_path: str
        The folder where the files should be modified.
    colab_notebook_path: str
        The folder where the colab notebooks are saved.
    show_installation_code: boolean, default=False
        If True, the pip and jupypter installation lines are shown. If False,
        these instructions are commented out
    google_colab_link: str or None, default=None
        The link to google colab that can be used to open notebooks directly in
        google colab.
    """
    tutorials = [f for f in os.listdir(tutorials_path) if f.endswith(".ipynb")]
    for file_name in tutorials:
        file_path = f"{tutorials_path}/{file_name}"
        file_path_colab = f"{colab_notebook_path}/{file_name}"

        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
        except OSError:
            file_content = None

        if file_content is not None:
            processed_file_content = copy.copy(file_content)
            processed_file_content = replace_colab_link(
                processed_file_content,
                file_path_colab,
                google_colab_link
            )
            if show_installation_code:
                processed_file_content = uncomment_installation_code(
                    processed_file_content
                )

            if file_content != processed_file_content:
                try:
                    with open(file_path, 'w') as f:
                        f.write(processed_file_content)
                except OSError:
                    print("Error while writing {}")
                    pass


def replace_colab_link(
        file_content,
        colab_path,
        google_colab_link=None
):
    """This function replaces the placeholder (<colab_link>) within
    `file_content` with the link that matches the location once the notebook is
    included into the deployed documentation.

    Parameters
    ----------
    file_content: str
        The content of the jupyter notebook.
    colab_path: str
        The relative path to the colab notebook.
    google_colab_link_prefix: str, default=None
        The Google Colab address where you can specify the notebook to open in
        Google Colab. If None, it is assumed that the official scikit-activeml
        documentation is used.

    Returns
    -------
    output : str
        The notebook that includes the Google Colab link if there was a
        placeholder.
    """
    google_colab_link = check_google_colab_link(google_colab_link)
    colab_link = f"{google_colab_link}/{colab_path}"
    output = re.sub(
        pattern="<colab_link>",
        repl=colab_link,
        string=file_content
    )
    return output


def check_google_colab_link(google_colab_link):
    """This function checks if `google_colab_link` is a string. If it is, it is
    returned as is. If it is `None`, a valid string that points to the official
    scikit-activeml documentation is returned.

    Parameters
    ----------
    google_colab_link : str or None, default=None
        The Google Colab address where you can specify the notebook to open in
        Google Colab. If None, it is assumed that the official scikit-activeml
        documentation is used.

    Returns
    -------
    output : str
        Returns the string that was provided if it was not None. If it is None,
        the string that points to the official scikit-activeml documentation
        is returned
    """
    output = google_colab_link
    if google_colab_link is None:
        colab_github = 'https://colab.research.google.com/github'
        docs_repo_name = 'scikit-activeml/scikit-activeml.github.io'
        docs_branch_path = 'blob/gh-pages/latest'
        output = (
            f"{colab_github}/{docs_repo_name}/{docs_branch_path}"
        )
    return output


def uncomment_installation_code(file_content):
    """This function removes the comment symbols for pip install and jupyter
    nbextension install commands.

    Parameters
    ----------
    file_content: str
        The content of the jupyter notebook.

    Returns
    -------
    output : str
        The notebook that would install the needed packages.
    """
    pattern = r'\"# (!pip install .*?)\"'
    repl = r'"\1"'
    output = re.sub(
        pattern=pattern,
        repl=repl,
        string=file_content
    )

    pattern = r'\"# (!jupyter nbextension install .*?)\"'
    repl = r'"\1"'
    output = re.sub(
        pattern=pattern,
        repl=repl,
        string=output
    )
    return output


def export_legend(handles, labels, ax, path="legend.pdf", expand=None):
    if expand is None:
        expand = [-5, -5, 5, 5]

    ax.axis("off")
    legend = ax.legend(
        handles,
        labels,
        loc=3,
        framealpha=1,
        frameon=True,
        ncol=4,
        mode="expand",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        fontsize=8,
    )

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.abspath(path), dpi="figure", bbox_inches=bbox)
    plt.close(fig)


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


def generate_classification_legend(path):
    handles = []
    labels = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

    labels.append("Decision Boundary")
    handles.append(ax.plot([], [], color="black")[0])
    labels.append("Labeled Sample")
    handles.append(
        ax.scatter([], [], marker=".", color="gray", s=100, alpha=0.8)
    )
    labels.append("Sample Of Class 0")
    handles.append(ax.scatter([], [], marker=".", color="blue"))
    labels.append("Sample Of Class 1")
    handles.append(ax.scatter([], [], marker=".", color="red"))
    labels.append("75% Confidence Class 0")
    handles.append(ax.plot([], [], color="blue", ls="--")[0])
    labels.append("75% Confidence Class 1")
    handles.append(ax.plot([], [], color="red", ls="--")[0])
    labels.append("High Utility Score")
    handles.append(mpatches.Patch(color="green", alpha=1.0))
    labels.append("Low Utility Score")
    handles.append(mpatches.Patch(color="green", alpha=0.2))

    export_legend(handles, labels, ax, path=path)


def generate_regression_legend(path):
    handles = []
    labels = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

    handles.append(ax.plot([], [], color="black")[0])
    labels.append("Regression Curve")
    handles.append(ax.plot([], [], color="green")[0])
    labels.append("Utilitiy Score")
    handles.append(ax.scatter([], [], marker=".", color="lightblue", s=100))
    labels.append("Unlabeled Sample")
    handles.append(ax.scatter([], [], marker=".", color="orange", s=100))
    labels.append("Labeled Sample")

    export_legend(handles, labels, ax, path=path)


def generate_switcher(
        repo_path=None,
        switcher_location=None,
        blacklisted_versions=None
):
    """Creates the version switcher file used by the PyDate theme.

    Parameters
    ----------
    repo_path : str or None, default=None
        The path to the repository root. If this parameter is None, ".." is
        used instead.
    switcher_location : str or None, default=None
        The path where the switcher json is saved to. If this parameter is
        None, "_static/switcher.json" is used instead.
    blacklisted_versions : list of str or None, default=None
        A list of versions which should be ignored for the switcher. If this
        parameter is None, no versions are ignored.
    """
    if repo_path is None:
        repo_path = ".."

    if switcher_location is None:
        switcher_location = "_static/switcher.json"

    print(f"current path: {os.path.abspath('.')}")
    print(f"repository path: {os.path.abspath(repo_path)}")
    repo = git.Repo(repo_path)
    tag_list_str = repo.git.ls_remote("--tags", "origin")
    versions_str = re.findall(r"\trefs\/tags\/(\d+.\d+.\d)", tag_list_str)
    sorted_versions = sorted(versions_str, key=packaging.version.Version)

    print(f"Found versions: {sorted_versions}")
    # remove versions which are not accessible
    if blacklisted_versions is not None:
        print(f"Versions to remove: {blacklisted_versions}")
        for blacklisted_version in blacklisted_versions:
            if blacklisted_version in sorted_versions:
                sorted_versions.remove(blacklisted_version)

    print(f"Versions to create switcher for: {sorted_versions}")
    switcher_text = create_switcher_text(sorted_versions)
    with open(switcher_location, "w") as f:
        for item in switcher_text:
            f.write(item)


def create_switcher_text(versions, docs_link=None):
    """This function generates the content for the switcher json file.

    Parameters
    ----------
    versions : list of str
        A list of versions for which documentations are saved.
    docs_link : str, default=None
        The address to the documentation. If None, the address for the official
        documentation is used instead.

    Returns
    -------
    list of str
        The content of the switcher json file separated by line
    """
    versions_short = [".".join(version.split(".")[:2]) for version in versions]
    unique_versions, unique_index, unique_counts = np.unique(
        versions_short, return_index=True, return_counts=True
    )
    versions_highest = np.array(versions)[unique_index + unique_counts - 1]
    if docs_link is None:
        docs_link = "https://scikit-activeml.github.io"
    # Create an entry for every version
    content_list = []
    content_list.append("[\n")
    content_list.append("  {\n")
    content_list.append('    "name": "latest",\n')
    content_list.append('    "version": "latest",\n')
    content_list.append(f'    "url": "{docs_link}/latest/"\n')
    content_list.append("  },\n")
    content_list.append("  {\n")
    content_list.append('    "name": "development",\n')
    content_list.append('    "version": "development",\n')
    content_list.append(f'    "url": "{docs_link}/development/"\n')
    content_list.append("  },\n")
    for ver, ver_s in zip(versions_highest[::-1], unique_versions[::-1]):
        content_list.append("  {\n")
        content_list.append(f'    "name": "{ver_s}",\n')
        content_list.append(f'    "version": "{ver}",\n')
        content_list.append(f'    "url": "{docs_link}/{ver_s}/"\n')
        content_list.append("  },\n")
    content_list[-1] = "  }\n"
    content_list.append("]")
    return content_list
