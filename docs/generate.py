import os
from collections import OrderedDict
from pathlib import Path
import black

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
from docutils.utils import column_width
from matplotlib.lines import Line2D
from pybtex.database import parse_file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from joblib import Parallel, delayed
import git

import skactiveml

for module in skactiveml.__all__:
    importlib.import_module("skactiveml." + module)

warnings.filterwarnings("ignore")


def is_skactiveml_method(class_name, method_name):
    """Return whether a class method is implemented by scikit-activeml."""
    module_name, class_name = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    method = inspect.getattr_static(cls, method_name)
    if isinstance(method, (classmethod, staticmethod)):
        method = method.__func__
    method_module = getattr(method, "__module__", "") or ""
    return method_module == "skactiveml" or method_module.startswith(
        "skactiveml."
    )


def get_skactiveml_methods():
    """Return native method names for every public scikit-activeml class."""
    methods_by_class = {}
    visited_modules = set()

    def collect_methods(module):
        if module.__name__ in visited_modules:
            return
        visited_modules.add(module.__name__)

        for item in module.__all__:
            if not hasattr(module, item):
                importlib.import_module(f"{module.__name__}.{item}")
            obj = getattr(module, item)
            if inspect.ismodule(obj):
                collect_methods(obj)
            elif inspect.isclass(obj):
                class_name = f"{module.__name__}.{item}"
                methods_by_class[class_name] = tuple(
                    method_name
                    for method_name in dir(obj)
                    if is_skactiveml_method(class_name, method_name)
                )

    collect_methods(skactiveml)
    return methods_by_class


# Group strategies by their API scenario, independently of the example's
# gallery directory or demonstrated prediction task.
OVERVIEW_SCENARIOS = OrderedDict(
    [
        ("pool", "Pool-based Active Learning"),
        ("stream", "Stream-based Active Learning"),
    ]
)

OVERVIEW_FILTERS = [
    (
        "task",
        "Task",
        [("classification", "Classification"), ("regression", "Regression")],
    ),
    (
        "target",
        "Targets",
        [("single-output", "Single-output"), ("multi-label", "Multi-label")],
    ),
    (
        "annotators",
        "Annotators",
        [
            ("single-annotator", "Single annotator"),
            ("multi-annotator", "Multiple annotators"),
        ],
    ),
    (
        "batch",
        "Pool Batching",
        [
            ("batch", "Batch-capable"),
            ("top-k-batch", "Top-k"),
            ("diverse-batch", "Diverse"),
            ("random-batch", "Random"),
            ("single-sample", "Single-sample only"),
        ],
    ),
]

OVERVIEW_BATCH_LABELS = {
    "top-k-batch": "Top-k batch",
    "diverse-batch": "Diverse batch",
    "random-batch": "Random batch",
    "single-sample": "Single-sample only",
}


def get_strategy_metadata(data):
    """Resolve a documented method's supported setups and pool batching.

    The API declares complete (task, target type, annotator type) tuples.
    Example tags restrict these to the documented method variant; they do
    not define additional combinations. Constructor parameters that change
    capabilities belong in ``capability_params``, independently of the code
    used to demonstrate the method in ``init_qs``.
    """
    package = importlib.import_module(f"skactiveml.{data['package']}")
    cls = getattr(package, data["class"])
    params = {
        name: None
        for name, parameter in inspect.signature(cls).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    conditional = cls in (
        skactiveml.pool.SubSamplingWrapper,
        skactiveml.pool.ParallelUtilityEstimationWrapper,
    )
    if conditional:
        # These wrappers preserve the wrapped strategy's target semantics.
        # A task-agnostic strategy exposes the target kinds they can preserve;
        # the overview explicitly notes the dependency on the wrapped method.
        params["query_strategy"] = skactiveml.pool.RandomSampling()
    params.update(data.get("capability_params", {}))
    declared = cls(**params)._target_capabilities
    tags = set(data["tags"])
    capabilities = sorted(
        capability
        for capability in declared
        if capability[0] in tags
        and capability[2] in tags
        and (capability[1] != "multi-label" or "multi-label" in tags)
    )
    capability_tags = {
        "classification",
        "regression",
        "multi-label",
        "single-annotator",
        "multi-annotator",
    }
    supported_tags = {value for setup in capabilities for value in setup}
    if not capabilities or not (tags & capability_tags) <= supported_tags:
        raise ValueError(
            f"Unsupported capability tags for {data['method']!r}."
        )

    batch = sorted(tags & OVERVIEW_BATCH_LABELS.keys())
    scenario = data["package"].split(".")[0]
    if scenario == "stream" and batch:
        raise ValueError(
            f"Stream method {data['method']!r} has pool batch tags."
        )
    if scenario == "pool" and (
        not batch or ("single-sample" in batch and len(batch) != 1)
    ):
        raise ValueError(f"Specify pool batching for {data['method']!r}.")
    return {
        "api_class": f"{package.__name__}.{cls.__name__}",
        "capabilities": capabilities,
        "batch": batch,
        "conditional": conditional,
    }


EXAMPLE_SETUP_LABELS = {
    "pool-classification": "single-output classification",
    "pool-multilabel": "multi-label classification",
    "pool-regression": "regression",
    "pool-multi-annotator": "classification with multiple annotators",
    "stream-classification": "single-output classification",
}


def _example_setup(data):
    """Identify the demonstrated setup, independently of supported tasks."""
    template = Path(data.get("template", "")).stem
    if not template:
        return None
    scenario = data["package"].split(".")[0]
    if "regression" in template:
        setup = "regression"
    elif "multilabel" in template:
        setup = "multilabel"
    elif "multi_annotator" in template:
        setup = "multi-annotator"
    else:
        setup = "classification"
    return f"{scenario}-{setup}"


def _format_generated_python_script(file_path, line_length=79):
    """Formats a generated Python example script with Black."""
    black.format_file_in_place(
        Path(file_path),
        fast=False,
        mode=black.Mode(line_length=line_length),
        write_back=black.WriteBack.YES,
    )


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
    automodule(skactiveml, api_root_path=gen_path)


def automodule(module, api_root_path, level=0):
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
    header_str = ""
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
    header_str += title + "\n"
    header_str += "".ljust(len(title), "=") + "\n\n"

    header_str += f".. automodule:: {module.__name__}\n"
    header_str += f"    :no-members:\n"
    header_str += f"    :no-inherited-members:\n\n"

    header_str += f'.. currentmodule:: {module.__name__.split(".")[0]}\n\n'

    if modules:
        rst_str += f"Submodules\n"
        rst_str += f"----------\n\n"

        rst_str += f".. list-table::\n"
        rst_str += f"  :header-rows: 0\n"
        rst_str += f"  :widths: 100\n"
        rst_str += "\n"
        for item in modules:
            item_module_path = f"{module.__name__}.{item}"
            rst_str += f"  * - :mod:`{item_module_path}`\n"
        rst_str += "\n"

    if classes:
        rst_str += f"Classes\n"
        rst_str += f"-------\n\n"

        rst_str += f".. autosummary::\n"
        rst_str += f"   :nosignatures:\n"
        rst_str += f"   :toctree: .\n"
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
        rst_str += f"   :toctree: .\n"
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

    path = os.path.join(api_root_path, f"{module.__name__}.rst")
    if level == 0:
        with open(path, "w") as file:
            file.write("===\n")
            file.write("API\n")
            file.write("===\n")
            file.write("\n")
            file.write("This is an overview of the API.\n")
            file.write("\n")
            file.write(".. toctree::\n")
            file.write("  :titlesonly:\n")
            file.write("  :includehidden:\n")
            file.write("  :maxdepth: 4\n")

            file.write("\n")
            for item in modules:
                file.write(f"  {module.__name__}.{item}\n")
            file.write("\n")
            file.write("\n")
            file.write(".. module:: skactiveml\n")
    else:
        with open(path, "w") as file:
            file.write(header_str)
            file.write("\n")
            file.write(".. toctree::\n")
            file.write("  :hidden:\n")
            file.write("\n")
            for item in modules:
                file.write(f"  {module.__name__}.{item}\n")
            file.write("\n")
            file.write(rst_str)

    for item in modules:
        automodule(
            getattr(module, item), api_root_path=api_root_path, level=level + 1
        )


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

        file.write("|\n\n")
        file.write(
            ".. image:: ../logos/scikit-activeml-query-strategy-overview.svg\n"
        )
        file.write("   :class: dark-light\n")
        file.write("   :align: center\n")
        file.write("   :width: 100%\n\n")
        file.write("|\n\n")

        file.write(
            f"This is an overview of all implemented pool- and stream-based "
            f"active learning strategies, which are often divided into three "
            f"main categories based on the utilities they compute for sample "
            "selection:\n\n"
            "1. **Informativeness-based** strategies mostly select samples "
            "for which the model is most uncertain (e.g., via "
            "information-theoretic measures).\n\n"
            "2. **Representativeness-based** strategies select samples that "
            "capture the overall data distribution (e.g., via clustering or "
            "density estimation).\n\n"
            "3. **Hybrid** strategies combine both criteria to select "
            "samples that are informative and representative.\n\n"
        )
        file.write("\n")
        file.write(
            "Furthermore, we distinguish between **regression** and "
            "**classification** as supervised learning tasks. Classification "
            "targets may be **multi-label**, and labels can be provided by a "
            "**single annotator** or "
            "**multiple annotators**. Pool strategies can select batches "
            "using **top-k** utility scores, **diversity**, or **random "
            "selection**. Some methods select only one sample at a time. "
            "Stream strategies decide which arriving samples to label "
            "within a **budget**.\n\n"
            "Each table entry is a method, variant, or wrapped configuration. "
            "Filters describe the setups that entry supports. The linked "
            "visualization illustrates the example setup noted below its "
            "method name. API class names link to the implementation "
            "documentation.\n"
        )
        file.write("\n")
        file.write(
            ".. raw:: html\n"
            "\n"
            '   <section id="strategy-filters" class="strategy-filters" '
            'aria-labelledby="strategy-filter-title" '
            'aria-describedby="strategy-filter-hint" hidden>\n'
            '     <div class="strategy-filter-header">\n'
            '       <h2 id="strategy-filter-title" '
            'class="strategy-filter-title">Filter Strategies</h2>\n'
            '       <button type="button" id="clear-strategy-filters" '
            'class="strategy-filter-clear">Clear all</button>\n'
            "     </div>\n"
            '     <p id="strategy-filter-hint" '
            'class="strategy-filter-hint">Choose your intended setup, or '
            "leave a group on Any. Pool batching options apply to pool "
            "strategies. With multiple annotators, a batch contains "
            "sample–annotator pairs.</p>\n"
        )
        for key, group_name, options in OVERVIEW_FILTERS:
            file.write(
                '     <fieldset class="strategy-filter-group" '
                f'data-filter="{key}">\n'
                f"       <legend>{group_name}</legend>\n"
                '       <div class="strategy-filter-options">\n'
            )
            for value, label in [("", "Any")] + options:
                option_id = f"strategy-filter-{key}-{value or 'any'}"
                checked = " checked" if not value else ""
                file.write(
                    '         <label class="strategy-filter-option" '
                    f'for="{option_id}">\n'
                    '           <input type="radio" '
                    f'name="strategy-{key}" id="{option_id}" '
                    f'value="{value}"{checked}>\n'
                    f"           <span>{label}</span>\n"
                    "         </label>\n"
                )
            file.write("       </div>\n     </fieldset>\n")
        file.write(
            '     <p id="tag-filter-status" class="strategy-filter-status" '
            'role="status" aria-atomic="true"></p>\n'
            "   </section>\n"
        )
        file.write("\n")

        for scenario, table in strategy_table.items():
            title = OVERVIEW_SCENARIOS[scenario]
            file.write(f"{title}\n")
            file.write("-" * column_width(title) + "\n\n")
            file.write(
                table_data_to_rst_table(
                    table,
                    widths="36 23 1 25 15",
                    header_lines=1,
                    classes="strategy-table",
                )
            )
            file.write("\n")

        file.write("References\n")
        file.write("----------\n")
        file.write(".. footbibliography::")
        file.write("\n")


def json_data_to_strategy_table(json_data, gen_path):
    """Collect one table per scenario while preserving example links."""
    rows_by_scenario = {scenario: [] for scenario in OVERVIEW_SCENARIOS}
    # The hidden third column carries complete setups for browser filtering.
    head_line = [
        "Method",
        "API class",
        "Filter data",
        "Properties",
        "Reference",
    ]
    rel_api_path = os.path.join(os.path.basename(gen_path), "api").replace(
        "\\", "/"
    )

    for section_name, section_items in json_data.items():
        for data in section_items["data"]:
            metadata = get_strategy_metadata(data)
            # Collect the data needed to generate the strategy overview.
            qs_name = data["class"]
            method = data["method"]
            # Match the page title's capitalization without changing its URL.
            title = data.get("title", method)
            method_label = (
                title if title.casefold() == method.casefold() else method
            )
            package = skactiveml
            for subpackage in data["package"].split("."):
                package = getattr(package, subpackage)
            methods_text = (
                f":doc:`{method_label} </generated/sphinx_gallery_examples/"
                f"{section_name}/plot-{qs_name}-"
                f'{method.replace(" ", "_")}>`'
            )
            example_label = EXAMPLE_SETUP_LABELS.get(_example_setup(data), "")
            if example_label:
                methods_text += f"\n\n*Example: {example_label}.*"
            strategy_text = (
                f":doc:`{qs_name} </{rel_api_path}/"
                f"{package.__name__}.{qs_name}>`"
            )
            filter_data = f"``{json.dumps(metadata)}``"
            ref_text = ""
            for ref in data["refs"]:
                ref_text += f":footcite:t:`{ref}`, "
            ref_text = ref_text[0:-2]
            category = (
                data["category"]
                if "category" in data.keys() and data["category"] != ""
                else "Others"
            )

            properties = [f"**{category}**"]
            setups = metadata["capabilities"]
            target_labels = {
                (
                    "classification",
                    "single-output",
                ): "Single-output classification",
                (
                    "classification",
                    "multi-label",
                ): "Multi-label classification",
                ("regression", "single-output"): "Regression",
            }
            for setup, label in target_labels.items():
                if any(capability[:2] == setup for capability in setups):
                    properties.append(label)
            properties.extend(
                label
                for value, label in [
                    ("single-annotator", "Single annotator"),
                    ("multi-annotator", "Multiple annotators"),
                ]
                if any(capability[2] == value for capability in setups)
            )
            if data["class"] == "SubSamplingWrapper":
                properties.append("Batch rule depends on wrapped strategy")
            elif metadata["batch"]:
                properties.extend(
                    OVERVIEW_BATCH_LABELS[value] for value in metadata["batch"]
                )
            else:
                properties.append("Stream budget")
            if metadata["conditional"]:
                properties.append("Target support depends on wrapped strategy")
            if data.get("batch_note"):
                properties.append(data["batch_note"])
            properties_text = "\n".join(f"| {value}" for value in properties)
            scenario = data["package"].split(".")[0]
            rows_by_scenario[scenario].append(
                [
                    methods_text,
                    strategy_text,
                    filter_data,
                    properties_text,
                    ref_text,
                ]
            )

    # Sort by the displayed method, then the API class for duplicate names.
    return {
        scenario: [head_line]
        + sorted(
            rows,
            key=lambda row: (
                row[0].split(" <", 1)[0].casefold(),
                row[1].casefold(),
            ),
        )
        for scenario, rows in rows_by_scenario.items()
        if rows
    }


def table_data_to_rst_table(
    a, caption="", widths=None, header_lines=0, indent=0, classes=None
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
    classes : str or None, default=None
        Space-separated CSS classes for the table.

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
    if classes is not None:
        table += f"{indents}   :class: {classes}\n"
    if widths is None:
        table += "\n"
    elif widths == "auto":
        table += f"{indents}   :widths: auto\n\n"
    else:
        table += f"{indents}   :widths: {widths}\n\n"
    for row in a:
        table += f"{indents}   *"
        for column in row:
            content = str(column).replace("\n", f"\n{indents}       ")
            table += " - " + content + f"\n{indents}    "
        table = table[0 : -4 - indent]
    return table + "\n"


def generate_examples(
    gen_path,
    json_path,
    example_notebook_directory,
    recursive=True,
    version="latest",
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.

    Returns
    -------
    dict : Holds the data needed to create the strategy overview.
    """

    # create directory if it does not exist.
    os.makedirs(gen_path, exist_ok=True)

    json_data = OrderedDict()
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
                        notebook_directory=example_notebook_directory,
                        version=version,
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
                        notebook_directory=example_notebook_directory,
                        version=version,
                    )
                )
        for json_data_list in json_data_lists:
            package_structure = sub_dir_str.split(os.sep)
            json_data_entry = json_data
            for sp in package_structure:
                if sp not in json_data_entry.keys():
                    json_data_entry[sp] = OrderedDict()
                json_data_entry = json_data_entry[sp]
            if "data" not in json_data_entry.keys():
                json_data_entry["data"] = list()
            json_data_entry["data"].extend(json_data_list)

        if not recursive:
            break
    return json_data


def _generate_single_example(
    filename, root, local_dir_path, dst, notebook_directory, version="latest"
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.
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
                current_package = skactiveml
                for subpackage in data["package"].split("."):
                    current_package = getattr(current_package, subpackage)
                generate_example_script(
                    filename=plot_filename + ".py",
                    dir_path=dst,
                    local_dir_path=local_dir_path,
                    data=data,
                    package=current_package,
                    template_path=os.path.abspath(data["template"]),
                    notebook_directory=notebook_directory,
                    version=version,
                )
    elif not filename.startswith("template"):
        if filename.endswith(".py") or filename.endswith(".ipynb"):
            src = os.path.join(root, filename)
            example_string = format_plot({}, src)
            dst_path = os.path.join(dst, filename)
            with open(dst_path, "w") as file:
                file.write(example_string)
            if filename.endswith(".py"):
                _format_generated_python_script(dst_path, line_length=79)
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
    google_colab_link=None,
    version="latest",
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.
    """
    # create directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    # Validation of 'data'.
    if data["class"] not in package.__all__:
        raise ValueError(f'"{data["class"]}" is not in "{package}.__all__".')

    google_colab_link = check_google_colab_link(google_colab_link, version)

    notebook_filename = filename.replace(".py", ".ipynb")

    data["colab_link"] = "/".join(
        [
            google_colab_link,
            notebook_directory,
            local_dir_path,
            notebook_filename,
        ]
    )

    first_title = True
    file_path = os.path.join(dir_path, filename)
    # Create the file.
    with open(file_path, "w") as file:
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

    _format_generated_python_script(file_path, line_length=79)


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
                    data[key] = "20"
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


def generate_tutorials(src_path, dst_path, dst_path_colab, version="latest"):
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.
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
        show_installation_code=False,
        version=version,
    )
    post_process_tutorials(
        dst_path_colab,
        colab_notebook_path=dst_path_colab,
        show_installation_code=True,
        version=version,
    )


def post_process_tutorials(
    tutorials_path,
    colab_notebook_path,
    show_installation_code=False,
    google_colab_link=None,
    version="latest",
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.
    """
    tutorials = [f for f in os.listdir(tutorials_path) if f.endswith(".ipynb")]
    for file_name in tutorials:
        file_path = f"{tutorials_path}/{file_name}"
        file_path_colab = f"{colab_notebook_path}/{file_name}"

        try:
            with open(file_path, "r") as f:
                file_content = f.read()
        except OSError:
            file_content = None

        if file_content is not None:
            processed_file_content = copy.copy(file_content)
            processed_file_content = add_orphan_metadata(
                processed_file_content
            )
            processed_file_content = replace_colab_link(
                processed_file_content,
                file_path_colab,
                google_colab_link,
                version,
            )
            if show_installation_code:
                processed_file_content = uncomment_installation_code(
                    processed_file_content
                )

            if file_content != processed_file_content:
                try:
                    with open(file_path, "w") as f:
                        f.write(processed_file_content)
                except OSError:
                    print("Error while writing {}")
                    pass


def add_orphan_metadata(file_content):
    """This function adds the orphan metadata that nbsphinx require so that the
    notebooks don't need to appear in any toctree.

    Parameters
    ----------
    file_content: str
        The content of the jupyter notebook.

    Returns
    -------
    output : str
        The notebook with the added metadata.
    """
    notebook_json = json.loads(file_content)
    nbsphinx_dict = notebook_json["metadata"].get("nbsphinx", {})
    nbsphinx_dict["orphan"] = True
    notebook_json["metadata"]["nbsphinx"] = nbsphinx_dict
    output = json.dumps(notebook_json)
    return output


def replace_colab_link(
    file_content, colab_path, google_colab_link=None, version="latest"
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
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.

    Returns
    -------
    output : str
        The notebook that includes the Google Colab link if there was a
        placeholder.
    """
    google_colab_link = check_google_colab_link(google_colab_link, version)
    colab_link = f"{google_colab_link}/{colab_path}"
    output = re.sub(
        pattern="<colab_link>", repl=colab_link, string=file_content
    )
    return output


def check_google_colab_link(google_colab_link, version="latest"):
    """This function checks if `google_colab_link` is a string. If it is, it is
    returned as is. If it is `None`, a valid string that points to the official
    scikit-activeml documentation is returned.

    Parameters
    ----------
    google_colab_link : str or None, default=None
        The Google Colab address where you can specify the notebook to open in
        Google Colab. If None, it is assumed that the official scikit-activeml
        documentation is used.
    version: str, default="latest"
        The version string of this documentation that is used to point toward
        the documentation of this version.

    Returns
    -------
    output : str
        Returns the string that was provided if it was not None. If it is None,
        the string that points to the official scikit-activeml documentation
        is returned
    """
    output = google_colab_link
    if google_colab_link is None:
        colab_github = "https://colab.research.google.com/github"
        docs_repo_name = "scikit-activeml/scikit-activeml.github.io"
        docs_branch_path = f"blob/gh-pages/{version}"
        output = f"{colab_github}/{docs_repo_name}/{docs_branch_path}"
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
    pattern = r"\"# (!pip install .*?)\""
    repl = r'"\1"'
    output = re.sub(pattern=pattern, repl=repl, string=file_content)

    pattern = r"\"# (!jupyter nbextension install .*?)\""
    repl = r'"\1"'
    output = re.sub(pattern=pattern, repl=repl, string=output)
    return output


def export_legend(handles, labels, ax, path="legend.pdf", expand=None, ncol=4):
    if expand is None:
        expand = [-5, -5, 5, 5]

    ax.axis("off")
    legend = ax.legend(
        handles,
        labels,
        loc=3,
        framealpha=1,
        frameon=True,
        ncol=ncol,
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


def generate_multi_annotator_pool_legend(path):
    handles = []
    labels = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    handles.append(plt.Rectangle((0, 0), 1, 1, color="grey"))
    labels.append("Requested Labels (Total)")
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=0.5,
            markeredgewidth=0.2,
        )
    )
    labels.append("AP = Annotator Performance")
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=0.5,
            markeredgewidth=0.2,
        )
    )
    labels.append(r"$\widehat{\text{AP}}$ = Estimated Annotator Performance")
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor=plt.get_cmap("coolwarm")(0),
            markersize=10,
        ),
    )
    labels.append("Erroneous majority vote for class 0")
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor=plt.get_cmap("coolwarm")(0.99),
            markersize=10,
        ),
    )
    labels.append("Erroneous majority vote for class 1")
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=plt.get_cmap("coolwarm")(0),
            markersize=10,
        ),
    )
    labels.append("Correct majority vote for class 0")
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=plt.get_cmap("coolwarm")(0.99),
            markersize=10,
        ),
    )
    labels.append("Correct majority vote for class 1")

    export_legend(handles, labels, ax, path=path, ncol=3)


def generate_switcher(
    repo_path=None, switcher_location=None, blacklisted_versions=None
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
