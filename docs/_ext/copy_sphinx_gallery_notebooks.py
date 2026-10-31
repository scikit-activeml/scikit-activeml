import json
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path

from sphinx.application import Sphinx

# A reST raw HTML block as Sphinx-Gallery copies it into a markdown cell.
RAW_HTML_PATTERN = re.compile(
    r"^\.\. raw:: html\n\n((?:[ \t]+.*(?:\n|$))+)", re.M
)


def copy_gallery_notebooks_callback(app: Sphinx, exception):
    """A callback for the `build-finished` signal that is used by the
    `copy_gallery_notebooks` extension, that copies and post-processes
    notebooks from sphinx gallery examples. It also renders the raw HTML
    blocks of the downloadable notebooks.

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object that is used to access the parsed sphinx
        config.
    exception : Exception or None
        This parameter is `None` if no exceptions were raised during the
        building process. If an exception was raised, that exception will be
        accessible via `exception`.
    """
    if exception is not None:
        return

    src_dir = app.config["copy_gallery_notebooks_src_path"]
    dst_dir = app.config["copy_gallery_notebooks_dst_path"]

    src_dir = f"{app.srcdir}/{src_dir}"
    dst_dir = f"{app.outdir}/{dst_dir}"

    copy_gallery_notebooks(src_dir, dst_dir)
    render_downloaded_notebooks(f"{app.outdir}/_downloads")


def copy_gallery_notebooks(src_dir, dst_dir):
    """This function copies and post-processes notebooks from sphinx gallery
    examples.

    Parameters
    ----------
    src_dir : str
        The source directory from which notebooks are copied.
    dst_dir : str
        The destination directory to which all notebooks are copied.
    """
    os.makedirs(dst_dir, exist_ok=True)

    for src_root, dirs, files in os.walk(src_dir):
        dst_root = src_root.replace(src_dir, dst_dir)
        notebook_files = [f for f in files if f.endswith(".ipynb")]
        if len(notebook_files):
            os.makedirs(dst_root, exist_ok=True)
            for f in notebook_files:
                src_path = f"{src_root}/{f}"
                dst_path = f"{dst_root}/{f}"
                process_notebook(src_path, dst_path)


def process_notebook(src_path, dst_path):
    """This function copies a jupyter notebook and removes the comment symbol
    from `!pip install <packeges>` commands such that they will be executed by
    default.

    Parameters
    ----------
    src_path : str
        The file path to the notebook that needs to be modified.
    dst_path : str
        The file path where the modified notebook is saved to.
    """
    with open(src_path, "r") as f:
        file_content = f.read()

    pattern = r"\"# (!pip install .*?)\""
    repl = r'"\1"'
    output = re.sub(pattern=pattern, repl=repl, string=file_content)
    output = render_raw_html(output)

    with open(dst_path, "w") as f:
        f.write(output)


def render_raw_html(notebook):
    """Replace the reST raw HTML blocks in markdown cells by their HTML.

    Parameters
    ----------
    notebook : str
        The content of a jupyter notebook file.

    Returns
    -------
    str
        The notebook content, unchanged if it has no raw HTML block.
    """
    content = json.loads(notebook)
    rendered = False
    for cell in content["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        html = RAW_HTML_PATTERN.sub(
            lambda match: textwrap.dedent(match.group(1)), source
        )
        if html != source:
            cell["source"] = [html]
            rendered = True
    return json.dumps(content, indent=2) if rendered else notebook


def render_downloaded_notebooks(downloads_dir):
    """Render the raw HTML blocks of the notebooks offered for download,
    including those in zip archives.

    Parameters
    ----------
    downloads_dir : str
        The directory of the downloadable files of the build.
    """
    for path in Path(downloads_dir).rglob("*.ipynb"):
        notebook = path.read_text()
        rendered = render_raw_html(notebook)
        if rendered != notebook:
            path.write_text(rendered)
    for path in Path(downloads_dir).rglob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            entries = [
                (info, archive.read(info)) for info in archive.infolist()
            ]
        rendered = [
            (
                info,
                (
                    render_raw_html(data.decode()).encode()
                    if info.filename.endswith(".ipynb")
                    else data
                ),
            )
            for info, data in entries
        ]
        if rendered != entries:
            with zipfile.ZipFile(path, "w") as archive:
                for info, data in rendered:
                    archive.writestr(info, data)


def setup(app: Sphinx):
    """This function is called during the initialization of the Sphinx
    extension. Here, default values are added to the config and callbacks
    are connected, such that they can be called after the corresponding signal
    is emitted.

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object where config values are added and the
        callback is connected to.

    Returns
    -------
    dict
        Extension meta data and restrictions in parallel processing.
    """

    app.add_config_value(
        "copy_gallery_notebooks_src_path",
        "generated/sphinx_gallery_examples",
        "html",
    )
    app.add_config_value(
        "copy_gallery_notebooks_dst_path",
        "generated/sphinx_gallery_notebooks",
        "html",
    )

    app.connect("build-finished", copy_gallery_notebooks_callback)

    return {
        "version": "0.1",
        "env_version": 1,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
