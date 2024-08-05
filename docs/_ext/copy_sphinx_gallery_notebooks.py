import os
import re
import shutil
from sphinx.application import Sphinx


def copy_gallery_notebooks_callback(app: Sphinx, exception):
    """A callback for the `build-finished` signal that is used by the
    `copy_gallery_notebooks` extension, that copies and post-processes
    notebooks from sphinx gallery examples.

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

    src_dir = app.config['copy_gallery_notebooks_src_path']
    dst_dir = app.config['copy_gallery_notebooks_dst_path']

    src_dir = f"{app.srcdir}/{src_dir}"
    dst_dir = f"{app.outdir}/{dst_dir}"

    copy_gallery_notebooks(src_dir, dst_dir)


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
        notebook_files = [f for f in files if f.endswith('.ipynb')]
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
    with open(src_path, 'r') as f:
        file_content = f.read()

    pattern = r'\"# (!pip install .*?)\"'
    repl = r'"\1"'
    output = re.sub(
        pattern=pattern,
        repl=repl,
        string=file_content
    )

    with open(dst_path, 'w') as f:
        f.write(output)


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
        'copy_gallery_notebooks_src_path',
        'generated/sphinx_gallery_examples',
        'html'
    )
    app.add_config_value(
        'copy_gallery_notebooks_dst_path',
        'generated/sphinx_gallery_notebooks',
        'html'
    )

    app.connect('build-finished', copy_gallery_notebooks_callback)

    return {
        'version': '0.1',
        'env_version': 1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
