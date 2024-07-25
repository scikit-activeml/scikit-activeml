import os
import re
import shutil
from sphinx.application import Sphinx


def copy_gallery_notebooks_callback(app: Sphinx, exception):
    if exception is not None:
        return

    src_dir = app.config['copy_gallery_notebooks_src_path']
    dst_dir = app.config['copy_gallery_notebooks_dst_path']

    dst_dir = f"{app.outdir}/{dst_dir}"

    copy_gallery_notebooks(src_dir, dst_dir)


def copy_gallery_notebooks(src_dir, dst_dir):
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
    print('copy_sphinx_gallery_notebooks setup')

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


# if __name__ == '__main__':
#     src_dir = 'docs/generated/sphinx_gallery_examples'
#     dst_dir = 'docs/test_output/generated/sphinx_gallery_notebooks'
#     copy_gallery_notebooks(src_dir, dst_dir)
