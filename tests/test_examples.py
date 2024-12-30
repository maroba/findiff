# https://nbclient.readthedocs.io/en/latest/client.html

from pathlib import Path

import nbclient
import nbformat
import pytest


@pytest.mark.parametrize(
    "filename",
    [str(fp) for fp in Path("examples/").glob("**/*.ipynb")],
)
@pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")
def test_example_notebooks(filename):
    nb = nbformat.read(filename, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
