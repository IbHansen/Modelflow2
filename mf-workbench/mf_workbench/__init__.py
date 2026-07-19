"""ModelFlow model workbench for JupyterLab.

The frontend extension lives in ``labextension``; the kernel-side comm
target lives in :mod:`mf_workbench.kernel` and is imported (silently) by
the frontend when the panel connects to a notebook kernel.
"""
__version__ = "0.1.0"


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "mf-workbench"}]
