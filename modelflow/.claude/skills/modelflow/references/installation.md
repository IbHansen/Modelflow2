# Installation and environment

The authoritative install instructions are in chapter 3 of `MFMod_Python_Modelflow.pdf` and on the project README. This file is the short version: what to install, what tends to break, and how to make the notebook display cleanly under both classic Jupyter and Notebook 7 / JupyterLab.

## The standard conda install

ModelFlow lives on conda (`ibh` channel) and in two flavors:

- `ModelFlow_book` — frozen to match the published manual; safest for reproducing the examples.
- `ModelFlow_stable` — the rolling release; gets new features and fixes.

Pick one. Don't install both into the same environment.

```bash
conda create -n ModelFlow -c ibh -c conda-forge ModelFlow_book -y
conda activate ModelFlow

# Windows only, and only if the user actually has EViews installed:
conda install py2eviews -c eviews

# All platforms:
pip install dash_interactive_graphviz
```

The classic Notebook extensions (`hide_input_all`, `splitcell`, `toc2`, `varInspector`) are useful in JN6 but irrelevant under Notebook 7. Don't run those `jupyter nbextension enable` lines unless the user is on JN6 specifically.

Tested on Python 3.10 on Windows; works on macOS and Linux as well. If the user is on Apple Silicon, conda-forge generally has arm64 wheels for everything ModelFlow depends on, but `py2eviews` is Windows-only.

## Google Colab

Colab needs a small bootstrap because ModelFlow isn't pre-installed and `graphviz` is a system package:

```python
if 'google.colab' in str(get_ipython()):
    import os
    os.system('apt -qqq install graphviz')
    os.system('pip -qqq install ModelFlowIb')
```

PDF rendering from `.pdf()` does not work on Colab out of the box — Colab has no LaTeX install. Tell the user to use `.show` or `display()` instead, or to install texlive themselves if they really need PDFs.

## Updating

```bash
conda activate ModelFlow
conda update -c ibh -c conda-forge ModelFlow_stable
```

If a user has been on `ModelFlow_book` and wants the rolling release, the cleanest path is a fresh environment rather than trying to swap packages in place.

## Notebook display setup

Every notebook should open with these two lines after the import:

```python
from modelclass import model
model.widescreen()
model.scroll_off()
```

`widescreen()` injects CSS that widens the cell output area; `scroll_off()` disables the auto-scroll on long outputs (important for big tables). Both are no-ops on backends that don't recognize them, so they're safe to leave in.

## Jupyter Notebook 7 / JupyterLab compatibility

This is an active area of churn. `modelclass.py` historically targeted classic Jupyter Notebook (≤6) and the JS guards plus CSS selectors that worked there can fail silently under Notebook 7 (which is JupyterLab under the hood). Things to know:

- **`IPython` and the AMD `base/js/namespace` module are gone in JN7.** Classic JS that did `IPython.notebook...` or `require(['base/js/namespace'], ...)` throws a `ReferenceError` in the *browser*. A Python `try/except` around the cell will not catch this — the cell looks like it succeeded but the JS did nothing.
- **The DOM structure changed.** CSS selectors that targeted `.container`, `#notebook`, or `div.output_scroll` in classic Jupyter don't match JupyterLab's layout. Look for `.jp-Notebook`, `.jp-OutputArea`, `.jp-Cell` instead.

The fixes that Ib has been applying take the form of two guards plus an updated selector. Reuse this pattern when patching `scroll_off`, `scroll_on`, `widescreen`, `modelflow_auto`, or anything else that injects JS or CSS:

```python
from IPython.display import display, HTML, Javascript

js = """
if (typeof IPython !== 'undefined' && typeof require !== 'undefined') {
    // classic Notebook path
    IPython.OutputArea.prototype._should_scroll = function(){ return false; };
} else {
    // Notebook 7 / JupyterLab — use CSS instead
    var style = document.createElement('style');
    style.innerHTML = '.jp-OutputArea-output { max-height: none !important; overflow: visible !important; }';
    document.head.appendChild(style);
}
"""
display(Javascript(js))
```

The guard `typeof IPython !== 'undefined' && typeof require !== 'undefined'` correctly identifies classic JN; everything else (JN7, JupyterLab, VS Code, nbconvert HTML) falls through to the CSS path.

When the user asks "why doesn't `scroll_off()` work in my new JupyterLab" — this is the answer.

## Verifying the install

A two-line smoke test:

```python
from modelclass import model
print(model.__module__)   # should print 'modelclass'
```

If the import fails, ninety percent of the time it's because the wrong conda environment is active. Suggest `conda info --envs` to confirm.
