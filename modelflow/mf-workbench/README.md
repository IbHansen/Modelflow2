# mf-workbench

A JupyterLab 4 extension: a **Model Workbench** panel for ModelFlow.
It binds to the current notebook's kernel over a Jupyter comm and shows,
for any `model` instance in the notebook namespace:

- **Dependencies** – the local dependency graph around a variable
  (Graphviz SVG rendered kernel-side; click a node to re-focus).
- **Variables** – endogenous/exogenous explorer with descriptions,
  last values, and live filtering (click a row to focus that variable).
- **Attribution** – attribution table for the focused variable
  (`get_att_pct` by default, `dekomp` fallback).

All ModelFlow-specific logic lives in `mf_workbench/kernel.py` in the
three `_mf_*` functions — that is where to customize.

## Layout

```
mf-workbench/
├── pyproject.toml          # hatchling + hatch-jupyter-builder
├── package.json            # JupyterLab 4 prebuilt extension
├── tsconfig.json
├── src/index.ts            # frontend: panel + comm client
├── style/index.css
└── mf_workbench/
    ├── __init__.py         # labextension paths
    └── kernel.py           # comm target + ModelFlow introspection
```

## Build (one-time setup)

Requires Node.js >= 18 and JupyterLab >= 4 in the environment
(`conda install -c conda-forge nodejs jupyterlab`).

```bash
cd mf-workbench
pip install -e .                 # builds the labextension via hatch hook
jupyter labextension develop . --overwrite
jupyter lab                      # Command palette -> "ModelFlow: Open Model Workbench"
```

On conda/miniforge, install the build deps into the env first and skip
pip's build isolation (otherwise pip downloads a second JupyterLab just
to build):

```bash
conda activate book314
pip install hatchling hatch-jupyter-builder
pip install -e . --no-build-isolation
jupyter labextension develop . --overwrite
```

The graph pane pipes DOT text through the Graphviz `dot` executable -
the same binary `model.draw` uses, so any environment where `.draw`
works needs nothing extra. (If `dot` is missing:
`conda install -c conda-forge graphviz`. The python-graphviz wrapper is
NOT used.)

To use the panel in other environments WITHOUT node or a conda channel,
build the wheel once (see Release) and `pip install` it there - the
compiled labextension travels inside the wheel, and JupyterLab 4 picks
up prebuilt extensions with no rebuild.

The package must also be importable **in the kernel's environment**
(the panel silently runs `import mf_workbench.kernel` on connect). With
a single conda env this is automatic; with multiple kernels, install it
into each.

## Development loop

```bash
jlpm watch        # rebuild TypeScript on save; refresh the browser tab
```

Kernel-side changes to `kernel.py`: restart the kernel (or
`importlib.reload`), then press **Reconnect** in the panel.

## Release

```bash
pip install build
python -m build                  # sdist + wheel with the labextension bundled
```

The wheel is a normal pure-Python artifact — suitable for the `ibh`
conda channel via a standard noarch recipe (add `nodejs` and
`hatch-jupyter-builder` to the build requirements, or build the wheel
first and package that).

## Message protocol

See the docstring at the top of `mf_workbench/kernel.py`.

## Ideas / extension points

- Period selector for the variable explorer (send a `period` field).
- Waterfall chart for attribution: render matplotlib to PNG kernel-side,
  send as base64, drop into an `<img>`.
- Follow the active notebook automatically via
  `tracker.currentChanged.connect(...)` instead of the Reconnect button.
