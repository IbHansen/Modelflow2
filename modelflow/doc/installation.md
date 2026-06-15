# Installing ModelFlow

This guide walks through a clean install of ModelFlow on **Windows** or **Linux**,
starting from a fresh Python setup. There are two supported routes:

1. **conda** from the `ibh` channel — recommended; pulls in the full scientific stack.
2. **pip** from PyPI (`modelflowib`) — lighter, for environments you manage yourself.

---

## 1. Install Miniforge

We recommend **Miniforge** as the Python/conda installer.

### Why Miniforge?

- It defaults to the community **conda-forge** channel, not Anaconda's `defaults`
  channel. This avoids the commercial licensing terms that Anaconda Inc. applies to
  its repository — Miniforge + conda-forge are free to use for everyone, including
  organisations.
- It's **minimal** (like Miniconda): just `conda`/`mamba` and Python, nothing you
  don't need.
- It behaves **identically across Windows, Linux and macOS**, so the instructions
  below are essentially the same everywhere.

```{note}
If you already have Miniconda or Anaconda, ModelFlow will still install — but make
sure you install from the `conda-forge` channel (add `-c conda-forge`) to avoid
mixing incompatible builds.
```

### Windows

1. Download the installer **Miniforge3-Windows-x86_64.exe** from the
   [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest).
2. Run it and accept the defaults (install "for me only" — no admin rights needed).
3. Open **"Miniforge Prompt"** from the Start menu. All conda commands below run there.

### Linux

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
```

Accept the prompts, then restart your shell (or run `source ~/.bashrc`) so that
`conda` is on your `PATH`.

---

## 2. Install ModelFlow with conda (recommended)

Create a dedicated environment named `modelflow`, installing the package from the
`ibh` channel and its dependencies from `conda-forge`:

```bash
conda create -n modelflow -c ibh -c conda-forge modelflow
conda activate modelflow
```

```{tip}
Keep ModelFlow in its own environment rather than `base`. It's easy to remove or
rebuild (`conda env remove -n modelflow`) and keeps your base install clean.
```

### Add the interactive graph component (pip)

ModelFlow's interactive model graphs use **`dash_interactive_graphviz`**, a Dash
component that is **not** published on any conda channel. Install it with pip *into
the active conda environment*:

```bash
pip install dash_interactive_graphviz
```

```{note}
This is the one piece the conda package can't pull in automatically. You only need
it if you use the interactive Dash graph views; the rest of ModelFlow works without it.
```

---

## 3. Install ModelFlow with pip from PyPI (alternative)

If you prefer to manage the environment yourself, install the PyPI distribution
**`modelflowib`**. It declares all its dependencies — including
`dash-interactive-graphviz` — so pip pulls everything in one step.

Create an isolated environment first (a conda env or a venv), then:

```bash
# using a conda env just for Python:
conda create -n modelflow-pip python=3.12
conda activate modelflow-pip

pip install modelflowib
```

or with a plain virtual environment:

```bash
python -m venv modelflow-env
# Windows:
modelflow-env\Scripts\activate
# Linux:
source modelflow-env/bin/activate

pip install modelflowib
```

```{note}
The PyPI name is **`modelflowib`**, but you still import it as usual — e.g.
`from modelclass import model`. The distribution name and the import names differ.
```

```{warning}
Drawing model graphs also needs the **Graphviz system program** (the `dot` engine).
The conda route installs this for you (via the `graphviz` package). On the pip route
you may need to install Graphviz separately — `conda install -c conda-forge graphviz`,
your Linux package manager (`apt install graphviz`), or from
[graphviz.org](https://graphviz.org/download/).
```

---

## 4. Verify the installation

In the activated environment:

```bash
python -c "from modelclass import model; print('ModelFlow ready')"
```

If that prints `ModelFlow ready` with no errors, you're set.

---

## Summary

| | conda (`ibh`) | pip (PyPI) |
|---|---|---|
| Command | `conda create -n modelflow -c ibh -c conda-forge modelflow` | `pip install modelflowib` |
| Scientific stack | installed automatically | installed automatically |
| `dash_interactive_graphviz` | `pip install` separately | included |
| Graphviz system program | included | install separately |
| Import | `from modelclass import model` | `from modelclass import model` |
