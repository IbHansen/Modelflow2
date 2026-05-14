# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ModelFlow Is

ModelFlow is a Python library for managing, solving, and analyzing dynamic economic and financial models. Users write models in a high-level **Business Logic Language (BLL)** — a Domain Specific Language composed of `FRML` statements — and ModelFlow generates efficient Python/Numba code to solve them. Models operate on Pandas DataFrames where rows are time periods.

## Installation / Setup

```bash
conda install modelflow -c ibh -c defaults -c conda-forge
# or for local development:
pip install -e .
```

Key dependencies: `pandas`, `numpy`, `networkx`, `numba`, `sympy`, `statsmodels`, `lmfit`, `matplotlib`, `seaborn`, `ipywidgets`, `tqdm`, `scipy`. Optional: `cvxopt`, `xlwings`.

There is no test suite or lint configuration in this repo. Development typically happens in Jupyter notebooks.

## Workflow

The user manages commits and branches via **GitHub Desktop** — do not run `git` commands unless explicitly asked.

## Module Architecture

The `model` class (assembled in `modelclass.py`) is the central object. It is built from many mixin classes and imports from almost all other modules. The data flow for typical use is:

```
BLL text (FRML statements)
    → modelpattern.py      (tokenize/parse)
    → modelmanipulation.py (macro expansion: DO/LIST/TLIST loops, unrolling)
    → modelnormalize.py    (normalize formulas, resolve DLOG/DIFF/MOVAVG etc.)
    → modelclass.py        (analyzemodelnew: build variable graph, solve order via networkx)
    → Solver_Mixin         (generate & exec Python/Numba solver code, run Newton)
    → result DataFrame
```

### Core modules

| Module | Role |
|---|---|
| `modelclass.py` | Central `model` class assembled from all mixins; `analyzemodelnew()` is the entry-point for parsing; `Solver_Mixin` handles Gauss–Seidel / Newton solving |
| `modelpattern.py` | Regex-based tokenizer and BLL parser; defines `FrmlParts`, `udtryk_parse`, `find_frml`; enumerates valid function names from `modelBLfunk` + `modeluserfunk` |
| `modelmanipulation.py` | Template expansion: `DO`/`ENDDO` loops, `LIST`/`TLIST` substitution, `tofrml()`, `dounloop()` |
| `modelnormalize.py` | Formula normalization using SymPy; resolves `DLOG`, `DIFF`, `MOVAVG`, `PCT_GROWTH`; produces `Normalized_frml` dataclass |
| `modelBLfunk.py` | Functions callable inside BLL (`logit`, `logit_inverse`, `lifetime_credit_loss`, cvxopt wrappers); auto-discovered by `modelpattern.py` |
| `modeluserfunk.py` | User-extensible functions for BLL; also auto-discovered |
| `modelhelp.py` | Low-level utilities: `update_var`, `debug_var`, graph helpers |
| `modelmf.py` | Pandas accessor `.mfcalc` — run BLL equations directly on a DataFrame |
| `modelnet.py` | Graph/network analysis of model structure (uses networkx) |
| `modeldekom.py` | Attribution / decomposition analysis |
| `modelinvert.py` | `targets_instruments` class for solving target–instrument problems |
| `modeldiff.py` | Differentiation utilities |
| `modelvis.py` | Visualization helpers (matplotlib/seaborn); `meltdim()` for multi-dimensional data |
| `modelreport.py` | `LatexRepo`, `DisplaySpec`, HTML/LaTeX report generation |
| `modeljupyter.py` | Jupyter notebook integration, interactive widgets |
| `modeldash.py` / `modeldashboot.py` | Dash web-app wrappers |
| `modelnewton.py` | Newton solver implementation |
| `model_cvx.py` | cvxopt optimization helpers (`mv_opt`, `mv_opt_prop`) |
| `modelclass2.py` | Experimental subclass `simmodel` of `model`; older Cython/Numba experiments |

### Higher-level DSL layers

These modules sit *above* the core BLL and provide richer authoring surfaces or construction pipelines:

| Module | DSL layer / purpose |
|---|---|
| `modelconstruct.py` | Dataclass-driven model construction DSL (2025). Handles the full pipeline: `clean_expressions()` normalizes raw BLL text (collapses `LIST`/`TLIST`/`DO` blocks, calls `tofrml`), then builds typed dataclass representations of equations. Integrates with `model_latex_class` so equations can be rendered to LaTeX inline. Entry point for new model authoring. |
| `modelconstruct_estimation.py` | Extension of `modelconstruct` that adds estimation support: wraps `Estimate_*` backends into the construction pipeline so equations can be econometrically estimated and then embedded directly in a model. |
| `modelmanipulation.py` | Macro-expansion layer of the BLL: `LIST`/`TLIST` substitution (template unrolling over bank or entity lists), `DO`/`ENDDO` loops, `tofrml()` normalization. Used both standalone and called from `modelconstruct`. |
| `modelnormalize.py` | Formula-level normalization: rewrites high-level shorthands (`DLOG`, `DIFF`, `MOVAVG`, etc.) into plain arithmetic via SymPy, computes add-factor forms, fitted forms, and un-normalized forms. Produces `Normalized_frml` dataclasses consumed by the grab/onboarding pipeline. |

### Estimation modules

| Module | Role |
|---|---|
| `modelestimator_new.py` | Plugin-style estimation backends: `Estimate_ols` (statsmodels OLS), `Estimate_nls_lmfit` (nonlinear via lmfit), `Estimate_nls_eviews` (nonlinear via EViews round-trip). Base class `EstimatorBackend` defines the contract; add new backends by subclassing it and implementing `_fit()` and `_render_summary_html()`. |
| `modelestimation.py` | Older estimation helpers: `LSResult` (HTML report wrapper), `EqContainer` (stitch multiple estimated equations into a model), `EqContainer.init_add_factors()`. Parses EViews-style `C(n)` parameter notation to ModelFlow `prefix__n` form. |

### Onboarding from external systems

Each module ingests a foreign model format and produces a `model` instance plus a populated DataFrame:

| Module | Source format | Key class / function | Notes |
|---|---|---|---|
| `modelgrabwf2.py` | EViews `.wf1` / `.wf2` | `GrabWf2Model` | Full pipeline: launches EViews, unlinks model, saves as `.wf2` (JSON), extracts equations + data, normalizes via `modelnormalize`, adds add-factors, dummies, and fitted-value models. Most complete EViews importer. |
| `modelgrab.py` | World Bank model format (EViews-style text + Excel data) | `GrapWbModel` | Older grab class; reads formula text + Excel sheet, normalizes, creates main model + add-factor model + fitted model. |
| `modelmacrograb.py` | World Bank macro model (plain equation text) | `GrabMacroModel` | Simpler / newer alternative to `modelgrab`; uses `modelnormalize.normal()` directly; no Excel dependency. |
| `modelgrabgdx.py` | GAMS GDX files | `GdxDataset`, `model_grab_gdx()` | Requires `pip install "gamsapi[transfer]"` and a GAMS installation. Loads free variables with a time dimension into a wide DataFrame; bundles multiple scenarios into a single model. |
| `model_dynare.py` | Dynare `.mod` files | `grap_modfile` | Reads expanded `.mod` output; produces three models: main (`mthismodel`), residuals (`mresmodel`), parameter injection (`mparamodel`). Saves `.frm`, `_res.frm`, `_para.frm`, `_cons.mod`, `.inf` files. |
| `model_Excel.py` | Excel workbooks | `findequations()` + `Excel_Mixin` | Translates Excel cell formulas to BLL using openpyxl (multicell ranges expanded to comma lists). Also provides xlwings automation helpers used by `modeldump_excel` / `modelload_excel` in `modelclass`. |
| `model_latex.py` | LaTeX equation blocks | `rebank()` + parsing functions | Regex-based; translates a specific LaTeX equation style to BLL. Format-specific — inspect before use on new LaTeX sources. |
| `model_latex_class.py` | LaTeX equation blocks | `a_latex_model`, `a_latex_equation` | Dataclass-based rewrite of `model_latex` (2022). Used by `modelconstruct` to render BLL equations back to LaTeX and to parse LaTeX into BLL. Provides `defrack`, `depower`, `debrace`, `defunk` helpers. |

### BLL Language Basics

A model is a multi-line string of `FRML` statements:
```
FRML <name> endovar = expression $
```
- Variables reference lags with `varname(-1)`, `varname(-2)`, etc.
- `DLOG(X)` = log difference, `DIFF(X)` = first difference, `MOVAVG(X,n)` = moving average
- `LIST` / `TLIST` blocks and `DO`/`ENDDO` loops allow macro-style expansion
- `£` is the comment character

### Key Conventions

- All variable names inside BLL are uppercased automatically.
- The `model` class stores parsed equation metadata in `self.allvar` (dict keyed by variable name).
- Solve order is determined topologically via networkx on the dependency graph; simultaneous blocks are solved iteratively.
- Numba JIT compilation is used for performance-critical solver loops (optional, degrades gracefully if not available).
- The `upd` class in `modelclass.py` is a Pandas DataFrame extension for model-aware updates.
- `modelnormalize.Normalized_frml` is the standard dataclass for a single parsed/normalized equation.
