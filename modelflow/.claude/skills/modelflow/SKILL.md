---
name: modelflow
description: How to use the ModelFlow Python framework for macro-structural economic models (e.g. World Bank MFMod). Use this skill whenever the user mentions ModelFlow, MFMod, MFModSA, `.pcim` files, calls like `model.modelload`, `.mfcalc`, `.upd`, `mpak`, `mtur`, `mnpl`, the World Bank Pakistan/Bolivia/Croatia/Iraq/Nepal/Türkiye/climate models, or anything about installing ModelFlow, loading or building a macro model, running scenarios, plotting baseline-vs-shock results, building report tables/plots/text, estimating equations with OLS/NLS, working with `_A`/`_D`/`_X` add-factor variables, EViews equation porting, targeting/inversion, decomposing/attributing scenario impacts back to drivers, tracing causal chains, compiling with numba/ljit, or analyzing why a shock propagated the way it did. Also trigger when the user imports `modelclass`, `modelconstruct`, `modelestimation`, `modelreport`, `modelinvert`, or `modeldekom` — even if they don't name ModelFlow explicitly.
---

# Using ModelFlow

ModelFlow is a Python framework for macro-structural economic models, originally built around the World Bank's MFMod system. The central module is `modelclass`, which exposes a single `model` class assembled from many mixins (Solver, Display, Graph, Dekomp, Excel, Dash, Modify, Fix, Stability, Report, ...). Almost everything goes through a `model` instance plus pandas DataFrames extended with two ModelFlow methods (`.upd` and `.mfcalc`).

A canonical session opens like this:

```python
%matplotlib inline
import pandas as pd
from modelclass import model

model.widescreen()
model.scroll_off()

mpak, baseline = model.modelload('../models/pak.pcim', run=True, keep='Baseline')
```

## How to use this skill

This SKILL.md is a router. The actual material lives in `references/`, one file per arena. Read the reference file(s) matching the user's task before answering — they each fit in well under the model's context budget and contain the gotchas, parameter names, and patterns that aren't safe to improvise.

| User's task | Read this reference file |
|---|---|
| Installing ModelFlow; setting up conda/pip environments; fixing Jupyter Notebook 7 / JupyterLab rendering quirks | `references/installation.md` |
| Loading a model, building shocks with `.upd`, transforming DataFrames with `.mfcalc`, plotting with `.plot` / `keep_plot`, building report tables/plots/text | `references/usage.md` |
| Running simulations: solver options (`alfa`, `ljit`, `silent`, `max_iterations`), kept scenarios, `.fix` / exogenization, add-factor mechanics, `.set_smpl`, targeting / inversion | `references/simulation-methods.md` |
| Downloading and onboarding the public World Bank country models (Bolivia, Croatia, Iraq, Nepal, Pakistan, Türkiye, ...), MFMod naming conventions, climate-aware variants | `references/worldbank-onboarding.md` |
| Building a new model from scratch — writing `FRML` strings, EViews `.mod` porting, the `Mexplode` DSL, `.equpdate` / `.eqdelete` to modify existing models | `references/model-construction.md` |
| Authoring a model in markdown or LaTeX with `Makemodel` / `%%Makemymodel` — `>` / `>>` equation lines, tags (`<estimator>`, `<ident>`, `<smpl>`, `<constraints>`, ...), LIST/DO templating, `replacements=`, reports | `references/makemodel.md` |
| Estimating equations — backends (OLS via statsmodels, NLS via lmfit, NLS via EViews), inline `ST.` constraints (`=[lo,hi]`, `>`, `<`, `=`, `~`, `:=`), `Estimate_*` classes, `org_eq_baked`, Makemodel integration | `references/estimation.md` |
| Decomposing why a scenario moved — single-equation attribution via `.dekomp` / `get_att_pct` / `dekomp_plot`, model-level attribution via `.totdif` and the `totdif` class, causal-chain tracing via `.tracepre` / `.tracedep` / `.draw` | `references/decomposition.md` |

Several tasks span multiple files (e.g. "build a model from estimated equations and run a scenario" touches construction-with-estimation, simulation, and usage). In that case read all relevant files first, then answer. Reading three small reference files is cheap; guessing at API surface from memory is not.

## What never to skip

- **Read the relevant reference file before writing code.** ModelFlow has many silent gotchas (`upd` requires spaces around operators, `mfcalc` initializes new variables to zero, `_D` flips behavioral equations between estimated and exogenized, etc.). The references capture these.
- **Treat `MFMod_Python_Modelflow.pdf` as the authoritative manual.** When a reference file says "see chapter N of the PDF", actually point the user there for the deep version rather than reproducing it.
- **Inspect the source when something isn't covered.** The codebase is mostly in `modelclass.py` and a set of mixin modules (`modelconstruct.py`, `modelestimation.py`, `modelreport.py`, `modelinvert.py`, `modeldekom.py`, `modelmanipulation.py`, ...). Reading a module's docstrings is usually faster than guessing.
- **Don't invent method names.** If the user asks about something and no reference file covers it, search the source files rather than fabricating an API. The mixin pattern means methods are scattered — `grep` is your friend.

## Style reminders for ModelFlow code

These hold across every arena and are worth keeping in mind regardless of which reference file is open:

- Variable mnemonics are always uppercase; ModelFlow silently uppercases lowercase input. World Bank models follow ISO-3 country prefixes (`PAK`, `TUR`, `NPL`, ...) — use them when constructing wildcard patterns.
- Wildcards use `*` and `?`; a leading `!` switches search to variable descriptions (`mpak['!*Carbon*']`).
- `.upd` and `.mfcalc` are **DataFrame** methods. `.plot`, `.table`, `.text`, `.report`, `.fix`, `.invert` are **model** methods. The user will mix these up — gently correct.
- Most "render" methods return display-definition objects, not figures. To actually show output: evaluate the object alone in a Jupyter cell, call `display(obj)`, use `obj.show`, or `obj.pdf()`.
- `keep='scenario name'` on a model call stores a copy of the result in `mpak.keep_solutions[name]`. The report and plot machinery looks there.
- `mpak.modeldump(path, keep=True)` and `model.modelload(path)` round-trip everything including stored reports, kept scenarios, and variable descriptions.
