# Onboarding World Bank models

The World Bank publishes a collection of country-specific MFMod-based macro-structural models for use with ModelFlow. This file explains how to get them, what's in them, and the naming conventions you need to read them effectively. PDF chapters 8–10 are the deep reference.

---

## What's available

As of 2025, the public World Bank repo at https://github.com/worldbank/MFMod-ModelFlow contains:

- **Bolivia** — annual
- **Croatia** — quarterly
- **Iraq** — annual
- **Nepal** — annual, climate-aware
- **Pakistan** — annual, climate-aware
- **Türkiye** — annual, climate-aware

The "climate-aware" variants extend the standard MFMod framework with the features described in Burns et al. (2021): emissions accounting (CO₂, CH₄, N₂O), carbon tax instruments, temperature- and weather-driven damage functions, and so-called co-benefits (pollution, health, productivity). Variants now exist for ~70 countries internally — only the subset above is currently public, though more get released over time.

There are also ~100+ stand-alone variants known as **MFModSA** (MacroFiscalModel StandAlones), built off the main internal model. They live outside the public ModelFlow repo and are reached through different channels.

The earlier `isimulate` platform (https://isimulate.worldbank.org) also hosts MFMod variants for browser-based simulation, but those don't expose the Python code.

---

## Downloading the models

`model.Worldbank_Models()` clones the public GitHub repo into a local folder:

```python
from modelclass import model
model.Worldbank_Models()                    # default: ./WorldbankModels/
model.Worldbank_Models('./my_models')       # custom destination
```

Options:

| Option | Default | Purpose |
|---|---|---|
| `destination` | `./WorldbankModels` | Local folder. Cannot point above the CWD (security). |
| `owner` | `'worldbank'` | GitHub owner; rarely changed. |
| `repo_name` | `'MFMod-ModelFlow'` | Repository name. |
| `branch` | `'main'` | Branch to download. |
| `silent` | `True` | Verbose download log when `False`. |
| `replace` | `False` | If `True`, overwrites local files with remote versions. `False` keeps user changes. |
| `go` | `True` | Whether to display the TOC after download. |

To fully refresh, delete the destination folder first, then re-run. `model.display_toc(folder='WorldbankModels')` lists everything that was downloaded — typically a tree of country folders, each containing one or more notebooks and a `models/` subfolder with `.pcim` files.

---

## Loading a country model

Once downloaded, the workflow is identical to any other ModelFlow model:

```python
from modelclass import model
mpak, baseline = model.modelload('./WorldbankModels/Pakistan/models/pak.pcim',
                                  alfa=0.7, run=True, keep='Baseline')
```

Conventional naming for the model instance: `m` + ISO-3 country code, lowercase. So `mpak`, `mtur`, `mnpl`, `mbol`, `mhrv`, `mirq`. Notebooks use this consistently, and matching it makes example code in the manual transferable.

The model's `.name` attribute typically holds the ISO-3 code (`'PAK'`). It also appears in `mpak.modelname` and is used by string-substitution placeholders like `{cty}` (e.g. in titles via `title='{cty_name} GDP components'`).

---

## Variable naming conventions

This is the convention you'll see throughout MFMod models. Internalizing it lets you build wildcard patterns and read mnemonics quickly:

```
[CTY][CATEGORY][SUBCATEGORY][SECTOR][UNIT][SUFFIX]
```

Examples decoded:

| Mnemonic | Decoded |
|---|---|
| `PAKNYGDPMKTPKN` | PAK / NY / GDP / MKTP / KN — Pakistan / National accounts / GDP / Market prices / Constant LCU |
| `PAKNECONPRVTKN` | PAK / NE / CON / PRVT / KN — Pakistan / Expenditure / Consumption / Private / Constant LCU |
| `PAKGGREVCO2CER` | PAK / GG / REV / CO2C / ER — Pakistan / General government / Revenue / CO2-Coal / Effective rate |
| `PAKBXGSRMRCHCD` | PAK / BX / GS / RMRCH / CD — Pakistan / BoP Exports / Goods&services / Merchandise / Current US$ |
| `PAKCCEMISCO2TKN` | PAK / CC / EMIS / CO2 / TKN — Pakistan / Climate / Emissions / CO2 / Tons |

Category abbreviations to know:

| Abbr | Meaning |
|---|---|
| `NY` | National accounts |
| `NE` | National expenditure |
| `BX` / `BM` | Balance of payments — exports / imports |
| `BN` | Balance of payments — net |
| `GG` | General government |
| `FM` | Financial / monetary |
| `FP` | Prices |
| `SL` / `SP` | Labor / population |
| `CC` | Climate |
| `EN` | Energy |

Unit suffixes:

| Suffix | Meaning |
|---|---|
| `KN` | Constant local currency units (real) |
| `CN` | Current local currency units (nominal) |
| `KD` / `CD` | Constant / current US dollars |
| `XN` | Deflator / price index |
| `ZG` | Growth rate |
| `ZS` | Percent / share |

This means a request like "show me real exports for all countries in my dataset" maps cleanly to `*BXGSR*KN`, regardless of which country prefix is involved.

### The four companion variables on every behavioral equation

Every behavioral equation has four companion variables:

| Variable | Purpose |
|---|---|
| `VAR` | The endogenous variable itself |
| `VAR_A` | Add-factor (offset on the estimated RHS) |
| `VAR_D` | Fix-dummy (0 = use equation, 1 = use `VAR_X`) |
| `VAR_X` | Fix-value (used when `VAR_D = 1`) |
| `VAR_FITTED` | Fitted-value sub-equation (for diagnostics) |

A description search shows them all:

```python
mpak['!*government*expenditure*goods*'].des
```

returns:

```
PAKGGEXPGNFSCN        : General government expenditure on goods and services
PAKGGEXPGNFSCN_A      : Add factor:...
PAKGGEXPGNFSCN_D      : Fix dummy:...
PAKGGEXPGNFSCN_X      : Fix value:...
PAKGGEXPGNFSCN_FITTED : Fitted value:...
```

Identities (accounting relationships) only have `VAR` — no `_A` / `_D` / `_X` / `_FITTED`. So a wildcard like `*_D` returns exactly the behavioral variables, which is a useful structural query.

---

## Wildcards and variable selection

ModelFlow's wildcard system shows up everywhere — in `pat=` arguments, in the `[]` selector, in `.fix()`, in `var_groups` definitions:

| Syntax | Matches |
|---|---|
| `*` | any string |
| `?` | one character |
| `!pattern` | search **variable descriptions** rather than mnemonics |
| `#GROUPNAME` | predefined variable group (see `mpak.var_groups`) |
| `#ENDO` | all endogenous variables |
| `#EXO` | all exogenous variables |

Examples:

```python
mpak['*NYGDP*'].plot()                       # all GDP-related variables
mpak['PAKNYGDP??????'].rename().plot()       # exactly nine-char names starting PAKNYGDP
mpak['!*Carbon*'].des                        # everything with "Carbon" in its description
mpak['#Headline'].rtable()                   # the model's headline-indicator group
```

Variable groups are defined in the `.pcim` file when the model is built; for WB models, common groups are `#Headline`, `#Fiscal`, `#External`, `#Real`, `#Prices`. Inspect `mpak.var_groups` to see what's defined for the current model.

---

## What's in a `.pcim` file

`.pcim` is a zipped JSON containing:

- Equations (FRML strings, including the `<DAMP,STOC>` flags that drive the solver)
- Variable descriptions
- Variable groups
- The data (historical and projection)
- Solver options used at save time (re-used when `run=True`)
- Optionally: kept scenarios (`keep_solutions`)
- Optionally: stored report specifications (`reports`)

Round-tripping is lossless: `mpak.modeldump('out.pcim', keep=True)` followed by `model.modelload('out.pcim', run=True)` reproduces the working state including all scenarios and reports. This is the recommended way to share a working session with a colleague.

---

## Climate-aware specifics

The climate-aware models (Pakistan, Nepal, Türkiye, ...) have these additional building blocks:

- **Emissions block** (`*CCEMIS*`): tracks tons of CO₂, CH₄, N₂O from various activities (coal, gas, oil, agriculture, industry, transport, ...).
- **Carbon tax instruments**: `*GGREVCO2CER`, `*GGREVCO2GER`, `*GGREVCO2OER` — effective tax rates on coal, gas, oil in USD/ton.
- **Damage functions**: temperature- and rainfall-linked impacts on agricultural output, labor productivity, capital depreciation.
- **Co-benefits**: pollution-linked health expenditure adjustments, informality and labor force shifts.

A common pilot scenario is "introduce a $30/ton carbon tax on all three fuels from 2025 onward, hold it nominal vs inflation-adjust it, examine emissions, GDP, fiscal balance". The PDF chapter 12 walks through this for Pakistan and is a good template for new analysts.

---

## When something looks wrong

When a variable's value or description doesn't match the user's expectations, the diagnostic ladder is:

1. `mpak[VAR].des` — confirm the description.
2. `mpak[VAR].frml` — show the equation and its normalized form. For an exogenous variable, this just says "exogenous".
3. `mpak[VAR].show` — show the equation **and** the values of LHS and RHS variables from `.basedf` and `.lastdf` over the current sample.
4. `mpak[VAR].draw()` (or `.dep(...)`) — graph the dependency tree.

`mpak.modeltext` returns the full model as a single string, useful for grepping. `mpak.endogene`, `mpak.exogene`, `mpak.allvar` give programmatic access to the variable sets.
