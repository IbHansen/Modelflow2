# Building a model from equations

This file covers constructing a ModelFlow model from scratch (or modifying an existing one). The estimation side — fitting equations to data with OLS/NLS — is a separate file (`model-construction-estimation.md`); this one focuses on the *structure*: writing equations, normalizing them, and assembling them into a runnable model.

Relevant source modules: `modelclass.py` (the `model` class itself), `modelmanipulation.py` (parsing and normalization), `modelconstruct.py` (the `Mexplode` DSL preprocessor), `modelnormalize.py` (equation normalization), `modelgrab.py` and `modelgrabwf2.py` (EViews and WF2 import). PDF chapters 10, 12.6, and 18.7 cover the basics.

---

## The simplest path: build a model from a string

The `model` constructor accepts a string of FRML statements:

```python
from modelclass import model

eqs = """
FRML <>           Y = C + I + G + X - M $
FRML <DAMP,STOC>  C = 0.6 * Y(-1) + C_A $
FRML <>           I = K - K(-1) $
FRML <>           K = 0.95 * K(-1) + I $
"""

mtoy = model(eqs, modelname='toy')
```

Anatomy of a FRML line:

```
FRML <flags>   <left-hand side> = <right-hand side> $
```

- `FRML` is the literal keyword.
- `<flags>` (in angle brackets) carry metadata: `<>` is an identity (no flags), `<DAMP>` marks the equation for damping during Gauss-Seidel, `<STOC>` marks it as behavioral (estimated, gets `_A`/`_D`/`_X` companions), `<Z>` is an alias for `<DAMP>`, `<EXO>` makes the equation fixable.
- Equation ends with `$`.

Lowercase is fine on input — ModelFlow uppercases everything. Lags use `VAR(-1)`, leads use `VAR(+1)` or `VAR(1)`.

Once built:

```python
import pandas as pd
df = pd.DataFrame(...)        # must contain initial values for all variables
result = mtoy(df, start, end)
```

For very small ad-hoc models, `.mfcalc` on a DataFrame achieves the same thing without committing to a `model` object — useful for prototyping.

---

## Flags worth knowing

| Flag | Meaning |
|---|---|
| `<>` | Plain identity. No add-factor, no fixable structure. Solver assumes it holds exactly. |
| `<DAMP>` | Apply damping (`alfa`) during Gauss-Seidel. Standard for any equation that the solver might oscillate on. |
| `<STOC>` | Behavioral / stochastic equation. Triggers add-factor generation (`_A`) and fixable structure (`_D`, `_X`). Almost always paired with `<DAMP>`: `<DAMP,STOC>`. |
| `<EXO>` | Make the equation fixable (generate `_D`, `_X`) without it being behavioral. |
| `<FIT>` | Generate a fitted-value sub-equation (`_FITTED`). |
| `<IDENT>` | Pure identity — used by `modelgrabwf2.py` when importing WF2 files; mostly equivalent to `<>`. |
| `<QUASIIDENT>` | "Quasi-identity": treated as stochastic for solver purposes but flagged separately for grouping. |
| `<CALC_ADD_FACTOR>` | Marks an auxiliary equation that calculates an add-factor. The grabber generates these automatically. |
| `<ADD_SUFFIX 'XYZ'>` | Override the default `_A` suffix for the add-factor — e.g. `<ADD_SUFFIX '_ADJ'>`. |
| `<ENDO 'VARNAME'>` | Override which variable is treated as the LHS endogenous (useful when normalization would otherwise pick the wrong one). |

The combination `<DAMP,STOC>` is the workhorse — every behavioral equation in an MFMod-style model carries it. Identities carry `<>`.

---

## Normalization

Equations don't have to be written with the endogenous variable already isolated on the left. ModelFlow can normalize:

```python
import modelmanipulation as mp
import modelnormalize as nz

# Equation written with the endogenous variable inside a transformation:
raw = "DLOG(Y) = 0.02"
print(nz.normalize(raw))
# → Y = EXP(LOG(Y(-1)) + 0.02)
```

What normalization does:

- Picks the first variable encountered on the LHS as the endogenous variable (unless `<ENDO>` says otherwise).
- Algebraically rearranges to isolate it.
- Handles common wrappers: `LOG`, `DLOG`, `DIFF`, `EXP`.
- Optionally adds an add-factor (`add_add_factor=True`) and/or fixable structure (`make_fixable=True`).

For most workflows the normalization happens automatically when the model is built; you only call `nz.normalize` directly when debugging "why did my equation get turned into *that*".

`mfcalc` on a DataFrame uses the same normalizer, which is why `df.mfcalc('dlog(a) = 0.02', showeq=True)` prints the EViews-style FRML.

---

## Modifying an existing model: `.equpdate` and `.eqdelete`

You rarely build a brand-new MFMod from scratch — typical workflow is to **load** an existing model and tweak it. PDF chapter 12.6 has the worked example.

```python
# Replace an existing equation, e.g. make a carbon tax real instead of nominal:
new_eq = """
FRML <DAMP,STOC>
    PAKGGREVCO2CER = PAKGGREVCO2CER_TARGET * (PAKNYGDPMKTPXN / PAKNYGDPMKTPXN(2025)) + PAKGGREVCO2CER_A $
"""

mpak_v2, newdf = mpak.equpdate(new_eq, add_add_factor=True, calc_add=True,
                                newname='Pakistan real CT')
```

`equpdate` returns a new model instance and a DataFrame with recomputed add-factors. The original model is untouched.

Options worth knowing:

| Option | Default | Purpose |
|---|---|---|
| `updateeq` | required | New equations, multiple separated by newline. Each ends with `$`. |
| `add_add_factor` | False | If True, generate the `_A` companion. |
| `calc_add` | True | If True, calculate add-factors so the new equation fits the historical data exactly on construction. |
| `do_preprocess` | True | Run the full preprocessing pipeline (normalization, `Mexplode`, ...) before insertion. |
| `newname` | `''` | Name for the new model. Defaults to `'<old name> Updated'`. |
| `silent` | True | Suppress diagnostic prints. |

To remove an equation:

```python
mpak_v3, df = mpak.eqdelete('PAKNECONPRVTKN', newname='Pakistan without C eq')
```

Deleting an equation makes the variable exogenous in the new model. The user must supply its values in the input DataFrame.

---

## Importing from EViews

ModelFlow has dedicated grabbers for the two main EViews artifact types:

- **`.mod` files** (EViews Model Objects) — `modelgrab.py`. These are the structured model files EViews writes out from a Model object.
- **`.wf1` / `.wf2` workfiles** — `modelgrabwf2.py`. These need `py2eviews` installed (Windows + EViews-installed only) for the binary `.wf1` form. WF2 (text) files don't need EViews.

Typical EViews → ModelFlow flow:

```python
from modelgrab import EViewsModel

ev = EViewsModel('mymodel.mod')      # parse the EViews file
ev.test()                            # inspect equation types and originals
mmodel = ev.mmodel                   # ModelFlow model instance
mres   = ev.mres                     # auxiliary model that calculates add-factors
base   = ev.base_input               # input DataFrame with add-factors pre-calculated
```

What the grabber handles:

- Stripping EViews-specific syntax (`@` functions, EViews comment markers, `"..."` quoting).
- Translating `^` to `**`.
- Identifying behavioral vs. identity equations (EViews flags them differently).
- Generating add-factor calculation equations alongside the main model.
- Producing a "fitted values" sub-model when `make_fitted=True` is set.

When something doesn't translate cleanly — usually because the original `.mod` file uses an EViews function ModelFlow doesn't recognize — the fix is one of:

- Add a custom function via the `funks=[...]` argument when constructing the model. The function must be Python-callable and have a definition the normalizer can handle.
- Pre-process the equation text before passing it to the grabber, using the `replacements` argument (a list of `(pattern, replacement)` tuples).
- Rewrite the offending equation in ModelFlow syntax by hand.

---

## The `Mexplode` DSL: scalable model templates

For large structural models with repeated patterns (e.g. one consumption equation per region, one investment equation per sector), writing every equation by hand is tedious. `modelconstruct.py` provides `Mexplode`, which accepts a compact DSL with lists, `sum`, and `do ... enddo` loops and expands them into full FRMLs.

```python
from modelconstruct import Mexplode

template = """
LIST regions = US EU JP RW $

do region in regions:
    Y_{region} = C_{region} + I_{region} + NX_{region} $
    C_{region} = 0.6 * Y_{region}(-1) + C_{region}_A $
enddo

WORLD_GDP = sum(region in regions, Y_{region}) $
"""

mx = Mexplode(template, type_input='modelflow')
print(mx.expanded_frml)   # see the unrolled equations
mmodel = mx.mmodel        # the resulting ModelFlow model
```

`Mexplode` exposes the intermediate stages as properties (`clean_frml_statements`, `post_doable`, `post_do`, `post_sum`, `expanded_frml`, `normal_expressions`), so when something doesn't unroll the way you expected you can inspect each transformation in turn.

Two input modes:

- `type_input='modelflow'` — treat the input as a flat block of ModelFlow syntax.
- `type_input='markdown'` — extract model code from fenced blocks inside a markdown document. This is useful for "literate model files" where the model lives alongside its narrative.

`Mexplode` is also accessible via `model_specification.ipynb` in the repo, which is the user-facing tutorial — read that for worked examples before improvising.

---

## Adding user-defined functions

When equations need a function that isn't built into ModelFlow (e.g. a custom interpolation, a special transform), supply it via `funks=`:

```python
def my_smooth(x, alpha=0.7):
    """Exponentially smoothed lag."""
    return alpha * x + (1 - alpha) * x.shift(1)

mmodel = model(eqs, funks=[my_smooth], modelname='custom')
```

The function must:

- Take pandas-Series-shaped inputs and return the same.
- Be importable from a module so the JIT path (`ljit=True`) can find it. Lambdas defined in the notebook work in the non-JIT path but fail under JIT.
- Have a name that doesn't collide with built-ins (`LOG`, `EXP`, `DLOG`, `DIFF`, `SUM`, `IF`, ...).

`modeluserfunk.py` is where ModelFlow's bundled custom functions live; reading it shows the expected shape.

---

## Saving and reloading a constructed model

Same as any other model:

```python
mmodel.modeldump('mymodel.pcim', keep=False)   # equations + var descriptions + last data
mmodel.modeldump('mymodel.pcim', keep=True)    # also includes keep_solutions
```

Reload with `model.modelload('mymodel.pcim', run=True)`. The user's variable descriptions, var-groups, and stored reports survive the round-trip — make sure to populate them before dumping:

```python
mmodel.var_description = mmodel.var_description | {
    'Y':     'Gross Domestic Product',
    'C':     'Private Consumption',
    'I':     'Investment',
}

mmodel.var_groups['#Headline'] = ['Y', 'C', 'I']
```

For estimation-driven workflows (where the equations come out of fitted OLS/NLS rather than being typed by hand), see `model-construction-estimation.md`.
