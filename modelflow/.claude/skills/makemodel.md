---
name: makemodel
description: Guide to building ModelFlow models with the Makemodel class — markdown equation format, tags, LIST/DO templating, replacements, estimation integration, and report generation.
---

# Makemodel Guide

`Makemodel` is the high-level entry point for authoring ModelFlow models. It takes a *markdown*-style text containing prose mixed with equations, expands LIST/DO templates, runs any tagged estimations, normalizes the result, and exposes a ready-to-solve `model` object.

This skill covers the authoring workflow end-to-end. For deeper detail on estimator backends (OLS, lmfit, EViews, constraints), see the **estimation** skill.

`Makemodel` lives in `modelconstruct_estimation.py` (and a non-estimation flavor in `modelconstruct.py` — use the `_estimation` one unless you specifically don't want estimation support).

## TL;DR

```python
from modelconstruct_estimation import Makemodel
from modelestimator_new import Estimate_nls

ls = Estimate_nls.with_defaults(input_df=df, smpl=(2002, 2018))

mm = Makemodel("""
A small consumption model.

> <estimator=ls> LOG(C) = C(1) + C(2)*LOG(Y) + C(3)*R
> <ident>       S = Y - C
> <ident>       I = S - DEF
""", input_df=df, estimator=ls)

mm.mmodel              # ready-to-solve modelflow model
mm.show                # print normalized FRMLs
mm.draw                # visualize the dependency graph
mm.estimation_report() # write makemodel_estimation_report.html
```

## The markdown equation format

`Makemodel` reads markdown text where:

| Line prefix | Meaning |
|---|---|
| `>` | An equation. No `FRML` keyword, no `$` terminator. |
| `>>` | Continues the previous equation (the two lines are joined with a space). |
| `>list ...` | A LIST or TLIST definition (see Templating section). |
| anything else | Prose. Ignored by the parser — you can mix documentation and equations freely. |

Tags written inside `< ... >` attach metadata to the equation. `<estimator=...>` triggers estimation; `<ident>` declares an identity; others (`<smpl>`, `<caption>`, `<constraints>`, `<endo>`, `<stoc>`, etc.) modify behavior.

### Continuation example

```text
A long equation broken over several lines:

> <estimator=ls> DLOG(Y) = C(1) + C(2)*DLOG(X)
>>                       + C(3)*DLOG(Z)
>>                       + C(4)*(LOG(Y(-1)) - LOG(X(-1)))
```

is equivalent to

```text
> <estimator=ls> DLOG(Y) = C(1) + C(2)*DLOG(X) + C(3)*DLOG(Z) + C(4)*(LOG(Y(-1)) - LOG(X(-1)))
```

## Tags reference

| Tag | Meaning |
|---|---|
| `<ident>` | Identity (definition). The default if no other tag is present. |
| `<estimator=NAME>` | Run estimator `NAME` on this equation; bake the estimated coefficients in. |
| `<est=NAME>` | Short alias for `<estimator=NAME>`. |
| `<smpl=START END>` | Override the estimation sample for this equation. |
| `<caption='...'>` | Override the caption shown in the report. |
| `<constraints='...'>` | Constraints on estimated parameters (same syntax as inline `ST.`). |
| `<stoc>` | Stochastic equation — add an add-factor variable so the model can be calibrated to history. |
| `<exo>` | Exogenize the LHS — add a fixable add-factor variable. |
| `<fit>` | Generate a fitted-value variant alongside the equation. |
| `<endo=NAME>` | Override which variable on the equation is endogenous. |
| `<endo_lhs=FALSE>` | The LHS is not the endogenous variable (use with `<endo=...>`). |
| `<implicit>` | Equation is implicit (cannot be solved by simple substitution). |

Multiple tags can appear in one `< ... >` block, separated by commas:

```text
> <estimator=ls, smpl=2010 2018, constraints='C(2)>0'> DLOG(Y) = C(1) + C(2)*DLOG(X)
```

## Templating

### LIST blocks — enumerated sublists

A `LIST` defines a named set with optional sublists. The pattern is `LIST name = sublist : values / sublist : values / ...`. Inside equations, `{name}` expands to each value and `{sublist}` expands to the parallel value from another sublist.

```text
>list banks = banks   : IB      SOREN  MARIE /
>>            country : DENMARK SWEDEN DENMARK /
>>            selected : 1       1      0

>list sectors = sectors : NFC SME HH

Loss equation, expanded for every bank × sector:

> doable LOSS__{banks}__{sectors} = HOLDING__{banks}__{sectors} * PD__{banks}__{sectors}
```

After expansion, `Makemodel` produces nine equations (3 banks × 3 sectors), e.g. `LOSS__IB__NFC = HOLDING__IB__NFC * PD__IB__NFC`.

### TLIST — transposed list

When the natural reading is columns-by-rows rather than rows-by-columns, `TLIST` accepts the transposed form and rewrites it internally to a regular `LIST`:

```text
>tlist banks = banks country selected /
>>             IB    DENMARK 1 /
>>             SOREN SWEDEN  1 /
>>             MARIE DENMARK 0
```

is equivalent to the `LIST` example above.

### DO / ENDDO — explicit loops

For control over which list to loop and which conditions apply:

```text
> do banks
>   £ comment: equations for bank {banks} in {country}
>   x_{banks} = 42
> enddo
```

The `£` symbol is the BLL comment character (comments are preserved into the FRML output but not parsed).

### DOABLE — single-line DO

`doable` (also written `DOABLE`) is a one-line loop over every LIST appearing inside `{...}`:

```text
> doable LOSS__{banks}__{sectors} = HOLDING__{banks}__{sectors} * PD__{banks}__{sectors}
```

Equivalent to nested `do banks / do sectors / ... / enddo / enddo` but more compact.

### Filtering with `[sublist=value]`

You can restrict a `doable` to a subset using a sublist condition:

```text
> doable [banks selected=1] x_{banks} = 42
```

Only generates equations for banks where `selected = 1`.

### `replacements=` — Python-side string substitution

Apply one or more `(old, new)` string substitutions to the model text *before* parsing. Useful for two patterns:

**Parsimony** — write a placeholder once, substitute the real name at construction:

```python
template = """
> <estimator=ls> LOG(CONS) = C(1) + C(2)*LOG(__INCOME)
"""

mm = Makemodel(
    template,
    replacements=[('__INCOME', 'GDP_REAL_DISP')],
    input_df=df,
    estimator=ls,
)
```

**Same template, many entities** — write the model once, then loop:

```python
template = """
> <estimator=ls> LOG(LOANS__dim) = C(1) + C(2)*LOG(GDP__dim)
> <ident>       NPL__dim = LOANS__dim * NPL_RATIO__dim
"""

bank_models = {
    bank: Makemodel(
        template,
        replacements=[('__dim', f'_{bank}')],
        input_df=df,
        estimator=ls,
    )
    for bank in ['IB', 'SOREN', 'MARIE']
}
```

**Accepted shapes:**

```python
replacements=('OLD', 'NEW')              # single pair
replacements=[('A', 'B'), ('C', 'D')]    # list, applied in order
replacements=[]                          # no-op (default)
```

**Notes:**
- Plain `str.replace` — no regex. Make sentinels distinctive (`__dim`, `__INCOME`, …) so they don't collide.
- Substitution is order-sensitive: later pairs see earlier output.
- Operates on the entire model text — including prose between `>` lines.
- Use `LIST` for "all banks in one model"; use `replacements=` for "one model per bank."

## Estimation integration

Any equation tagged with `<estimator=NAME>` is estimated during `Makemodel.__post_init__`. The estimated coefficients are baked into the equation before normalization, so the resulting `mmodel` has numeric coefficients in place of `C(1)`, `C(2)`, …

### Basic estimation

```python
from modelconstruct_estimation import Makemodel
from modelestimator_new import Estimate_nls

# Factory captures shared defaults
ls = Estimate_nls.with_defaults(input_df=df, smpl=(2002, 2018))

mm = Makemodel("""
> <estimator=ls> DLOG(Y) = C(1) + C(2)*DLOG(X) + C(3)*(LOG(Y(-1)) - LOG(X(-1)))
> <ident>       Z = Y * 0.5
""", input_df=df, estimator=ls)
```

The estimator name `ls` is resolved from the caller's namespace (so `Estimate_nls.with_defaults(...)` stored in a local variable just works), or from an explicit `estimator_classes={'ls': ...}` mapping if you prefer.

### Estimation-related `Makemodel` kwargs

| Kwarg | Purpose |
|---|---|
| `input_df=` | DataFrame fed to each estimator. Each estimator may override via its own `input_df`. |
| `estimator=` | Default estimator for `<estimator>` flags with no value. |
| `smpl=` | Default estimation sample. `<smpl=START END>` per equation overrides. |
| `estimator_kwargs={}` | Shared kwargs (e.g. `caption`, `add_add_factor`) passed to every estimator. |
| `estimator_classes={}` | Explicit name → class mapping (alternative to caller-namespace resolution). |
| `estimator_namespace={}` | Namespace dict for resolving `<estimator=ls>`-style names. |

### Constraints on estimated parameters

Two equivalent forms:

**Inline `ST.` clause** (constraints stay with the equation):

```text
> <estimator=ls> Y = C(1) + C(2)*X ST. C(1)~0.5; C(2)>0; C(2)<=1
```

**`<constraints=...>` tag** (separate from the equation):

```text
> <estimator=ls, constraints='C(1)~0.5; C(2)>0; C(2)<=1'> Y = C(1) + C(2)*X
```

Recognized constraint forms (full reference in the **estimation** skill):

| Form | Meaning |
|---|---|
| `NAME=[lo, hi]` | Bounded between `lo` and `hi` |
| `NAME > x` / `>= x` | Lower bound |
| `NAME < x` / `<= x` | Upper bound |
| `NAME = value` | Fixed (not estimated) |
| `NAME ~ value` | Initial value (free to change) |
| `NAME := expr` | Derived from other params |

Constraints work with `Estimate_nls_lmfit`. OLS and EViews raise an error if constraints are supplied.

### Inspecting estimations

After construction:

```python
mm.estimation_records      # list of dicts, one per estimated equation
mm.estimation_report()     # writes HTML report (all estimations)
mm.markdown_with_estimation  # original markdown + inline coefficient tables
mm.render_est              # interactive display of the above
```

Each `estimation_records` entry contains:

```python
{
    'equation_index': int,           # position in the original model
    'frmlname': str,                 # original FRML name flag
    'estimator': str,                # display name of the estimator
    'smpl': tuple,                   # sample used
    'original_expression': str,      # before baking
    'baked_expression': str,         # with coefficients substituted
    'estimator_object': EstimatorBackend,  # the live estimator
}
```

Use `record['estimator_object'].mfresult.get_html_report()` for the full single-equation HTML view.

## Outputs and useful attributes

| Attribute | What it is |
|---|---|
| `mm.mmodel` | The full ready-to-solve `model` (cached). Use this for forecasting/simulation. |
| `mm.clean_model` | A `model` containing only the core equations (no add-factor or fitted variants). |
| `mm.add_model` | A `model` that *calculates* add factors from historic data. |
| `mm.normal_frml` | The fully normalized FRML text. |
| `mm.clean_frml` | Normalized FRMLs with `ADD`/`EXO`/`FIT` options stripped. |
| `mm.show` | Prints `normal_frml`. |
| `mm.draw` | Plots the dependency graph (uses `modelnet`). |
| `mm.modellist` | Dict of parsed LIST definitions. |
| `mm.showlists` | Pretty-prints the LIST definitions. |
| `mm.render` | Renders the markdown source (equations + prose) in a notebook. |
| `mm.render_est` | Same as `render`, plus estimation tables inline. |
| `mm.estimation_records` | List of estimation result records (see above). |

## Aligning to historic data — `init_addfactors`

For models with `<stoc>` or `<exo>` tags, ModelFlow generates add-factor variables (`*_A`). `init_addfactors` computes the add factors needed to make the model exactly reproduce a historic dataframe:

```python
aligned_df = mm.init_addfactors(df, start=2002, end=2020, show=True, check=True)
# aligned_df has the add-factor columns filled in
# show=True prints them; check=True re-simulates to verify
```

After this, `mm.mmodel(aligned_df, ...)` reproduces history exactly, and any change to a variable produces a counter-factual.

## Combining models — `+`

Two `Makemodel` instances can be concatenated:

```python
core   = Makemodel(core_equations)
shocks = Makemodel(shock_equations)
full   = core + shocks
full.mmodel   # contains both blocks
```

LIST definitions are merged. Useful for keeping a base model in one cell and scenario overrides in another.

## Recipes

### Quick identity-only model

```python
mm = Makemodel("""
> Y = C + I + G + X - M
> S = Y - T
> DEF = G - T
""")
mm.mmodel
```

(No `<ident>` needed — equations without tags default to identity.)

### One estimated equation in a larger model

```python
mm = Makemodel("""
Consumption is estimated; the other equations are identities.

> <estimator=ls> LOG(C) = C(1) + C(2)*LOG(Y) + C(3)*R
> S = Y - C
> I = S - DEF
""", input_df=df, estimator=ls)
```

### Mixed estimators

```python
mm = Makemodel("""
> <estimator=nls_lmfit> DLOG(C) = C(1) + C(2)*DLOG(Y) + C(3)*(LOG(C(-1)) - LOG(Y(-1))) ST. C(3)=[-1,0]
> <estimator=ols>       LOG(I) = C(1) + C(2)*LOG(Y) + C(3)*R
> <ident>               S = Y - C
""", input_df=df)
```

### Stochastic equation (add-factor variant)

```python
mm = Makemodel("""
> <estimator=ls, stoc> LOG(C) = C(1) + C(2)*LOG(Y)
""", input_df=df, estimator=ls)

mm.init_addfactors(df_history)  # populates LOG_C_A so model reproduces history
```

### Bank-by-bank credit-loss model

```python
template = """
> <estimator=ls> LOG(LOSS__bank) = C(1) + C(2)*LOG(GDP__bank) + C(3)*UR__bank
> <ident>       NPL__bank = LOSS__bank * NPL_FACTOR__bank
"""

models = {
    b: Makemodel(template, replacements=[('__bank', f'_{b}')], input_df=df, estimator=ls)
    for b in ['IB', 'SOREN', 'MARIE']
}
```

### Many sectors inside a single model

```python
mm = Makemodel("""
>list sectors = sectors : NFC SME HH

> doable <estimator=ls, stoc> LOG(LOSS__{sectors}) = C(1) + C(2)*LOG(GDP)
""", input_df=df, estimator=ls)
# Three estimated equations: LOSS_NFC, LOSS_SME, LOSS_HH
```

## Common pitfalls

- **`<estimator=ls>` not found** — make sure `ls` is in the caller's scope, or pass `estimator_classes={'ls': ls}` / `estimator_namespace=locals()`.
- **Equations don't expand inside DO blocks** — check that you're using `{name}` (with braces) not `name`.
- **`replacements=` matched too much** — your sentinel collided with a real token. Use distinctive prefixes (`__bank`, not `BANK`).
- **`replacements=` is plural** — the kwarg name is `replacements`, not `replace` (a common typo).
- **Estimator can't take constraints** — OLS and EViews backends raise on any `ST.` clause or `<constraints=...>` tag. Switch to `Estimate_nls_lmfit`.
- **Estimation sample shrinks unexpectedly** — OLS drops NaN/inf rows from lags and `LOG()` transforms; the actual sample is reported in `estimator_obj.estimation_smpl`.
- **`mm.mmodel` is cached** — re-running estimation requires constructing a new `Makemodel`. The cached model property doesn't refresh automatically.

## Related modules and skills

- **estimation** skill — deeper detail on the estimator backends (OLS / lmfit / EViews), constraint forms, initial-value strategies, and authoring new backends.
- `modelconstruct.py` — same dataclass without estimation hooks. Use the `_estimation` variant unless you specifically want the smaller surface.
- `modelclass.py` — the underlying `model` class that `mm.mmodel` returns.
- `modelnormalize.py` — handles `DLOG`/`DIFF`/`MOVAVG` rewriting; called from `Makemodel.__post_init__`.
- `modelmanipulation.py` — `tofrml`, `dounloop`, `sumunroll`; the LIST/DO expansion engine.
- `modelpattern.py` — `kw_frml_name` extracts tag values like `<estimator=NAME>`.
