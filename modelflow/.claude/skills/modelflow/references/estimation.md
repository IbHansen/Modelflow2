---
name: estimation
description: Guide to estimating econometric equations in ModelFlow — OLS, NLS via lmfit, NLS via EViews, inline constraints, and Makemodel integration.
---

# ModelFlow Estimation Guide

This skill is a practical reference for estimating econometric equations in ModelFlow. It covers the three estimator backends, how to specify constraints, and how to wire estimation into a `Makemodel` pipeline.

The estimation system lives in two files:

- **`modelestimator_new.py`** — the estimator backends and the `Eq_parent` contract
- **`modelconstruct_estimation.py`** — `Makemodel` integration (the `<estimator=...>` tag and friends)

## TL;DR

```python
from modelestimator_new import Estimate_ols, Estimate_nls_lmfit

# Plain OLS
est = Estimate_ols(
    org_eq="Y = C(1) + C(2)*X + C(3)*Z",
    input_df=df,
    smpl=(2002, 2018),
)
est.coef_estimate_dict   # {'C__1': 1.23, 'C__2': 0.45, ...}
est.org_eq_baked         # 'Y = (1.23) + (0.45)*X + (0.78)*Z'
est.mfresult             # LSResult wrapper for HTML reports

# NLS with bounds on C(2) and a derived C(3)
est = Estimate_nls_lmfit(
    org_eq="Y = C(1) + C(2)*X + C(3)*Z ST. C(2)=[0,1]; C(3):=C(1)*0.5",
    input_df=df,
    smpl=(2002, 2018),
)
```

## Choosing a backend

| Backend | Use when | Equation form |
|---|---|---|
| `Estimate_ols` | Linear-in-parameters, no constraints | `Y = C(1) + C(2)*X + C(3)*LOG(Z)` |
| `Estimate_nls_lmfit` | Nonlinear-in-parameters, bounds, fixed values, or derived params | `Y = C(1) * X^C(2)` |
| `Estimate_nls_eviews` | You have EViews installed and want EViews-style NLS output | Same as lmfit |

The `Estimate_nls` factory dispatches: `Estimate_nls(eq, solver='lmfit')` returns an `Estimate_nls_lmfit` instance.

**ECM:** `Estimate_nls_lmfit` defaults to `ecm=True`. The parser expects an error-correction form (`DLOG(Y) = ...`). Set `ecm=False` if you're estimating a level equation.

## Equation conventions

- Parameters are written as `C(1)`, `C(2)`, … (EViews-style). Internally they become `C__1`, `C__2`.
- Alternate parameter prefixes are allowed via `param_names`:

  ```python
  Estimate_nls_lmfit(
      org_eq="Y = ALFA + BETA*X",
      param_names=['ALFA', 'BETA'],
      ...
  )
  # ALFA -> ALFA__1, BETA -> BETA__1
  # Also accepts ALFA(2), BETA(3), etc.
  ```

- Lags: `X(-1)`, `X(-2)`.
- Transformations inside the equation: `LOG(X)`, `DLOG(X)`, `DIFF(X)`, `MOVAVG(X, n)`.
- The intercept is just `C(1)`; for a no-intercept regression, omit it.

## Inline constraints — the `ST.` clause

For nonlinear estimation via lmfit you can attach constraints directly to the equation, separated from the equation by `ST.` (subject to) and from each other by `;`:

```python
Estimate_nls_lmfit(
    org_eq="Y = C(1) + C(2)*X + ALFA*Z ST. C(1)=[0,1]; C(2)>0; ALFA:=C(1)+C(2)",
    param_names=['ALFA'],
    input_df=df,
)
```

### Constraint syntax reference

| Form | Meaning | lmfit kwargs |
|---|---|---|
| `NAME=[lo, hi]` | Bounded between `lo` and `hi` | `min=lo, max=hi` |
| `NAME > x` or `NAME >= x` | Lower bound | `min=x` |
| `NAME < x` or `NAME <= x` | Upper bound | `max=x` |
| `NAME = value` | Fixed (not estimated) | `value=v, vary=False` |
| `NAME ~ value` | Initial value (free to change) | `value=v, vary=True` |
| `NAME := expr` | Derived from other params | `expr='...'` |

**Notes:**
- Names can be `C(1)` or `ALFA` — they are normalized just like in the equation body, so `ALFA` and `ALFA(1)` are equivalent.
- **Mnemonic for `=` vs `~`:** `=` is a hard pin (does not move during estimation); `~` is "approximately" — a starting value the solver is free to update. They are otherwise identical.
- Multiple constraints on the same parameter accumulate: `C(1) ~ 0.5; C(1) >= 0` means "start at 0.5, stay non-negative."
- The `:=` form uses lmfit's `expr`: the parameter is computed, not fitted. Reference other parameters by their normalized name (`C__1`, `ALFA__1`).
- Inter-parameter inequalities like `C(1) > C(2)` are **not supported** — lmfit only handles simple bounds. Reparameterize as `C(2) = C(1) - DELTA; DELTA > 0`.

### Other ways to supply constraints

Three equivalent ways:

```python
# 1. Inline ST. (recommended — constraints stay with the equation)
Estimate_nls_lmfit(org_eq="Y = C(1)+C(2)*X ST. C(1)=[0,1]; C(2)>0")

# 2. As a kwarg (constraints is a normalized dict; kwarg wins on collisions)
Estimate_nls_lmfit(
    org_eq="Y = C(1)+C(2)*X",
    constraints={'C__1': {'min': 0, 'max': 1}, 'C__2': {'min': 0}},
)

# 3. Via Makemodel <constraints=...> tag (see Makemodel section)
> <estimator=nls_lmfit, constraints='C(1)=[0,1]; C(2)>0'> Y = C(1)+C(2)*X
```

## Initial values — three input paths

When nonlinear estimation has trouble converging, a good initial guess matters. There are three input paths; pick whichever fits your workflow.

### 1. `~` inside the `ST.` clause (recommended for one-offs)

```python
Estimate_nls_lmfit(
    org_eq="DLOG(Y) = C(1) + C(2)*DLOG(X) + C(3)*(LOG(Y(-1)) - LOG(X(-1))) "
           "ST. C(3) ~ -0.3; C(3) = [-1, 0]"
)
```

Reads naturally — the starting guess sits next to the bound. Best when you're iterating on a single equation in a notebook.

### 2. `default_params=` kwarg (existing API, dict-based)

```python
Estimate_nls_lmfit(
    org_eq="...",
    default_params={
        'C__1': {'value': 0.2},
        'C__3': {'value': -0.3, 'min': -1.0, 'max': 0.0},
    },
)
```

Useful when:
- You're driving estimation programmatically and already have a dict of starting values from a previous fit.
- You want to keep the equation string clean and put all the numeric details in code.

### 3. `constraints=` kwarg (same shape as `default_params`)

```python
Estimate_nls_lmfit(
    org_eq="...",
    constraints={'C__3': {'value': -0.3, 'min': -1.0, 'max': 0.0}},
)
```

Functionally identical to `default_params` for initial-value purposes. The two differ only by name and by precedence: when a parameter appears in both, `constraints` wins. Use `constraints` for things you'd express in an `ST.` clause, `default_params` for "default starting values I'd accept any of."

### Which one should I reach for?

| Situation | Reach for |
|---|---|
| Iterating in a notebook, want to see initial value next to the equation | `ST. C(1) ~ 0.5` |
| Programmatic pipeline, starting values come from a previous fit | `default_params={...}` |
| `Makemodel` with many estimated equations, each needing different initial values | `<constraints='C(1)~0.5; ...'>` on each line |
| You want the initial value to be a hard fallback the solver can override only inside bounds | `default_params={'C__1': {'value': 0.5}}` + `ST. C(1)=[0,1]` |

All three feed the same `params.add(name, **kwargs)` call inside `Estimate_nls_lmfit._fit()` — they merge with `default_params` lowest priority, `constraints` (inline `ST.` and kwarg) highest.

### Which backends accept constraints?

- `Estimate_nls_lmfit`: **yes**
- `Estimate_ols`, `Estimate_nls_eviews`: **no** — passing constraints raises `ValueError` at construction.

## Inspecting an estimator object

Every `EstimatorBackend` instance exposes:

| Attribute | What it is |
|---|---|
| `org_eq` | Original equation (after uppercasing and param normalization, `ST.` clause stripped) |
| `org_eq_clean` | Same as `org_eq` with any `coef_dict` substitutions applied |
| `org_eq_baked` | Equation with estimated coefficients substituted in place (the "result" equation) |
| `coef_estimate_dict` | `{'C__1': value, …}` |
| `coef_ser` | Same as a pandas Series |
| `regression_model` | Native fit result (statsmodels / lmfit / EViews spool) |
| `mfresult` | `LSResult` wrapper — call `.get_html_report()` for the full HTML view |
| `constraints` | The parsed-and-merged constraint dict |
| `c_params` | Sorted list of parameter placeholders found in the equation |
| `endo_var` | The endogenous variable name |
| `estimation_smpl` | The sample actually used (after NaN trimming for OLS) |

## Makemodel integration

`Makemodel` (from `modelconstruct_estimation.py`) accepts a *markdown* model text where each equation lives on a line beginning with `>` and continuation lines begin with `>>`. Tags such as `<estimator=...>` attach to the equation. The parser extracts these lines (ignoring surrounding prose), runs the estimator for each tagged equation, and bakes the estimated coefficients into the equation before normalization.

### Markdown equation format

- `>` starts an equation. No `FRML` keyword, no `$` terminator.
- `>>` continues the previous equation on the next line (the two are joined with a space).
- Tags written inside `< ... >` attach metadata to the equation; `<estimator=...>` is the most important one.
- Lines not starting with `>` are prose — they are ignored by the model parser, so the file can mix documentation with equations.

### Basic usage

```python
from modelconstruct_estimation import Makemodel
from modelestimator_new import Estimate_nls

ls = Estimate_nls.with_defaults(input_df=df, smpl=(2002, 2018))

mm = Makemodel("""
Some prose explaining the model.

> <estimator=ls> DLOG(Y) = C(1) + C(2)*DLOG(X) 
>> + C(3)*(LOG(Y(-1)) - LOG(X(-1)))
> <ident> Z = Y * 0.5
""")
mmodel = mm.mmodel              # ready-to-solve modelflow model
mm.estimation_report()          # writes makemodel_estimation_report.html
```

### Supported tags

| Tag | Meaning | Example |
|---|---|---|
| `<estimator=ls>` | Use the local-namespace `ls` object as the estimator class | `<estimator=ls>` |
| `<estimator=nls_lmfit>` | Use a registered estimator class | `<estimator=nls_lmfit>` |
| `<est=ls>` | Short alias for `<estimator=...>` | `<est=ls>` |
| `<smpl=2002 2018>` | Override sample for this equation | `<smpl=2010 2020>` |
| `<caption='...'>` | Override caption shown in the report | `<caption='Inflation eq'>` |
| `<constraints='...'>` | Append constraints (same syntax as `ST.` clause) | `<constraints='C(1)=[0,1]; C(2)>0'>` |
| `<ident>` | Identity (no estimation) — equation is a definition | `<ident> Z = Y * 0.5` |

### Constraints from a tag

These three forms produce the same estimator state:

```text
A. Inline ST.
> <estimator=ls> Y = C(1)+C(2)*X ST. C(1)=[0,1]

B. Tag
> <estimator=ls, constraints='C(1)=[0,1]'> Y = C(1)+C(2)*X

C. Both — merged together (tag wins on key collisions)
> <estimator=ls, constraints='C(2)>0'> Y = C(1)+C(2)*X ST. C(1)=[0,1]
```

### Templating with `replacements=`

`Makemodel` accepts a `replacements=` kwarg that applies one or more `(old, new)` string substitutions to the model text *before* parsing. This is a plain `str.replace` — no regex — and it operates on the raw formula text.

Two common uses:

**1. Keeping the model parsimonious.** Write the equation once with a short placeholder, substitute in the real variable name at construction time:

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

**2. Same template, many entities (banks, countries, sectors).** Write the model once with a sentinel like `__dim`, then loop:

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
# bank_models['IB'].mmodel has LOANS_IB, GDP_IB, ...
# bank_models['SOREN'].mmodel has LOANS_SOREN, GDP_SOREN, ...
```

**Accepted shapes:**

```python
replacements=('OLD', 'NEW')               # single pair
replacements=[('A', 'B'), ('C', 'D')]     # list, applied in order
replacements=[]                           # no-op (default)
```

**Notes:**

- Substitution is order-sensitive: later pairs see the output of earlier ones, so make the sentinels unique enough that early replacements don't collide with later ones (the leading `__` convention helps).
- `replacements=` substitutes blindly — it will also touch any non-equation prose between the `>` lines. Keep sentinels (`__dim`, `__BANK_PLACEHOLDER`) distinctive so this doesn't matter in practice.
- Compared with `LIST`/`TLIST` blocks: `LIST` is the right tool when the *same model instance* needs to enumerate sublists at compile time (e.g. one set of equations that loops over all banks inside a single solve). `replacements=` is the right tool when you want *one model instance per entity*, each solvable independently. Use `LIST` for "all banks in one model"; use `replacements=` for "one model per bank."

## Recipes

### Bound the speed-of-adjustment coefficient in an ECM

```python
Estimate_nls_lmfit(
    org_eq=(
        "DLOG(Y) = C(1) + C(2)*DLOG(X) + C(3)*(LOG(Y(-1)) - LOG(X(-1))) "
        "ST. C(3) ~ -0.3; C(3) = [-1, 0]"
    ),
    input_df=df,
)
# Starts at -0.3 (a typical ECM speed-of-adjustment), stays in (-1, 0).
```

### Pin one coefficient, estimate the rest

```python
Estimate_nls_lmfit(
    org_eq="Y = C(1) + C(2)*X + C(3)*Z ST. C(3)=0.5",
    input_df=df,
)
# C(3) is held at 0.5 (vary=False) while C(1) and C(2) are estimated.
```

### Tie two coefficients together

```python
Estimate_nls_lmfit(
    org_eq="Y = C(1)*X + C(2)*Z + C(3)*W ST. C(3):=C(1)+C(2)",
    input_df=df,
)
# C(3) is computed as C(1)+C(2) — not a free parameter.
```

### Estimate one equation in a larger model

```python
mm = Makemodel("""
A small consumption model.

> <estimator=ls> LOG(C) = C(1) + C(2)*LOG(Y) + C(3)*R
> <ident>       S = Y - C
> <ident>       I = S - DEF
""", input_df=df, estimator=ls)
```

Only the first equation is estimated; the others remain as identities.

## Authoring a new estimator backend

Subclass `EstimatorBackend` and implement two methods:

```python
@dataclass
class Estimate_gls(EstimatorBackend):
    sigma: Optional[np.ndarray] = None
    ecm: bool = False

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        df = self.estimation_df
        y, X = df["LHS"], df.drop(columns=["LHS"])
        result = sm.GLS(y, X, sigma=self.sigma).fit()
        return result, self._map_statsmodels_params(result.params.to_dict())

    def _render_summary_html(self) -> str:
        return self.regression_model.summary().as_html()
```

That's it. Sample handling, A/F plots, `org_eq_baked` substitution, `LSResult` HTML, and `Makemodel` integration all work automatically.

To opt into constraints: set `_supports_constraints: ClassVar[bool] = True` and use `self.constraints` in `_fit()`.

To reject equation forms the backend can't handle: override `_validate_equation()` and raise `ValueError` with a helpful message.

## Common pitfalls

- **`Estimate_ols`: "no usable rows in sample"** — lags or `LOG` transforms produced NaN/inf rows. Either widen the sample, fix the data, or remove the problematic transformation.
- **`Estimate_ols`: "no parameters to estimate"** — the equation has no `C(...)` placeholders. There's nothing to fit. Add at least `C(1)`.
- **`Estimate_ols`: "no intercept" warning** — fitting through the origin. Usually unintentional; add `C(1)` if you want an intercept.
- **`ValueError: ... does not support constraints`** — you supplied constraints to OLS or EViews. Switch to `Estimate_nls_lmfit` or remove the `ST.` clause / `constraints=` kwarg.
- **`ValueError: constraints reference unknown parameters`** — typo: e.g. `ST. C(7)=[0,1]` when the equation only uses `C(1)..C(3)`. Either fix the constraint or add the missing parameter to the equation.
- **lmfit doesn't converge / coefficients blow up** — supply better initial values via `default_params={'C__1': {'value': 0.5}, ...}` or tighten bounds with the `ST.` clause.

## Reports

- **Single estimator:** `est.mfresult.get_html_report()` returns a full HTML report (regression summary + equation listing + actual-vs-fitted plot). Use `est.mfresult.show()` to open in a browser.
- **From a Makemodel:** `mm.estimation_report(path='html', filename='out.html', open_file=True)` exports all estimations in the model.

## Related modules

- `modelnormalize.py` — turns `DLOG(Y) = ...` into a solvable form; called by `Eq_parent`
- `modelclass.py` — the core `model` class; estimated equations flow into this
- `modelmf.py` — the `.mfcalc` pandas accessor used by OLS to materialize regressors
- `modelestimation.py` — *older* estimation module; superseded by `modelestimator_new.py`. Avoid for new work.
