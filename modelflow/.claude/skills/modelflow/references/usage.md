# Everyday usage: load, update, mfcalc, plot, reports

The five things below cover the bulk of day-to-day ModelFlow use. Deep reference material lives in `MFMod_Python_Modelflow.pdf` chapters 7, 9, 11, 12, and 14.

---

## 1. Load a model with `modelload`

`model.modelload()` is a **staticmethod** on the `model` class. It reads a `.pcim` file (zipped JSON with equations, data, descriptions, and optionally previously-kept scenarios) and returns a `(model_instance, dataframe)` tuple.

```python
mpak, baseline = model.modelload(
    '../models/pak.pcim',   # path or URL
    run=True,               # simulate immediately using saved options
    keep='Baseline',        # store the first solution under this name
    alfa=0.7,               # solver step-size in (0,1)
)
```

Key points:

- First return value is the **model object** (conventionally named after the country, e.g. `mpak`, `mtur`, `mnpl`); second is a **DataFrame** with input data (and simulation results, if `run=True`).
- `keep='Baseline'` puts a copy of the result in `mpak.keep_solutions['Baseline']`. Multi-scenario plots and reports read from this dictionary.
- `run=False` (default) loads without solving — useful when the user wants to inspect equations or modify input data before the first solve.
- After loading, `mpak.basedf` holds the baseline DataFrame and `mpak.lastdf` the most recent simulation. Both are reassignable.
- Save a working model (with kept scenarios) using `mpak.modeldump('path.pcim', keep=True)`. The `.pcim` extension is conventional.
- Public World Bank models can be downloaded via `model.Worldbank_Models()`. See `worldbank-onboarding.md` for details.

Useful properties on a loaded model: `.name`, `.equations`, `.var_description` (dict of mnemonic → long description), `.var_groups`, `.current_per` (active sample), `.keep_solutions`, `.reports`.

---

## 2. Update a DataFrame with `.upd`

`.upd()` is a method ModelFlow attaches to pandas DataFrames at import. It is the natural way to build a shocked input DataFrame from a baseline. It returns a **new** DataFrame — no in-place modification unless the user reassigns.

```python
shocked = baseline.upd('<2025 2030> PAKGGREVCO2CER = 30')
```

The string is a small DSL. Each non-blank, non-comment line:

```
[<begin end>]  VAR1 [VAR2 ...]  OP  VALUE [VALUE ...]   [# comment]
```

Operators:

| Op | Meaning |
|---|---|
| `=` | set the variable equal to the value |
| `+` | add the value to the variable |
| `*` | multiply by the value |
| `%` | multiply by `(1 + value/100)` — percent change |
| `=growth` | set the growth rate (percent) to the value |
| `+growth` | add the value to the current growth rate |
| `=diff` | set the period-to-period change to the value |

Common gotchas (flag these when reviewing code):

- **Spaces around the operator are required.** `df.upd('A = 7')` works; `df.upd('A =7')` raises. Same for `A * 1.1` vs `A*1.1`.
- **Lowercase variable names are silently uppercased.** `df.upd('c = 142')` creates a column `C`.
- **Time scope inherits from the previous line** until reset. Use `<-0 -1>` to reset to the full sample (`-0` = first row, `-1` = last). A single date like `<2023>` means one point.
- Number of values must be **1** (broadcast) or **exactly** the number of rows in the time window.
- `create=False` raises on unknown LHS variables — useful for typo-catching. Default is `True`.
- `keep_growth=True` (or per-line `--keep_growth_rate` / `--kg`) keeps the post-window growth rate the same as before the update. This is the right behavior for "ex-ante real shock" exercises.
- `lprint=True` prints before/after/diff tables per line — invaluable when debugging.
- `scale=<float>` multiplies every input value; makes families of varying-severity shocks trivial.

Multi-line example:

```python
ct30 = baseline.upd("""
    <2025 2100>  PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30
    <2025 2030>  PAKNECONGOVTKN  % 2          # +2% on government consumption
    <2025>       PAKBNCABFUNDCD =diff -1000   # one-off CAB shift in 2025
""")
```

`.upd` only handles literal right-hand sides — `df.upd('A = B')` does not work. For anything involving other variables, lags, or transforms, use `.mfcalc`.

---

## 3. Transform a DataFrame with `.mfcalc`

`.mfcalc()` is also DataFrame-attached and far more powerful than `.upd`. It compiles a mini-model on the fly and solves it on the DataFrame, so it understands full equations, lags, leads, `diff()`, `dlog()`, conditionals — the same business logic the main model uses.

Use it for:

- Initializing or transforming variables as functions of other variables (e.g. inflation-adjusted shocks).
- Recursive calculations with lags (accumulating a stock from a flow).
- Inspecting how ModelFlow normalizes an equation (`showeq=True`).

```python
df2 = df.mfcalc('x = x(-1) + a')            # recursive: x accumulates a
df3 = df.mfcalc('a = 1.02 * a(-1)')         # a grows at 2 percent
df.mfcalc('dlog(a) = 0.02', showeq=True)    # prints: A=EXP(LOG(A(-1))+0.02)
df.mfcalc('diff(a) = 2', showeq=True)       # prints: A=A(-1)+(2)
```

Behaviors that surprise newcomers:

- **New variables are initialized to zero**, not NaN. So `df.mfcalc('x = x(-1) + a')` gives `x = 0` in the first row (no lag-1 available) and accumulates from there. Leading zero often looks like a bug; it isn't.
- **Equations in a single call are solved simultaneously.** If a call has two equations and the second references the first, the second sees *intermediate* values of the first (the rows where the lag was defined), not the user's mental "final" values. When this matters, split into two `.mfcalc` calls — the output `df` of the first feeds the second.
- **A `<begin end>` clause on its own line restricts time scope** for subsequent equations in the same call.
- LHS may not contain transformations: write `diff(a) = 2`, not `2 = diff(a)`. Don't put expressions on the left.

Realistic pattern — building an inflation-adjusted carbon tax path:

```python
real_ct = baseline.mfcalc("""
    <2025 2100>
    PAKGGREVCO2CER = 30 * (PAKNYGDPMKTPXN / PAKNYGDPMKTPXN(2025))
""")
result = mpak(real_ct, 2020, 2100, keep='Real $30 carbon tax')
```

---

## 4. Plot results with `.plot` (and `keep_plot`)

`.plot()` is a **model method** that builds a figure object from `.basedf`, `.lastdf`, and the kept scenarios. It returns a `DisplayKeepFigDef` — nothing displays by itself. To render: evaluate the object alone in a cell, call `display(fig)`, use `fig.show`, or `fig.pdf()`.

```python
mpak.smpl(2020, 2040)   # restrict plot/display window

fig = mpak.plot(
    pat='*NYGDPMKTPKN *NECONPRVTKN',  # space-separated wildcard pattern
    name='gdp_and_consumption',
    title='GDP and consumption',
    datatype='growth',                 # 'level', 'growth', 'diflevel', 'difpctlevel', ...
    samefig=True,                      # arrange subplots in a grid
    by_var=True,                       # one panel per variable
    legend=True,
)
fig.show
```

Worth knowing:

- `pat` uses ModelFlow's wildcard syntax: `*` = anything, `?` = one character. Leading `!` switches to **variable descriptions** — `mpak['!*Carbon*']` finds every variable whose description mentions Carbon.
- `datatype`: `level`, `growth`, `change`, `diflevel`, `difpctlevel`, `difgrowth`, `gdppct`, `qoq_ar`. Default `growth`.
- `scenarios=` controls which kept solutions appear. Use `|`-separated exact names, `*` for all, or the special string `base_last` for just `.basedf`/`.lastdf`. Display names can be overridden via `mpak.basename = '...'` and `mpak.lastname = '...'`.
- `ax_title_template=` customizes panel titles. Placeholders: `{var_name}`, `{var_description}`, `{compare}`.
- `samefig=True` lays out everything in one figure (controlled by `ncol`); `False` produces an accordion widget.
- Two plot objects can be joined side-by-side with `|`: `fig1 | fig2` combines different datatypes in one figure.
- `fig.set_options(...)` updates everything except line-level options (`pat`, `datatype`, `mul`, `ax_title_template`), which must be set at creation.
- `fig.savefigs(location='./graph', experimentname='exp1', extensions=['svg','png'])` writes matplotlib figures to disk.

There's also a lower-level `mpak.keep_plot(...)` that returns a dict of raw matplotlib figures — useful for ad-hoc plotting outside the report framework. Same `pat`/`scenarios`/`showtype`/`diff`/`diffpct` ideas, but procedural.

The `[]` selector is the most concise route for exploratory charts:

```python
mpak['PAKNYGDP??????'].rename().plot(title='Wild-card selection')
mpak['PAKNYGDPMKTPKN PAKNECONPRVTKN'].difpctlevel.plot()
```

The selector exposes transformation properties (`.dif`, `.growth`, `.pct`, `.difpct`, `.difpctlevel`, `.yoy_ar`, `.qoq_ar`) before `.plot()` is called.

---

## 5. Reports: tables, text, and how they compose

ModelFlow's report system has four building blocks. All are model methods returning *display definition* objects — nothing renders until asked:

| Builder | Returns | Purpose |
|---|---|---|
| `mpak.table(...)` | `DisplayVarTableDef` | Tabular view of `.basedf`/`.lastdf` |
| `mpak.plot(...)` | `DisplayKeepFigDef` | Figures (section 4) |
| `mpak.text('...')` | text display | Inline text with optional `<html>` / `<latex>` / `<markdown>` blocks |
| `mpak.report(...)` / `+` operator | report container | Tables, plots, text, other reports |

Typical table call:

```python
tab = mpak.table(
    pat=' *NYGDPMKTPKN *NECONPRVTKN *NEGDIFTOTKN *NEEXPGNFSKN *NEIMPGNFSKN ',
    name='gdp_components',
    title='GDP components',
    foot='Source: World Bank',
    datatype='growth',
    dec=2,
)
tab           # renders HTML in Jupyter
tab.show      # plain-text rendering
tab.pdf()     # LaTeX → PDF (requires latex install)
```

Data comes from `.basedf` and `.lastdf` by default. For specific kept scenarios, set `scenarios='Baseline|My shock'` — names must match `mpak.keep_solutions` keys exactly. `mpak.smpl(start, end)` restricts the active sample before rendering.

Composition rules — easy to get wrong:

- **`|` (pipe) joins same-type objects.** `tab1 | tab2` → one combined table; `fig1 | fig2` → one combined figure. Will not mix table with plot.
- **`+` builds a report.** Any combination of tables, plots, text, other reports: `(tab | tab_gov) + fig + textblock`. Reports lift the same-type restriction.
- Non-transposed tables join with `|`. **Transposed tables (`transpose=True`) cannot** — put them in a report with `+` instead.
- Joined tables share the time window. Reports don't have this restriction.
- Use `heading=` on `.table()` to label sections inside a `|`-joined table.

Saving a report shape (not its data) for reuse:

```python
small = ((tab | tab_gov)
         .set_options(smpl=(2025,2029), title='{cty_name} GDP components')
         + fig).set_name('small_report')

mpak.add_report(small)                   # stored in mpak.reports['small_report']
new = mpak.get_report('small_report')    # rebuilds against current .basedf/.lastdf
```

`{cty_name}` is a string-substitution placeholder ModelFlow expands from model metadata. `mpak.reports` is part of model state and is preserved across `modeldump` / `modelload`, so stored reports survive disk round-trips.

Standard scenario-then-report workflow:

```python
mpak, baseline = model.modelload('../models/pak.pcim', run=True, keep='Baseline')

ct30 = baseline.upd('<2025 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30')
_    = mpak(ct30, 2020, 2100, keep='$30 carbon tax')

ct60 = baseline.upd('<2025 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 60')
_    = mpak(ct60, 2020, 2100, keep='$60 carbon tax')

mpak.smpl(2025, 2040)
tab = mpak.table(pat='*NYGDPMKTPKN *CCEMISCO2TKN', datatype='difpctlevel',
                 title='GDP and emissions vs baseline')
fig = mpak.plot(pat='*NYGDPMKTPKN *CCEMISCO2TKN', scenarios='*',
                samefig=True, datatype='growth')
report = (tab + fig).set_name('carbon_tax_report')
report.pdf()
```

For solver options on the simulation call (`alfa`, `ljit`, `silent`, etc.), see `simulation-methods.md`.
