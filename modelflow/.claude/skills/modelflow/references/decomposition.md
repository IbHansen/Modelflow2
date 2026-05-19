# Decomposition and impact attribution

When a scenario produces a change in some variable, the natural next question is "why?" — which of the RHS drivers actually moved the result, and by how much. ModelFlow has two distinct attribution machineries for this, plus a third visual one (the causal graph) that overlays attribution on the dependency structure.

Relevant source: `modeldekom.py` (the `totdif` class and the model-level attribution), `modelclass.py` (the `.dekomp()` engine on the model, plus the `get_att_*` and `dekomp_plot*` wrappers), `modelvis.py` (the `[]` selector's `.dekomp()` and `.tracepre()` shortcuts), and `modeljupyter.py` (the interactive widgets). PDF chapters 15 and 16 walk through every example here on the Pakistan model.

---

## Single-equation vs. model-level attribution

These solve different problems and rest on different machinery. Pick the right one before reaching for a method:

| Question | Use |
|---|---|
| "Why did my consumption equation produce this value?" — RHS variables of one equation explaining one LHS | **Single-equation** (`.dekomp`) |
| "Why did GDP move in this scenario?" — exogenous shocks propagating through the full model to one or more endogenous variables | **Model-level** (`.totdif`) |
| "What did I shock, and what's the dependency path it took?" | **Tracing** (`.tracepre`, `.tracedep`, `.draw`) |

Practical rule of thumb: if the user can write down the equation they want to decompose, use single-equation. If the user is asking about *propagation through the model*, use model-level. The two answers usually look similar near the source of a shock but diverge as you move downstream — for variables many steps away from any direct shock, single-equation attribution will assign almost everything to a small set of upstream endogenous variables, while model-level attribution will trace it all the way back to the original exogenous drivers.

---

## Single-equation attribution: `.dekomp()`

`.dekomp(varname)` calculates the contributions of each RHS variable in a single equation to the change in its LHS between `.basedf` and `.lastdf`. It works by partial differentiation — re-evaluating the equation while substituting one RHS variable at a time from the baseline values, and recording the delta.

```python
result = mpak.dekomp('PAKNECONPRVTKN', start=2021, end=2030, lprint=False)
```

Parameters:

| Param | Default | Purpose |
|---|---|---|
| `varnavn` | required | The endogenous variable whose equation is decomposed. |
| `start`, `end` | current sample | Period for the attribution. |
| `basedf`, `altdf` | `.basedf`, `.lastdf` | Override the baseline / alternative DataFrames. |
| `lprint` | True | Print the result tables. Set to False when you want only the namedtuple back. |
| `time_att` | False | Year-over-year mode instead of base-vs-alternative. See "time attribution" below. |

`.dekomp()` returns a namedtuple with five DataFrames:

| Field | Contents |
|---|---|
| `diff_level` | Base level, alternative level, level difference, percent difference. |
| `att_level` | Contribution of each RHS variable to the level difference. |
| `att_pct` | Same contributions expressed as percent of the total difference. |
| `diff_growth` | Base growth rate, alternative growth rate, difference. |
| `att_growth` | Contribution of each RHS variable to the growth-rate difference. |

A row label like `PAKXMKT(-1)` means "the lagged-by-one value of PAKXMKT"; ModelFlow keeps each lag of each RHS variable as a separate attribution row by default.

Unpack the namedtuple field by field when you want the underlying tables:

```python
dekomp_result = mpak.dekomp('PAKNECONPRVTKN', lprint=False)
for f, df in zip(dekomp_result._fields, dekomp_result):
    display(f, df)

# Or by name:
dekomp_result.att_pct.style.format('{:.2f}')
```

`mpak[VAR].dekomp()` is the shortcut form via the `[]` selector — same machinery, slightly more compact when chaining.

### Time attribution: `time_att=True`

By default, `.dekomp()` decomposes the *difference between two scenarios* at each point in time. Setting `time_att=True` changes the question to "what year-over-year movements in the RHS variables explain the year-over-year change in the LHS?", using only `.lastdf` (no baseline involved):

```python
mpak.PAKCCEMISCO2TKN.get_att(time_att=True, type='level', bare=False)
```

This is the right mode for forecast narratives like "consumption grew 2.1% in 2027; here's how much of that came from income, from rates, from prices..." rather than scenario comparisons.

### Extracting the tables: `get_att_pct` / `get_att_level`

For just the attribution table (without the rest of the namedtuple), use the wrappers. Both accept the same options:

```python
mpak.get_att_pct('PAKNECONPRVTKN', threshold=10, lag=False, start=2025, end=2035)
mpak.get_att_level('PAKNECONPRVTKN', filter=True, lag=True, time_att=False)
```

| Option | Default | Effect |
|---|---|---|
| `threshold` | 0.0 | Suppress rows whose maximum absolute contribution across the displayed period is below this. Dropped rows are aggregated into a single `small` row so totals still add up. |
| `lag` | True | If True, keep `VAR(-1)`, `VAR(-2)` as separate rows. If False, sum contributions across lags. |
| `filter` | False | Drop rows that are exactly zero across the entire period. |
| `start`, `end` | current sample | Time window. |
| `time_att` | False | See above. |

The `[]` selector's `.get_att()` accepts the same arguments plus `type='pct'/'level'/'growth'` and a `bare` toggle that prepends the actual LHS values to the output. The PDF chapter 16.6 walks through these option combinations in detail.

### Plotting: `dekomp_plot` and `dekomp_plot_per`

For a single endogenous variable over the current sample as a stacked bar chart:

```python
mpak.dekomp_plot('PAKNECONOTHRXN', pct=True, rename=True, threshold=0.01, lag=False)
```

For a waterfall chart at one specific period:

```python
mpak.dekomp_plot_per('PAKNYGDPFCSTXN', per=2027, threshold=5, sort=True)
```

| Option | Default | Effect |
|---|---|---|
| `pct` | True | Plot percent contributions; if False, plot level contributions. |
| `lag` | True | Separate bars per lag, or aggregate across lags. |
| `threshold` | 0.0 | Drop rows below this magnitude. |
| `rename` | True | Use `var_description` strings instead of mnemonics as bar labels. |
| `sort` | False (for `_per`) | Sort the waterfall bars by magnitude. |
| `top` | 0.9 | Title vertical placement. |
| `time_att` | False | Year-over-year decomposition instead of base-vs-alt. |

Both return a matplotlib `Figure` so they can be embedded in reports or saved with `fig.savefig(...)`.

---

## Model-level attribution: `.totdif()`

Single-equation attribution stops at the equation boundary. To trace a change all the way back to the *exogenous* drivers that caused it — across however many equations the impulse propagated through — use `.totdif()`.

The machinery: for every exogenous variable that has changed between `.basedf` and `.lastdf`, re-solve the model with that variable reset to baseline while everything else stays shocked. The difference between the full-shock solution and the partially-reset solution tells you how much that exogenous variable contributed.

```python
totdekomp = mpak.totdif()    # Total dekomp took : 3.728 Seconds (on Pakistan)
totdekomp.res.keys()         # dict_keys(['level', 'growth'])
```

`mpak.totdif()` stores the result on the model as `mpak.totdekomp` (so a second call hits the cache via `get_att_gui` etc.) and also returns it for chaining. Re-running is needed any time the scenario changes — the cached `totdekomp` only reflects the state of `.basedf` and `.lastdf` at the moment it was built.

Parameters of the `totdif` constructor (also accepted by `mpak.totdif(...)`):

| Param | Default | Purpose |
|---|---|---|
| `summaryvar` | `'*'` | Restrict the set of *endogenous* variables tracked. For large models with thousands of variables, narrow this to the ones you care about — saves a lot of time. |
| `desdic` | `mpak.var_description` (when called via `mpak.totdif()`) | Label dictionary used in the plots. |
| `experiments` | `None` (one per changed exogenous variable) | Group exogenous variables into named blocks; see below. |

### Grouping exogenous shocks: `experiments`

The default behavior is "one experiment per changed exogenous variable", which gives the maximum resolution but can be unreadable when the shock touches dozens of variables. The `experiments` argument accepts a dict mapping group names → lists of variables, so you can decompose by theme:

```python
groups = {
    'Carbon tax — coal': ['PAKGGREVCO2CER'],
    'Carbon tax — gas':  ['PAKGGREVCO2GER'],
    'Carbon tax — oil':  ['PAKGGREVCO2OER'],
    'Fiscal offsets':    ['PAKGGEXPGNFSCN', 'PAKGGEXPTRNFCN'],
}
totdekomp = mpak.totdif(experiments=groups)
```

In a multi-country setting it's natural to group by country (one experiment per country, all its variables inside); in a single-country model with many policy levers it's more useful to group by policy type.

### Displaying results: the `explain_*` family

The `totdif` instance exposes four display methods, each tuned for a different question, plus the umbrella `totexplain`:

| Method | What it shows |
|---|---|
| `explain_last(pat, ...)` | Decomposition at the **final period** of the sample. Useful for "where did we end up?" |
| `explain_per(pat, per=2027, ...)` | Decomposition at one **specific period**. Useful for highlighting a key year. |
| `explain_sum(pat, ...)` | **Cumulative** decomposition summed over the whole sample. Useful for "total impact of the policy". |
| `explain_all(pat, kind='area', stacked=True, ...)` | Time series of contributions across the whole sample. Useful when the *dynamics* matter (effects build, peak, and fade). |

Common options across the family:

| Option | Default | Effect |
|---|---|---|
| `pat` | `''` | Wildcard pattern selecting which endogenous variables to display. Multiple patterns can be passed as a space-separated string; each gets its own subplot. |
| `use` | `'level'` | Decompose levels (`'level'`) or growth rates (`'growth'`). |
| `threshold` | 0.0 | Drop contributions below this magnitude; aggregated into `small`. |
| `title` | `''` | Plot title; default is auto-generated. |
| `stacked` | True | Stacked bars/areas (True) or grouped (False). |
| `kind` | `'bar'` | matplotlib chart kind — `'bar'`, `'area'`, `'line'`. `'area'` reads best for `explain_all` over long samples. |
| `top` | 0.9 | Title vertical placement. |
| `ysize` | 5 or 10 | Plot height per panel. |

Examples:

```python
showvar = 'PAKNYGDPMKTPKN PAKCCEMISCO2CKN PAKCCEMISCO2OKN PAKCCEMISCO2GKN PAKGGREVTOTLCN'

totdekomp.explain_all(showvar, kind='area', stacked=True,
                      title='Contributions of different carbon taxes to Real GDP, growth')

totdekomp.explain_per(showvar, per=2028, ysize=8,
                      title='Decomposition, level=2028')

totdekomp.explain_last('PAKNYGDPMKTPKN', ysize=8,
                       title='Decomposition last period, level')

totdekomp.explain_sum('PAKNYGDPMKTPKN', ysize=8,
                      title='Decomposition, sum over all periods, level')
```

`totexplain(pat, vtype='all'|'per'|'last'|'sum', ...)` is the unified wrapper — `vtype` chooses which `explain_*` to call internally. Convenient when the choice of view is itself a parameter:

```python
mpak.totexplain('PAKNYGDPMKTPKN', vtype='per', per=2028, use='growth')
```

Note `mpak.totexplain(...)` (on the model) lazily creates `mpak.totdekomp` if it doesn't exist yet, so for a one-shot decomposition you can skip the explicit `totdif()` call. For repeated viewing of the same scenario, build `totdekomp = mpak.totdif()` once and call its `explain_*` methods directly to avoid recomputing.

### What's in `.res`

For custom output beyond the standard plots, `totdekomp.res` is a dict with two keys, `'level'` and `'growth'`, each holding a multi-indexed DataFrame:

- Index: the summary endogenous variables.
- Columns: a MultiIndex of (experiment name, period).

The helper functions in `modeldekom.py` — `GetAllImpact`, `GetSumImpact`, `GetLastImpact`, `GetOneImpact`, `AggImpact` — operate on this structure and are what the `explain_*` methods use internally. Drop down to them when the canned plots don't fit.

### `exodif()`: which exogenous variables actually changed

`mpak.exodif()` returns just the changed-exogenous columns of `.lastdf - .basedf`. This is what `totdif` calls under the hood to figure out which experiments to run. Use it directly to sanity-check what's actually being decomposed:

```python
mpak.exodif().abs().sum().sort_values(ascending=False).head(10)
```

If the user expected ten variables to be shocked and `exodif()` shows fifteen, that's a sign the input DataFrame has unintended differences (often from a stale `.upd` chain).

---

## Combining tracing with attribution

The dependency graph methods from `modelvis.py` — `.tracepre()`, `.tracedep()`, `.draw()` — automatically pick up attribution information when a simulation has been run. The thickness of each edge in the graph is proportional to that link's empirical importance in explaining the LHS change.

```python
# Causal predecessors of GDP, after a shock has been simulated:
mpak.PAKNYGDPMKTPKN.tracepre(filter=20, up=3, png=True)
```

Options worth knowing:

| Option | Default | Effect |
|---|---|---|
| `up` (tracepre) / `down` (tracedep) | 1 | How many levels of the causal chain to display. |
| `filter` | False | Prune edges whose attribution share is below this percent. `filter=20` keeps only the major channels. Too aggressive a filter can prune everything ("The graph is empty / Perhaps filter prunes to much"). |
| `showdata` / `sd` | False | Attach a values table to each node; `sd='*lcn'` filters which variables to tabulate. |
| `attshow` / `ats` | False | Attach an attribution table to each node showing what each upstream variable contributed. |
| `growthshow` / `gs` | False | Show growth-rate tables alongside levels. |
| `HR` | True | Horizontal vs. vertical layout. |
| `fokus2` | `''` | Space-separated list of variables to highlight with attached tables. |
| `png`, `svg`, `pdf`, `eps` | False | Output format. `svg` is best for zooming; `pdf` for print. |
| `saveas` | `''` | File name (without extension) to save to. |
| `browser` | False | Open in a browser tab — useful for very large graphs. |

`mpak[VAR1 VAR2].draw(up=1, down=1, filter=20)` works through the `[]` selector and gives the most concise syntax for "show me the immediate neighborhood of this variable, but only the strong links".

The combined "graph with attribution tables attached to each node" output (`showdata=True, attshow=True`) is the densest single view of how a shock propagated — it shows both the structure and the magnitudes in one picture. PDF section 16.7 has the worked example.

---

## Interactive widgets

For exploration, two GUI wrappers are useful in Jupyter:

```python
mpak.get_dekom_gui('PAKNECONPRVTKN')   # single-equation attribution explorer
mpak.get_att_gui(var='PAKGGREVTOTLCN', spat='*', ysize=7)   # model-level explorer
mpak.modeldash('PAKNYGDPMKTPKN', jupyter=True)   # tracepre/tracedep dashboard
```

These spin up ipywidgets-based controls that let the user flip variable, view type (`level`/`growth`), threshold, and lag-aggregation on the fly. Useful when handing the model off to a non-developer analyst. The PDF mentions these are not "live" in the printed manual — they only work in a running notebook.

---

## Pitfalls

- **`.totdif()` is expensive and must be re-run after every scenario change.** It re-solves the model once per experiment. For a model with 20 shocked variables, that's 20 solves. Use `ljit=True` on the underlying solve calls and limit `summaryvar` to keep this manageable.
- **`.dekomp()` (single equation) does not capture indirect effects.** If RHS variable `X` moved because of an upstream shock, `.dekomp()` attributes the LHS change to `X` directly — it doesn't tell you what moved `X`. To follow the chain back to the original exogenous driver, use `.totdif()` or chained `.dekomp()` calls up the dependency tree.
- **The `threshold` parameter behavior differs between contexts.** In `.dekomp_plot` and `.get_att_*`, `threshold` operates on absolute contribution magnitude (in the units of the variable, or in percent if `pct=True`). In `tracepre(filter=...)`, the filter is always a percent of the total. Reading "20" the same way in both will give wildly different results.
- **`time_att=True` and base-vs-alt mode are different decompositions.** They answer different questions. A user asking "why did consumption grow this much last year?" wants `time_att=True` on `.lastdf` alone. A user asking "how does my carbon-tax scenario compare to baseline?" wants the default. Be explicit about which is wanted.
- **`totdif` only attributes to changed *exogenous* variables.** If everything in the model moved but no exogenous variable was shocked (e.g. the scenario was constructed entirely via `.fix()` on endogenous variables and modifying `_X` values), `totdif` will report "No variables to attribute to" because `exodif()` returns empty. The fix-set variables are technically endogenous; their `_X` companions are exogenous. Make sure the shocked `_X` values actually differ between `.basedf` and `.lastdf`.
- **Variable descriptions matter for the plot labels.** The `explain_*` methods use `desdic` (defaulting to `mpak.var_description`) for axis labels. If your model has cryptic mnemonics and no descriptions, the plots will be hard to read — populate `mpak.var_description` before decomposing.
