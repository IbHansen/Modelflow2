# Simulation methods

This file covers actually running a model: the solve call, solver options, scenarios, exogenization via `.fix()`, the add-factor mechanism, sample-period management, and targeting/inversion. PDF chapters 11, 12, 13, and 18.9–18.14 are the deep reference.

---

## The solve call

A loaded model is callable. Passing it a DataFrame solves the model on that DataFrame and returns the result:

```python
result = mpak(input_df, start=2020, end=2100, alfa=0.7, silent=1, keep='My scenario')
```

What happens:

- The model object inspects the DataFrame, sets up the solver (Gauss-Seidel by default, Newton variants when needed), and runs over the period `[start, end]`.
- The full result is returned as a DataFrame **and** stored on the model as `.lastdf`.
- If `keep='name'` is set, a copy is also stored in `mpak.keep_solutions['name']`.
- The first time the model is solved in a session, `.basedf` is also set (this is the baseline).

Once the model has been called once, **most options persist** — they're stored in `mpak.oldkwargs` and reused on subsequent calls unless overridden. This is convenient but surprising: if a user sets `alfa=0.3` once, every later solve uses 0.3 until they pass a different value. `reset_options=True` wipes the persistent state.

## Solver options worth knowing

| Option | Default | What it does |
|---|---|---|
| `start`, `end` | full sample | Solve window. Strings, integers, or pd.Period all work depending on index dtype. |
| `alfa` | 0.2 | Step-size damping for behavioral equations marked `<DAMP>` / `<Z>`. Range (0, 1]. Larger = faster, less stable. 0.7 is a good starting point for MFMod-style models. |
| `max_iterations` | model-dependent | Cap on Gauss-Seidel iterations per period. Raise if the model fails to converge. |
| `absconv` | 0.01 | Absolute convergence tolerance. |
| `relconv` | 0.00001 | Relative convergence tolerance. |
| `silent` | True | Suppress per-period solver output. Set `silent=False` (or `silent=0`) when debugging non-convergence. |
| `ljit` | False | Compile the solver with Numba JIT. First call takes 30s–2min; subsequent calls are much faster. Worth it for targeting or any workflow with many solves. |
| `solver` | auto-chosen | Override the solver. Usually 'sim' (Gauss-Seidel topological), 'newton', 'newtonstack' (for forward-looking models with leads), or 'newton_un_normalized'. |
| `keep` | '' | Store the result under this name in `keep_solutions`. |
| `keep_variables` | all | Wildcard pattern restricting which columns are stored in the kept DataFrame. Useful when models have thousands of variables. |
| `dumpvar`, `ldumpvar` | [], False | Dump per-iteration values for the listed variables — essential when chasing convergence problems. Combine with `.dump_iter_plot()` to visualize. |
| `setbase` | False | Overwrite `.basedf` with the result. Rarely wanted; users usually want the loaded baseline. |
| `setlast` | True | Whether to store the result on `.lastdf`. |
| `reset_options` | False | Clear `oldkwargs` so persistent options don't carry over. |

When non-convergence happens, the standard diagnostic ladder is:

1. Set `silent=False` to see which period is failing.
2. Lower `alfa` (try 0.3, then 0.1).
3. Raise `max_iterations`.
4. Add `dumpvar=['VAR1','VAR2']`, `ldumpvar=True` and inspect with `mpak.dump_iter_plot('*')`.
5. If the problem is forward-looking and Gauss-Seidel can't handle it, force `solver='newtonstack'`.

## Sample period control: `.smpl()` and `.set_smpl()`

The model's *active sample* governs what tables, plots, and the `[]` selector show. Two methods manage it:

```python
mpak.smpl(2025, 2040)            # set the active sample (persistent)
mpak.smpl()                      # reset to the full range

with mpak.set_smpl(2025, 2040):  # set temporarily inside a with-block
    mpak['*NYGDP*'].plot()
```

`.set_smpl(...)` as a context manager is the right pattern when temporarily restricting display — it restores the previous sample on exit even if an exception fires.

Note this is purely about *display*. Solve windows are controlled by the `start` / `end` arguments to the model call, not by `.smpl()`.

---

## Scenarios and `keep_solutions`

Multi-scenario work follows a consistent shape:

```python
mpak.keep_solutions = {}                           # clear any stale scenarios

mpak(baseline,    2020, 2100, keep='Baseline')
mpak(shock_df_A,  2020, 2100, keep='Carbon tax 30')
mpak(shock_df_B,  2020, 2100, keep='Carbon tax 60')

for k in mpak.keep_solutions:
    print(k)
```

Then the plot and report machinery can refer to scenarios by name:

```python
mpak.plot('*NYGDP*', scenarios='Baseline|Carbon tax 30|Carbon tax 60', samefig=True)
```

`scenarios=` always uses **exact** keys from `keep_solutions`, separated by `|`. The special value `'*'` includes everything; `'base_last'` uses `.basedf` and `.lastdf` instead.

`mpak.keepswitch(...)` exists for picking a subset and reordering — useful when many scenarios are present and the user wants a specific narrative order on the plot.

---

## Shocking exogenous variables (the simplest case)

For an exogenous variable, just modify the input DataFrame and re-solve:

```python
df = baseline.upd('<2025 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30')
mpak(df, 2020, 2100, keep='$30 carbon tax')
```

That's it. No flags, no add-factors, no exogenization machinery.

---

## Exogenizing an endogenous variable: `.fix()` and the `_D`/`_X` mechanism

Every behavioral equation in an MFMod model has the structural form:

```
VAR = (1 - VAR_D) * [estimated_RHS]   +   VAR_D * VAR_X
```

`VAR_D` is the **fix dummy**, defaulting to 0 (use the estimated equation). `VAR_X` is the **fix value**, used when `VAR_D = 1`. So to take an endogenous variable and pin it to a chosen path:

1. Set `VAR_D = 1` over the relevant window.
2. Set `VAR_X` to the desired values over the same window.
3. Solve.

`.fix()` is the convenience method that does both:

```python
cfixed = mpak.fix(baseline, 'PAKNECONPRVTKN')
# Now cfixed has PAKNECONPRVTKN_D = 1 over the sample and
# PAKNECONPRVTKN_X = the baseline values of PAKNECONPRVTKN.
# Solving cfixed would reproduce the baseline.

# Then shock VAR_X to introduce the scenario:
cfixed = cfixed.upd('<2025 2040> PAKNECONPRVTKN_X * 1.025')

# Solve:
mpak(cfixed, 2020, 2040, keep='Real consumption +2.5%')
```

A few related variants worth knowing:

- **Temporary exogenization** — set `VAR_D = 1` only for a window, then back to 0. The rest of the simulation uses the estimated equation as normal.
- **Add-factor shock** (preserves the estimated equation but offsets it): leave `VAR_D = 0` and update `VAR_A`, the add-factor variable, instead.
- **`keep_growth` interaction**: when shocking `VAR_X` for a window and wanting the post-window path to keep the baseline's growth rate, pass `keep_growth=True` to `.upd()`.

`mpak.fix_endo` lists the variables that have the fixable structure — typically all behavioral variables. Identities don't have `_D`/`_X` companions.

---

## Add-factor mechanics

Behavioral equations also carry an additive `VAR_A` term:

```
VAR = (1 - VAR_D) * [estimated_RHS + VAR_A]   +   VAR_D * VAR_X
```

On model load (with the data that was used in estimation), `VAR_A` is set so the equation fits the historical data exactly. Going forward:

- Leaving `VAR_A` at its load value keeps the equation behaving as estimated.
- Setting `VAR_A` for future periods adds a calibrated shock that survives the dynamics — useful for "judgmental adjustments" the forecaster wants to impose without rewriting the equation.
- `add-factor-based endogenous simulation` is the technical name for using `VAR_A` to nudge a behavioral variable while keeping the equation active.

Wildcard `*_A` finds every add-factor in the model; `mpak.var_with_frmlname('DAMP')` returns the set of dampable (i.e. behavioral) variables, which is a close proxy for the set of variables that have an `_A`.

---

## Targeting and inversion: `.invert()`

To solve a "policy problem" — find values of instruments such that targets hit specified paths — use `.invert()`. This is in `modelinvert.py`; PDF chapter 13 is the worked example.

```python
instruments = [
    ['PAKGGREVCO2CER', 'PAKGGREVCO2GER', 'PAKGGREVCO2OER'],   # combined instrument
    'PAKGGEXPGNFSCN_A',                                       # single instrument
]

result = mpak.invert(
    baseline,                  # starting DataFrame
    targets       = target_df, # DataFrame indexed by date, columns = target variables
    instruments   = instruments,
    defaultimpuls = 20,        # impulse used to numerically differentiate
    defaultconv   = 2000.0,    # convergence tolerance on target levels
    varimpulse    = True,      # instrument changes from one iter carry forward
    nonlin        = 5,         # recompute Jacobian every N iterations
    maxiter       = 75,
    progressbar   = True,
    silent        = True,
)
```

Key constraints:

- At least as many instruments as targets.
- An "instrument" can be a list of variables — they all move in proportion to their relative impulses.
- Convergence is sensitive to scaling. `defaultimpuls` should be the same order of magnitude as a meaningful change in the instrument.
- Targets are usually best expressed relative to baseline (e.g. "emissions 30% below baseline") and inversions usually benefit from `ljit=True` because the solver runs many times.

When inversion fails to converge, the tuning options that help most are `nonlin` (recompute the Jacobian more often), `defaultimpuls` (smaller for finer differentiation), and the absolute/relative tolerances. PDF section 13.9 walks through this.
