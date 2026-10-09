# -*- coding: utf-8 -*-
"""
Speed test of the legacy and ng solvers on the PAK carbon-tax scenario
(2020-2100), for every equation form, without and with jit.

Each solver gets two timed runs:

* cold -- code regenerated and recompiled (``transpile_reset=True``) and
  Jacobians / Newton structures rebuilt (``newton_reset=True``);
* warm -- the same call again, everything reused.

The model is ``modeltest/data/pak.pcim`` (a copy of the mfdemo Pakistan model)
in three equation forms (``MODEL_FORMS``):

* ``'normalized'`` -- PAK as it is;
* ``'implicit'``   -- every equation rewritten to residual form,
  X___RES = (rhs) - (X)  (``modelmanipulation.un_normalize_model``);
* ``'hybrid'``     -- only the estimated (STOC) equations rewritten.

The solvers are grouped by the forms they handle (``SOLVERS_NORMALIZED_ONLY``,
``SOLVERS_IMPLICIT_ONLY``, ``SOLVERS_ANY_FORM``); each form runs its own group
plus the any-form solvers (hybrid: the any-form solvers only).  A group maps a
label to (solver name, options), so one solver can appear with different
options (``newtonstack_ng`` with and without ``split``).  Options: ``all_opts``
for every run, plus the run's own options.

Model loading is not timed.  Times are wall-clock time of the whole model call.
Every solution is compared with ``sim_ng`` on the normalized model (no jit).
The diagnostic cell prints iterations and Jacobian rebuilds of the stacked
Newton solvers.

Run the #%% cells in order (or ``%runfile`` the file; a full run takes several
minutes, mostly jit compiling).  ``modelclass`` must be importable from the
modelflow source folder (the setup cell prints where it was imported from).
Jit code is compiled in memory (``stringjit=True``), so no ``modelsource/``
files are written.  If ``modelclass`` is already imported from somewhere else
in this session, restart the kernel first.
"""
#%% setup
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    MODELTEST = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")
MODELFLOW = MODELTEST.parent
DATA = MODELTEST / 'data'               # pak.pcim copied from mfdemo
# if str(MODELFLOW) not in sys.path:
#     sys.path.insert(0, str(MODELFLOW))

import modelclass
from modelclass import model
import modelmanipulation as mp

print('modelclass imported from', modelclass.__file__)

#%% settings
START, END = 2020, 2100
MODEL_FORMS = ['normalized', 'implicit', 'hybrid']   # equation forms to test
LJIT = [False, True]             # run every solver without and with jit
# use_fbmin is a model option, read once when the model is built: True orders
# the core as daglist + fblist (minimal feedback set), False keeps the normal
# block order.  Passing it as a solve option has no effect.
USE_FBMIN = True
# collect garbage before each timed call and keep the collector off while the
# clock runs (False: measure as in normal use, collection pauses included)
GC_QUIET = True

# stringjit: compile the jit code from the generated text in memory instead of
# a modelsource/ file -- legacy and ng share some file names, and legacy
# makelos does not reload an already imported module, so a "cold" run could
# reuse code another solver compiled
all_opts = dict(chunk=30, stringjit=True)   # options for every solver
gauss_opts = dict()              # Gauss-Seidel solvers (sim, sim1d)
newton_opts = dict(nonlin=6)     # Newton solvers

# label: (solver name as given to model(solver=...), options), grouped by the
# equation forms a solver handles.  Legacy (Solver_Mixin) solvers are listed
# next to their ng counterparts.  Comment lines out to prune a run.

# Normalized models only
SOLVERS_NORMALIZED_ONLY = {
    'sim':                    ('sim',                  gauss_opts),
    'sim_ng':                 ('sim_ng',               gauss_opts),
    'sim1d':                  ('sim1d',                gauss_opts),
    'sim1d_ng':               ('sim1d_ng',             gauss_opts),
    # legacy newton rebuilds its Jacobian on every call (its default): it keeps
    # only the Jacobian of the last period it refreshed in, and reusing that
    # for the next solve fails in 2020 on PAK
    'newton':                 ('newton',               {**newton_opts, 'newton_reset': 1}),
    'newton_fbmin_ng':        ('newton_fbmin_ng',      newton_opts),
    'newtonstack':            ('newtonstack',          newton_opts),
    'newtonstack_fbmin_ng':   ('newtonstack_fbmin_ng', newton_opts),
}
# Fully implicit models only (every equation in residual form)
SOLVERS_IMPLICIT_ONLY = {
    # keeps its default newton_reset=1 (Jacobian rebuilt every call), like newton
    'newton_un_normalized':   ('newton_un_normalized', {**newton_opts, 'newton_reset': 1}),
    'newtonstack_un_normalized': ('newtonstack_un_normalized', newton_opts),
}
# Any form: normalized, implicit and hybrid (unified Newton solvers)
SOLVERS_ANY_FORM = {
    'newton_implicit':        ('newton_implicit',      newton_opts),
    'newton_ng':              ('newton_ng',            newton_opts),
    'newtonstack_implicit':   ('newtonstack_implicit', newton_opts),
    'newtonstack_ng':         ('newtonstack_ng',       newton_opts),
    # stacked pro/core/epi split: Newton and the stacked LU only on the core
    'newtonstack_ng split':   ('newtonstack_ng',       {**newton_opts, 'split': 1}),
}
FORM_SOLVERS = {
    'normalized': {**SOLVERS_NORMALIZED_ONLY, **SOLVERS_ANY_FORM},
    'implicit':   {**SOLVERS_IMPLICIT_ONLY, **SOLVERS_ANY_FORM},
    'hybrid':     SOLVERS_ANY_FORM,
}

# diagnostic cell: which forms, which stacked runs (labels), jit or not
DIAG_FORMS = ['normalized']
DIAG_SOLVERS = ['newtonstack', 'newtonstack_un_normalized', 'newtonstack_implicit',
                'newtonstack_ng', 'newtonstack_ng split', 'newtonstack_fbmin_ng']
DIAG_LJIT = True

#%% load PAK, make the scenario and the equation forms (not timed)
mpak, baseline = model.modelload(str(DATA / 'pak.pcim'), run=1, use_fbmin=USE_FBMIN,
                                 ljit=False)
alternative = baseline.upd("<2020 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30")
endo = sorted(mpak.endogene)       # the real (normalized) endogenous variables
# reference solution: sim_ng on the normalized model
ref = mpak(alternative, START, END, solver='sim_ng', silent=1,
           reset_options=True, ljit=False)[endo]


def to_residual_form(equations, select=lambda frmlname: True):
    """Rewrite the FRMLs whose name passes *select* to residual form,
    X___RES = (rhs) - (X); X stays the unknown.  The others are kept.
    The per-equation version of mp.un_normalize_model, for hybrid models."""
    out = []
    for comment, command, value in mp.find_statements(equations.upper()):
        if comment:
            out.append(comment)
        elif command == 'FRML':
            name = value[:value.find('>') + 1] if value.startswith('<') else ''
            frml = value[:-1]                       # without the closing $
            if select(name):
                frml = mp.un_normalize_expression(frml, implicit=True)
            out.append(f'FRML {frml} $')
        else:
            out.append(command + ' ' + value)
    return '\n'.join(out)


def make_model(form):
    """PAK in equation form *form*.  The ___RES columns missing in the
    scenario frame are added as 0 when solving."""
    if form == 'normalized':
        return mpak
    if form == 'implicit':
        text = mp.un_normalize_model(mpak.equations, implicit=True)
    else:   # hybrid: only the estimated (STOC) equations
        text = to_residual_form(mpak.equations,
                                lambda name: bool(mp.kw_frml_name(name, 'STOC')))
    m = model(text, modelname=f'{mpak.name} {form}', funks=mpak.funks,
              use_fbmin=USE_FBMIN)
    # the *_un_normalized solvers need a model with only residual-form equations
    if form == 'implicit':
        assert m.implicit, 'the implicit model still has normalized equations'
    return m


MODELS = {form: make_model(form) for form in MODEL_FORMS}
# the scenario frame of each form, with all of that model's columns (the
# implicit and hybrid models add X___RES columns).  Without them every call
# would count as new data (the frame never matches the extended column list
# recorded by the previous solve) and regenerate -- and with jit recompile --
# the model code, also in the warm runs.
ALTERNATIVES = {form: m.is_newdata(alternative)[1] for form, m in MODELS.items()}
for form, m in MODELS.items():
    n_res = sum(v.endswith('___RES') for v in m.endogene)
    print(f'{form:10s}: {n_res:3d} of {len(m.endogene)} equations in residual form '
          f'(normalized={m.normalized}, implicit={m.implicit}, hybrid={m.hybrid})')


def run_opts(form, label, cold):
    """(solver name, options) of the run *label* for *form*, with the
    cold/warm resets.

    Cold: code regenerated and Jacobian rebuilt.  Warm: both reused -- the
    resets are passed explicitly, because the legacy ``newton`` defaults to
    ``newton_reset=1``.  A run's own options win, so a run can keep rebuilding
    its Jacobian in the warm run (``'newton_reset': 1``).
    """
    solver, opts = FORM_SOLVERS[form][label]
    return solver, {**all_opts, 'transpile_reset': cold, 'newton_reset': cold,
                    **opts}


def timed_run(form, label, ljit, cold):
    """Solve the scenario with the run *label* on the *form* model; returns
    (seconds, result).

    With GC_QUIET the garbage collector runs before the clock starts and is off
    while it runs, so a collection pause (which grows with everything the
    session has created) is not charged to whichever solver happens to run."""
    solver, opts = run_opts(form, label, cold)
    if GC_QUIET:
        gc.collect()
        gc.disable()
    try:
        t0 = time.perf_counter()
        res = MODELS[form](ALTERNATIVES[form], START, END, solver=solver, silent=1,
                           reset_options=True, ljit=ljit, **opts)
        return time.perf_counter() - t0, res
    finally:
        if GC_QUIET:
            gc.enable()


#%% run: cold then warm for every form, jit setting and solver
rows, results = [], {}
for form in MODEL_FORMS:
    for ljit in LJIT:
        for run in FORM_SOLVERS[form]:              # run label
            label = f'{form:10s} {run}' + (' jit' if ljit else '')
            try:
                t_cold, _ = timed_run(form, run, ljit, cold=True)
                t_warm, res = timed_run(form, run, ljit, cold=False)
            except Exception as e:
                print(f'{label:40s} raised {type(e).__name__}: {e}')
                rows.append(dict(form=form, solver=run, ljit=ljit, cold=np.nan,
                                 warm=np.nan, error=type(e).__name__))
                continue
            results[(form, run, ljit)] = res
            rows.append(dict(form=form, solver=run, ljit=ljit, cold=t_cold,
                             warm=t_warm, error=''))
            print(f'{label:40s} cold {t_cold:9.3f} s   warm {t_warm:9.3f} s')

# table: times and difference from the reference solution (sim_ng, normalized model)
for row in rows:
    res = results.get((row['form'], row['solver'], row['ljit']))
    if res is None:
        row['max rel diff'] = np.nan
        continue
    absdiff = (res[endo] - ref).abs()
    row['max rel diff'] = float((absdiff / (ref.abs() + 1e-8)).max().max())

table = pd.DataFrame(rows).set_index(['form', 'solver', 'ljit'])
# summary: warm time per solver, one column per (form, jit)
warm = table['warm'].unstack(['form', 'ljit'])
warm = warm.reindex(columns=[(f, j) for f in MODEL_FORMS for j in LJIT])
warm = warm.reindex(list(dict.fromkeys(r['solver'] for r in rows)))   # run order
with pd.option_context('display.float_format', '{:,.4g}'.format,
                       'display.width', 160, 'display.max_rows', 200):
    print(table)
    print('\nWarm time (s) by equation form and jit:')
    print(warm)

#%% diagnostic: stacked Newton warm runs -- iterations and Jacobian refreshes
# A warm run that needs more than ``nonlin`` iterations rebuilds the stacked
# Jacobian for all periods (about 0.5 s on PAK).  Each solver first gets a cold
# run (newton_reset: fresh Jacobian, so no state from the previous solver), then
# a warm run with silent=0 (prints every iteration and each solver update) and
# stats=True (number of solver updates, setup and simulation time).
for form in DIAG_FORMS:
    for label in [l for l in DIAG_SOLVERS if l in FORM_SOLVERS[form]]:
        print(f'\n{"=" * 20} {form} {label}{" jit" if DIAG_LJIT else ""} {"=" * 20}')
        timed_run(form, label, DIAG_LJIT, cold=True)
        solver, opts = run_opts(form, label, cold=False)
        if GC_QUIET:
            gc.collect()
            gc.disable()
        t0 = time.perf_counter()
        try:
            MODELS[form](ALTERNATIVES[form], START, END, solver=solver, silent=0,
                         stats=True, reset_options=True, ljit=DIAG_LJIT, **opts)
        except Exception as e:
            print(f'{label} raised {type(e).__name__}: {e}')
        finally:
            if GC_QUIET:
                gc.enable()
        print(f'{label} warm wall time {time.perf_counter() - t0:.3f} s')
