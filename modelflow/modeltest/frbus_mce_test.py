# -*- coding: utf-8 -*-
"""
Stacked Newton test on FRB/US with model-consistent expectations (FRB/US MCE).

The model is ``modeltest/data/frbusmcetest.pcim`` (285 endogenous variables,
leads, solved 2020q1-2120q4).  The test is the experiment loop of
``frbus-run-mce_new.py``: shocks of 0, 1, 2 and 3 to the federal funds rate
residual ``rffintay_aerr`` in 2020q1, the Jacobian built on the first shock and
reused for the following (``newton_reset=0``).

For each run in ``RUNS``, without and with jit:

* cold    -- code regenerated and compiled, Jacobian built (shock 0);
* rebuild -- the same with the compiled code reused: Jacobian built (shock 0);
* shocks 1-3 with the Jacobian reused -- the time that matters in a loop.

Then the stacked Jacobian (whole model and split core): size, factorization
with SuperLU (NATURAL and COLAMD) and UMFPACK, conditioning; UMFPACK against
SuperLU and ``split=1`` against ``split=0`` on the model solution; the split
classification; and a ``timeit`` breakdown of one shock.

``newtonstack_fbmin_ng`` is left out: on FRB/US MCE the stacked graph is one
giant strongly connected component and it runs for minutes.

Run the #%% cells in order or ``%runfile`` the file (a few minutes).  Jit code
is compiled in memory (``stringjit=True``), so no ``modelsource/`` files are
written.  If ``modelclass`` is already imported from somewhere else in this
session, restart the kernel first.
"""
#%% setup
import gc
import re
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

try:
    MODELTEST = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")
DATA = MODELTEST / 'data'

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)

#%% settings
SHOCKS = [0, 1, 2, 3]            # rffintay_aerr + shock in 2020q1
LJIT = [False, True]             # run every solver without and with jit
# collect garbage before each timed call and keep the collector off while the
# clock runs
GC_QUIET = True

all_opts = dict(nonlin=15, max_iterations=300, nljit=0, chunk=30, stringjit=True)
# label: (solver, options)
RUNS = {
    'newtonstack':          ('newtonstack',    {}),
    'newtonstack_ng':       ('newtonstack_ng', {'diff_ng': 1, 'split': 0}),
    'newtonstack_ng split': ('newtonstack_ng', {'diff_ng': 1, 'split': 1}),
}
REFERENCE = ('newtonstack_ng', True)   # (run, ljit) the others are compared with

#%% load FRB/US MCE and make the shocks (not timed)
mfrbusmce, basedf = model.modelload(str(DATA / 'frbusmcetest.pcim'), run=0)
ALTS = {s: basedf.upd(f'<2020q1 2020q1> rffintay_aerr + {s} ') for s in SHOCKS}
endo = sorted(mfrbusmce.endogene)


def solve(label, ljit, shock, newton_reset, **extra):
    """Solve shock *shock* with the run *label*; returns (seconds, result,
    model evaluations).  The model statistics are captured to read the number
    of model evaluations."""
    solver, opts = RUNS[label]
    if GC_QUIET:
        gc.collect()
        gc.disable()
    out = StringIO()
    try:
        with redirect_stdout(out):
            t0 = time.perf_counter()
            res = mfrbusmce(ALTS[shock], solver=solver, silent=1, stats=1,
                            reset_options=True, ljit=ljit, newton_reset=newton_reset,
                            **{**all_opts, **opts, **extra})
            seconds = time.perf_counter() - t0
    finally:
        if GC_QUIET:
            gc.enable()
    found = re.search(r'Total model evaluations\s*:\s*([\d,]+)', out.getvalue())
    return seconds, res, int(found.group(1).replace(',', '')) if found else None


#%% run: every solver without and with jit
rows, results = [], {}
for label in RUNS:
    for ljit in LJIT:
        t_cold, _, _ = solve(label, ljit, 0, newton_reset=1, transpile_reset=True)
        t_rebuild, _, ev0 = solve(label, ljit, 0, newton_reset=1)
        shocks = [solve(label, ljit, s, newton_reset=0) for s in SHOCKS[1:]]
        results[(label, ljit)] = shocks[-1][1]
        rows.append(dict(solver=label, ljit=ljit, cold=t_cold, rebuild=t_rebuild,
                         **{f'shock {s}': t for s, (t, _, _) in zip(SHOCKS[1:], shocks)},
                         evaluations=' '.join(str(e) for e in [ev0] + [e for _, _, e in shocks])))
        print(f'{label:22s} {"jit" if ljit else "   "} cold {t_cold:7.2f} s  rebuild {t_rebuild:6.2f} s  '
              f'shocks {" ".join(f"{t:.3f}" for t, _, _ in shocks)} s')

per = mfrbusmce.current_per
ref = results[REFERENCE].loc[per, endo]
for row in rows:
    res = results[(row['solver'], row['ljit'])].loc[per, endo]
    row['max rel diff'] = float(((res - ref).abs() / (ref.abs() + 1e-8)).max().max())
table = pd.DataFrame(rows).set_index(['solver', 'ljit'])
with pd.option_context('display.float_format', '{:,.4g}'.format, 'display.width', 160):
    print(table)

#%% the split classification
s = mfrbusmce.ng_stack_split_struct
print(f"split: {len(s['core'])} core / {len(s['pro'])} prolog / {len(s['epi'])} epilog "
      f"of {len(mfrbusmce.endogene)} variables")
print('prolog:', ' '.join(sorted(s['pro'])))
print('epilog:', ' '.join(sorted(s['epi'])))

#%% the stacked Jacobian: size, factorization, conditioning (whole model and split core)
from cvxopt import matrix, spmatrix, umfpack


def factorization_report(jacobi):
    """Size of *jacobi*; factorize with SuperLU (NATURAL, COLAMD) and UMFPACK,
    time one solve with b = ones; conditioning and where the solutions differ."""
    jacobi = jacobi.tocsc()
    n = jacobi.shape[0]
    print(f'Jacobian {n:,} x {n:,} = {n * n:,} elements, {jacobi.nnz:,} nonzero '
          f'({jacobi.nnz / n / n * 100:.6f}%, {jacobi.nnz / n:.1f} per row)')
    b = np.ones(n)
    sol = {}
    for permc in ['NATURAL', 'COLAMD']:        # never MMD_AT_PLUS_A here, it hangs
        t0 = time.perf_counter()
        lu = spla.splu(jacobi, permc_spec=permc)
        t1 = time.perf_counter()
        x = lu.solve(b)
        t2 = time.perf_counter()
        fill = lu.L.nnz + lu.U.nnz
        sol[permc] = x
        if permc == 'NATURAL':
            lu_nat = lu
        print(f'{permc:8} factorize {t1-t0:6.2f} s   one solve {t2-t1:6.3f} s   '
              f'L+U nonzeros {fill:,} ({fill/jacobi.nnz:.1f} times the Jacobian)   '
              f'max |Jx-b| {np.max(np.abs(jacobi @ x - b)):.2e}')
    coo = jacobi.tocoo()
    A = spmatrix(matrix(coo.data), coo.row.tolist(), coo.col.tolist(), coo.shape)
    t0 = time.perf_counter()
    symbolic = umfpack.symbolic(A)
    t1 = time.perf_counter()
    numeric = umfpack.numeric(A, symbolic)
    t2 = time.perf_counter()
    xu = matrix(b)
    umfpack.solve(A, numeric, xu)
    t3 = time.perf_counter()
    xu = np.asarray(xu).ravel()
    print(f'UMFPACK  factorize {t2-t0:6.2f} s   one solve {t3-t2:6.3f} s   '
          f'(symbolic {t1-t0:.2f} s + numeric {t2-t1:.2f} s)   '
          f'max |Jx-b| {np.max(np.abs(jacobi @ xu - b)):.2e}')

    x_nat = sol['NATURAL']
    diff = np.abs(xu - x_nat)
    endovar = mfrbusmce.ng_newton_diff_stack.endovar
    nvar = len(endovar)
    i = int(np.argmax(diff))
    print(f'max |x| {np.max(np.abs(x_nat)):.2e}; UMFPACK - SuperLU largest difference '
          f'{diff[i]:.2e} ({diff[i] / np.max(np.abs(x_nat)):.2e} relative) '
          f'in {endovar[i % nvar]} {per[i // nvar]}')
    Jinv = spla.LinearOperator(jacobi.shape, matvec=lu_nat.solve,
                               rmatvec=lambda v: lu_nat.solve(v, trans='T'))
    print(f'Estimated condition number (1-norm) {spla.onenormest(jacobi) * spla.onenormest(Jinv):.2e}')
    x_ref = x_nat + lu_nat.solve(b - jacobi @ x_nat)
    print(f'One refinement step moves the SuperLU solution by at most {np.max(np.abs(x_ref - x_nat)):.2e}')


for split in (0, 1):
    print(f'\n===== split={split} =====')
    solve('newtonstack_ng split' if split else 'newtonstack_ng', True, 1, newton_reset=1)
    factorization_report(mfrbusmce.ng_newton_diff_stack.stacked)

#%% model solutions: UMFPACK against SuperLU, split=1 against split=0, res-check
_, r_split1, _ = solve('newtonstack_ng split', True, 1, newton_reset=1, factorizer=None)
_, r_umf, _ = solve('newtonstack_ng split', True, 1, newton_reset=1, factorizer='umfpack')
_, r_split0, _ = solve('newtonstack_ng', True, 1, newton_reset=1, factorizer=None)
mfrbusmce.oldkwargs['factorizer'] = None
show = lambda d: d.loc[per, endo].abs().max().sort_values().tail(5)
print('Largest absolute differences, UMFPACK - SuperLU (split=1):')
print(show(r_umf - r_split1))
print('Largest absolute differences, split=1 - split=0:')
print(show(r_split1 - r_split0))
print('Largest equation violations in the split=1 solution (res-check):')
print(show(mfrbusmce.res(r_split1, per[0], per[-1], silent=True) - r_split1))

#%% timeit breakdown of one shock (split=1, jit, Jacobian reused)
solve('newtonstack_ng split', True, 0, newton_reset=1)
solver, opts = RUNS['newtonstack_ng split']
mfrbusmce(ALTS[1], solver=solver, silent=1, reset_options=True, ljit=True,
          newton_reset=0, timeit=1, **{**all_opts, **opts})
