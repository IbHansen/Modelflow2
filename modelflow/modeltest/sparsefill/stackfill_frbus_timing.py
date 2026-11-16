"""
stackfill='plan' against stackfill='melt' on FRB/US MCE (modeltest/data/frbusmcetest.pcim,
solved 2020q1-2120q4): the same stacked jacobian and solution, and the timings of

* fill     -- get_diff_mat_tot alone (derivatives evaluated + matrix made), best of REPEAT
* rebuild  -- a solve of shock 0 with newton_reset=1 (jacobian built and factorized), best of REPEAT
* build    -- model.ng_stackbuildtime of that solve (fill + factorization)

for split=0 (whole model) and split=1 (core). No jit.
"""
#%% setup
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    MODELTEST = Path(__file__).resolve().parent.parent
except NameError:
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)
REPEAT = 3
opts = dict(nonlin=15, max_iterations=300, nljit=0, chunk=30, stringjit=True, diff_ng=1)

m, basedf = model.modelload(str(MODELTEST / 'data' / 'frbusmcetest.pcim'), run=0)
alt = basedf.upd('<2020q1 2020q1> rffintay_aerr + 1 ')
endo = sorted(m.endogene)


def timed(f):
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        res = f()
        return time.perf_counter() - t0, res
    finally:
        gc.enable()


def canon(a):
    a = a.tocsc().copy()
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return a


#%% run
rows = []
for split in (0, 1):
    res, mat = {}, {}
    for fill in ('plan', 'melt'):
        solve = lambda: m(alt, solver='newtonstack_ng', silent=1, reset_options=True, ljit=False,
                          newton_reset=1, split=split, stackfill=fill, **opts)
        solve()                                               # warm up: code generated, plan made
        rebuild, build = [], []
        for _ in range(REPEAT):
            t, res[fill] = timed(solve)
            rebuild.append(t)
            build.append(m.ng_stackbuildtime)
        diff = m.ng_newton_diff_stack
        assert diff.stackfill == fill
        mask = np.array([v.endswith('___RES') for v in diff.endovar], dtype=bool)
        fills = []
        for _ in range(REPEAT):
            # both fills at the same data (the plan solution), so the matrices can be compared
            t, mat[fill] = timed(lambda: diff.get_diff_mat_tot(df=res['plan'], is_residual_eq=mask))
            fills.append(t)
        rows.append(dict(split=split, stackfill=fill, n=mat[fill].shape[0], nnz=mat[fill].nnz,
                         fill=min(fills), build=min(build), rebuild=min(rebuild)))
        print(rows[-1])
    per = m.current_per
    a, b = canon(mat['plan']), canon(mat['melt'])
    same_pattern = np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    print(f'split={split}: same pattern {same_pattern}, '
          f'max abs difference in the jacobian {abs(a - b).max():.1e}, '
          f'in the solution {(res["plan"].loc[per, endo] - res["melt"].loc[per, endo]).abs().max().max():.1e}')

table = pd.DataFrame(rows).set_index(['split', 'stackfill'])
with pd.option_context('display.float_format', '{:,.3f}'.format, 'display.width', 160):
    print(table)
