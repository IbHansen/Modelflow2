"""
The number of MKL threads against the time of the PARDISO factorizations ('pardiso' and 'normal') of the stacked
jacobian of FRB/US MCE (modeltest/data/frbusmcetest.pcim, solved 2020q1-2120q4). For each thread count:

* factor   -- modelfactorize.make_solver on the stacked jacobian alone, best of REPEAT
* solve    -- one solve with those factors (incl. refinement), best of REPEAT
* rebuild  -- a solve of the shock with newton_reset=1 (jacobian built and factorized), best of REPEAT

Threads are set at runtime with MKL_Set_Num_Threads on the MKL that pypardiso loads, so the script needs pypardiso.
The default (MKL's own choice, normally the physical cores) is the first row. No jit.
"""
#%% setup
import ctypes
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypardiso

try:
    MODELTEST = Path(__file__).resolve().parent.parent
except NameError:
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")

import modelclass
import modelfactorize
from modelclass import model

print('modelclass imported from', modelclass.__file__)
REPEAT = 3
FACTORIZERS = ('pardiso', 'normal')
opts = dict(nonlin=15, max_iterations=300, nljit=0, chunk=30, stringjit=True, diff_ng=1)

MKL = pypardiso.PyPardisoSolver().libmkl
MKL.MKL_Get_Max_Threads.restype = ctypes.c_int


def mkl_threads(n=None):
    """Set MKL's thread count, or leave it as is when n is None. Returns the count MKL now uses."""
    if n is not None:
        MKL.MKL_Set_Num_Threads(ctypes.c_int(n))     # the C version takes the number by value
    return MKL.MKL_Get_Max_Threads()


DEFAULT = mkl_threads()
THREADS = [DEFAULT] + [n for n in (1, 2, 4, 8, 16) if n < DEFAULT]
print(f'MKL {modelfactorize.mkl_version() or "version unknown"}, default threads {DEFAULT}, testing {THREADS}')

m, basedf = model.modelload(str(MODELTEST / 'data' / 'frbusmcetest.pcim'), run=0)
alt = basedf.upd('<2020q1 2020q1> rffintay_aerr + 1 ')


def timed(f):
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        res = f()
        return time.perf_counter() - t0, res
    finally:
        gc.enable()


def solve_model(factorizer):
    return m(alt, solver='newtonstack_ng', silent=1, reset_options=True, ljit=False,
             newton_reset=1, factorizer=factorizer, **opts)


#%% the stacked jacobian at the solution
res = solve_model('pardiso')
diff = m.ng_newton_diff_stack
mask = np.array([v.endswith('___RES') for v in diff.endovar], dtype=bool)
A = diff.get_diff_mat_tot(df=res, is_residual_eq=mask).tocsr()
b = np.random.default_rng(0).standard_normal(A.shape[0])
print(f'stacked jacobian: n={A.shape[0]:,}, nnz={A.nnz:,}')

#%% run
rows = []
try:
    for factorizer in FACTORIZERS:
        solve_model(factorizer)                                   # warm up
        for n in THREADS:
            used = mkl_threads(n)
            factor, solve, rebuild = [], [], []
            for _ in range(REPEAT):
                t, s = timed(lambda: modelfactorize.make_solver(A, factorizer))
                factor.append(t)
                t, x = timed(lambda: s(b))
                solve.append(t)
                modelfactorize.release(s)
                t, _ = timed(lambda: solve_model(factorizer))
                rebuild.append(t)
            resid = np.abs(A @ x - b).max() / np.abs(b).max()
            rows.append(dict(factorizer=factorizer, threads=used, factor=min(factor), solve=min(solve),
                             rebuild=min(rebuild), build=m.ng_stackbuildtime, resid=resid))
            print(rows[-1])
finally:
    mkl_threads(DEFAULT)

table = pd.DataFrame(rows).set_index(['factorizer', 'threads'])
with pd.option_context('display.float_format', '{:,.3f}'.format, 'display.width', 160):
    print(table.drop(columns='resid'))
    print(table['resid'].map('{:.1e}'.format))
