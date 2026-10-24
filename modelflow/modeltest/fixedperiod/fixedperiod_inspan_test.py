# -*- coding: utf-8 -*-
"""
Tests of an endogenous variable at a fixed period inside the solve span:
Z = Y/Y(2006)*100 solved 2003-2010.

* The stacked solvers, newtonstack_ng (split=1 and split=0) and
  newtonstack_fbmin_ng, solve every period at once: Y(2006) is a column of the
  stacked Jacobian, a node of the stacked graph. They solve it exactly.
* The per-period solvers (xgenr_ng, sim_ng, newton_ng, newton_fbmin_ng) still
  refuse it: in the periods before 2006 they would read the starting value of
  Y in 2006. The message points to the stacked solvers.
* model() picks a per-period solver, so solver='newtonstack_ng' must be asked for.

The references are computed by hand in numpy, not with modelflow.
fixedperiod_test.py has the rest of the fixed-period tests.

Run the #%% cells in order.  Each cell prints what it checks; an assert stops
at the first wrong number.
"""
#%% setup
import numpy as np
import pandas as pd

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)

STACKED = ['newtonstack_ng', 'newtonstack_fbmin_ng']


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tol, rtol=tol)


def expect_error(f, what):
    try:
        f()
    except Exception as e:
        print(f'As expected, {what}:\n  {type(e).__name__}: {e}\n')
        return str(e)
    raise AssertionError(f'{what} did not raise')


def solve_all(m, df, start, end):
    """{name: result} from every stacked solver, newtonstack_ng also with split=0."""
    out = {name: getattr(m, name)(df, start, end, silent=1) for name in STACKED}
    out['newtonstack_ng split=0'] = m.newtonstack_ng(df, start, end, silent=1, split=0)
    return out


def recursive_ref(df, start, end, fixper):
    """Y = 0.5*Y(-1) + X, then Z = Y/Y(fixper)*100 over start..end, by position
    (works for a year index and a period index alike)."""
    ref = df.copy()
    first, last = ref.index.get_loc(start), ref.index.get_loc(end)
    y, x = ref['Y'].to_numpy(dtype=float, copy=True), ref['X'].to_numpy(dtype=float)
    for i in range(first, last + 1):
        y[i] = 0.5 * y[i - 1] + x[i]
    ref['Y'] = y
    ref.iloc[first:last + 1, ref.columns.get_loc('Z')] = y[first:last + 1] / ref.loc[fixper, 'Y'] * 100
    return ref


years = list(range(2000, 2011))
n = len(years)

#%% 1. recursive: Z = Y/Y(2006)*100 with 2006 inside 2003-2010
mins = model('Y = 0.5*Y(-1) + X \n Z = Y/Y(2006)*100', modelname='inspanrec')
dfi = pd.DataFrame({'X': np.linspace(1, 3, n), 'Y': 1.0, 'Z': 0.0}, index=years)
ref = recursive_ref(dfi, 2003, 2010, 2006)
for name, out in solve_all(mins, dfi, 2003, 2010).items():
    diff = (out.loc[2003:2010, ['Y', 'Z']] - ref.loc[2003:2010, ['Y', 'Z']]).abs().max().max()
    print(f'{name:24} max abs difference to the hand calculation: {diff:.2e}')
    assert close(out.loc[2003:2010, ['Y', 'Z']], ref.loc[2003:2010, ['Y', 'Z']])
    assert close(out.loc[2006, 'Z'], 100.0)          # Y(2006)/Y(2006)*100
# the split: Z reads Y in 2006 - a later period for 2003-2005 - so Z is in the core.
# Z reads no core variable (Y is in the prolog), the core Jacobian would be -I, so
# split=1 solves without split
print(mins.ng_stack_split_struct)
assert mins.ng_stack_split_struct['core'] == ['Z']
assert not mins.ng_stack_split_struct['coupled']
print('ok')

#%% 2. the fixed period as the first and as the last period of the span
for start, end in [(2006, 2010), (2003, 2006)]:
    ref = recursive_ref(dfi, start, end, 2006)
    for name, out in solve_all(mins, dfi, start, end).items():
        assert close(out.loc[start:end, ['Y', 'Z']], ref.loc[start:end, ['Y', 'Z']]), \
            f'{name} {start}-{end}'
    print(f'{start}-{end} ok')

#%% 3. the per-period solvers still refuse it, the message names the stacked solvers
for solver in ['xgenr_ng', 'sim_ng', 'newton_ng']:
    msg = expect_error(lambda: getattr(mins, solver)(dfi, 2003, 2010, silent=1),
                       f'{solver} with Y(2006) inside 2003-2010')
    assert 'newtonstack_ng' in msg
expect_error(lambda: mins(dfi, 2003, 2010, silent=1), 'model() with its default solver')
# model() with a stacked solver
out = mins(dfi, 2003, 2010, silent=1, solver='newtonstack_ng')
assert close(out.loc[2003:2010, ['Y', 'Z']], recursive_ref(dfi, 2003, 2010, 2006).loc[2003:2010, ['Y', 'Z']])
print('ok')

#%% 4. simultaneous: Y(2006) feeds back into C in every period
ftext2 = '''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(2006)
Y = C + G
Z = Y/Y(2006)*100
'''
msim2 = model(ftext2, modelname='inspansim')
dfs2 = pd.DataFrame({'G': np.linspace(20, 30, n), 'C': 60.0, 'Y': 100.0, 'Z': 0.0},
                    index=years)


def sim_ref(df, start, end, yat):
    """The model with yat in place of Y(2006): C = (0.6*G + 0.2*C(-1) + 0.1*yat)/0.4"""
    ref = df.copy()
    for t in range(start, end + 1):
        c = (0.6 * ref.loc[t, 'G'] + 0.2 * ref.loc[t - 1, 'C'] + 0.1 * yat) / 0.4
        ref.loc[t, ['C', 'Y']] = [c, c + ref.loc[t, 'G']]
    return ref


# the value read as Y(2006) must be the solution for Y in 2006: a fixed point
yat = dfs2.loc[2006, 'Y']
for _ in range(100):
    yat = sim_ref(dfs2, 2003, 2010, yat).loc[2006, 'Y']
ref2 = sim_ref(dfs2, 2003, 2010, yat)
assert abs(ref2.loc[2006, 'Y'] - yat) < 1e-10
ref2.loc[2003:2010, 'Z'] = ref2.loc[2003:2010, 'Y'] / yat * 100
print(f'Y in 2006: {yat:.6f}')

for name, out in solve_all(msim2, dfs2, 2003, 2010).items():
    diff = (out.loc[2003:2010, ['C', 'Y', 'Z']] - ref2.loc[2003:2010, ['C', 'Y', 'Z']]).abs().max().max()
    print(f'{name:24} max abs difference to the hand calculation: {diff:.2e}')
    assert close(out.loc[2003:2010, ['C', 'Y', 'Z']], ref2.loc[2003:2010, ['C', 'Y', 'Z']])
# C and Y are the simultaneous core, Z only reports
print(msim2.ng_stack_split_struct)
assert msim2.ng_stack_split_struct['core'] == ['C', 'Y']
assert msim2.ng_stack_split_struct['coupled']          # split=1 is used
assert 'Z' in msim2.ng_stack_split_struct['epi']
# model() picks sim_ng, which refuses
expect_error(lambda: msim2(dfs2, 2003, 2010, silent=1), 'model() with sim_ng')
print('ok')

#%% 5. the solution satisfies the equations: res_ng evaluates them at the solution
sol = msim2.newtonstack_ng(dfs2, 2003, 2010, silent=1)
chk = msim2.res_ng(sol, 2003, 2010, silent=1)
diff = (chk.loc[2003:2010, ['C', 'Y', 'Z']] - sol.loc[2003:2010, ['C', 'Y', 'Z']]).abs().max().max()
print(f'max abs residual: {diff:.2e}')
assert diff < 1e-6
print('ok')

#%% 6. forward looking: Y(2006) next to a lag and a lead of Y
mfw = model('Y = 0.5*Y(-1) + 0.3*Y(+1) + X + 0.1*Y(2006) \n Z = Y/Y(2006)*100',
            modelname='inspanlead')
span = list(range(2003, 2010))            # Y in 2010 is the terminal value (data)
pos = {t: i for i, t in enumerate(span)}
# the stacked system by hand: Y(t) - 0.5*Y(t-1) - 0.3*Y(t+1) - 0.1*Y(2006) = X(t)
A = np.eye(len(span))
b = dfi.loc[span, 'X'].to_numpy(dtype=float, copy=True)
for t in span:
    for coef, s in [(0.5, t - 1), (0.3, t + 1), (0.1, 2006)]:
        if s in pos:
            A[pos[t], pos[s]] -= coef
        else:
            b[pos[t]] += coef * dfi.loc[s, 'Y']
yref = np.linalg.solve(A, b)
zref = yref / yref[pos[2006]] * 100
for name, out in solve_all(mfw, dfi, 2003, 2009).items():
    diff = max(np.abs(out.loc[span, 'Y'] - yref).max(), np.abs(out.loc[span, 'Z'] - zref).max())
    print(f'{name:24} max abs difference to the hand calculation: {diff:.2e}')
    assert close(out.loc[span, 'Y'], yref) and close(out.loc[span, 'Z'], zref)
print('ok')

#%% 7. monthly: Y('2001-06') inside 2001-02 to 2002-12
midx = pd.period_range('2001-01', '2002-12', freq='M')
dfm = pd.DataFrame({'X': np.linspace(1, 3, len(midx)), 'Y': 1.0, 'Z': 0.0}, index=midx)
mm = model("Y = 0.5*Y(-1) + X \n Z = Y/Y('2001-06')*100", modelname='inspanm')
refm = recursive_ref(dfm, '2001-02', '2002-12', '2001-06')
for name, out in solve_all(mm, dfm, '2001-02', '2002-12').items():
    assert close(out.loc['2001-02':, ['Y', 'Z']], refm.loc['2001-02':, ['Y', 'Z']]), name
    print(f'{name:24} ok')

#%% 8. the same columns with another index: the row of 2006 moves, the code is regenerated
res_late = msim2.newtonstack_ng(dfs2.loc[2001:], 2003, 2010, silent=1)
assert close(res_late.loc[2003:2010, ['C', 'Y', 'Z']], ref2.loc[2003:2010, ['C', 'Y', 'Z']])
res_late = msim2.newtonstack_fbmin_ng(dfs2.loc[2001:], 2003, 2010, silent=1)
assert close(res_late.loc[2003:2010, ['C', 'Y', 'Z']], ref2.loc[2003:2010, ['C', 'Y', 'Z']])
print('ok')

#%% 9. jit (numba): the stacked split with the row of 2006 compiled in
out_nojit = msim2.newtonstack_ng(dfs2, 2003, 2010, silent=1)
out_jit = msim2.newtonstack_ng(dfs2, 2003, 2010, silent=1, ljit=1, stringjit=1)
assert close(out_jit.loc[2003:2010, ['C', 'Y', 'Z']], out_nojit.loc[2003:2010, ['C', 'Y', 'Z']])
print('ok')
