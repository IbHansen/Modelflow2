# -*- coding: utf-8 -*-
"""
Tests of the fill plan of the stacked Jacobian (phase 1 of
experiments/sparse fill stacked newton/sparse_fill_design.md):
newton_diff._stacked_plan and newton_diff._stacked_from_plan.

Every case builds the stacked Jacobian twice from the same derivative values:

* the former way: get_diff_mat_tot_melt, and the former "- diag_mask" of
  get_solvestacked for a model with ___RES equations;
* from the plan, with the values the former way used;
* with the new get_diff_mat_tot, which evaluates the derivatives itself.

The two must have the same values and, when the zeros are taken out of both, the
same pattern. (Today's matrix drops zeros, the matrix from the plan keeps them:
the pattern from the plan does not change from refresh to refresh.)

Cell 7 solves with the stacked solvers, which now build the Jacobian from the plan.

Run the #%% cells in order. An assert stops at the first difference.
"""
#%% setup
import numpy as np
import pandas as pd
import scipy.sparse as sps

import modelclass
from modelclass import model
from modelnewton import newton_diff

print('modelclass imported from', modelclass.__file__)

years = list(range(2000, 2011))
n = len(years)


def both(m, df, start, end, endovar=None, forcenum=False):
    """(today's matrix, matrix from the plan, plan, newton_diff) for start..end"""
    m.current_per = df.loc[start:end].index       # the periods of the stacked system
    diff = newton_diff(m, df=df, endovar=endovar, forcenum=forcenum)
    old = diff.get_diff_mat_tot_melt(df=df)
    isres = np.array([v.endswith('___RES') for v in diff.endovar], dtype=bool)
    mask = np.tile(isres, len(diff.numbers)) if isres.any() else None
    if mask is not None and not m.normalized:
        old = old - sps.diags((~mask).astype(float))          # as get_solvestacked
    # the values get_diff_mat_tot_melt just used: (periods, derivatives), the last column is 'number'
    values = diff.difres.iloc[:, :-1].to_numpy(dtype=float)
    plan = diff._stacked_plan(diff.difres.index, is_residual_eq=mask)
    new = diff._stacked_from_plan(plan, values)
    # the new get_diff_mat_tot: the same matrix, bit for bit, from its own evaluation
    tot = diff.get_diff_mat_tot(df=df, is_residual_eq=mask)
    assert tot is diff.stacked and diff._stackplan is plan
    assert np.array_equal(tot.indptr, new.indptr) and np.array_equal(tot.indices, new.indices)
    assert np.array_equal(tot.data, new.data), 'get_diff_mat_tot differs from the plan with the old values'
    assert diff.maxnumber == len(diff.numbers) - 1 and list(diff.numbers_to_date.values()) == list(df.loc[start:end].index)
    return old.tocsc(), new, plan, diff


def same(old, new, name):
    """Same values, and the same pattern without the zeros"""
    assert new.shape == old.shape, name
    assert new.has_canonical_format, f'{name}: the matrix from the plan is not sorted or has duplicates'
    maxdiff = abs(new - old).max() if old.nnz or new.nnz else 0.0
    a, b = old.copy(), new.copy()
    for x in (a, b):
        x.sum_duplicates()
        x.eliminate_zeros()
        x.sort_indices()
    print(f'{name:38} shape {new.shape}, nnz plan {new.nnz}, nnz today {old.nnz}, '
          f'without zeros {b.nnz}/{a.nnz}, max abs difference {maxdiff:.1e}')
    assert maxdiff == 0.0, name
    assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices), name
    assert np.array_equal(a.data, b.data), name


#%% 1. normalized model with lags and leads
m1 = model('''
C = 0.6*Y + 0.2*C(-1) + 0.05*Y(+1)
I = 0.2*(Y - Y(-2)) + 0.1*I(+2)
Y = C + I + G
''', modelname='plan1')
df1 = pd.DataFrame({'G': np.linspace(20, 30, n), 'C': 60.0, 'I': 20.0, 'Y': 100.0}, index=years)
old, new, plan, diff = both(m1, df1, 2002, 2008)
same(old, new, '1. lags and leads')
assert m1.normalized
# nonlinear, so the values depend on the data
m1b = model('''
C = 0.6*Y**0.9 + 0.2*C(-1) + 0.05*LOG(Y(+1))
Y = C + G
''', modelname='plan1b')
old, new, plan, diff = both(m1b, df1, 2002, 2008)
same(old, new, '1b. nonlinear')
print('ok')

#%% 2. a model with ___RES equations and the residual mask
m2 = model('''
FRML <> C = 0.6*Y + 0.2*C(-1) $
FRML <ENDO=Y> Y___RES = Y - C - G - 0.1*Y(+1) $
''', modelname='plan2')
assert sorted(m2.endogene) == ['C', 'Y___RES'] and not m2.normalized
df2 = df1.assign(Y___RES=0.0)
print('normalized:', m2.normalized)
old, new, plan, diff = both(m2, df2, 2002, 2008)
same(old, new, '2. ___RES and mask')
assert len(plan.const_slot) == 7           # the -1 only on the rows of C
# the mask with one value per equation gives the same plan
isres = np.array([v.endswith('___RES') for v in diff.endovar], dtype=bool)
delattr(diff, '_stackplan')
plan_short = diff._stacked_plan(diff.difres.index, is_residual_eq=isres)
assert np.array_equal(plan_short.indices, plan.indices) and np.array_equal(plan_short.slot, plan.slot)
print('ok')

#%% 3. a fixed-period reference inside and outside the periods
m3 = model('''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(2006)
Y = C + G
Z = Y/Y(2006)*100
''', modelname='plan3')
df3 = df1.assign(Z=0.0)
old, new, plan, diff = both(m3, df3, 2003, 2010)
same(old, new, '3. Y(2006) inside 2003-2010')
old, new, plan, diff = both(m3, df3, 2003, 2006)
same(old, new, '3. Y(2006) the last period')
old, new, plan, diff = both(m3, df3, 2007, 2010)
same(old, new, '3. Y(2006) outside 2007-2010')
# monthly, the period in quotes
midx = pd.period_range('2001-01', '2002-12', freq='M')
m3m = model("Y = 0.5*Y(-1) + X \n Z = Y/Y('2001-06')*100", modelname='plan3m')
dfm = pd.DataFrame({'X': np.linspace(1, 3, len(midx)), 'Y': 1.0, 'Z': 0.0}, index=midx)
old, new, plan, diff = both(m3m, dfm, '2001-02', '2002-12')
same(old, new, "3. Y('2001-06') monthly")
print('ok')

#%% 4. duplicates: the left hand side variable on the right hand side, and Y next to Y(2006) in 2006
m4 = model('''
C = 0.6*Y + 0.2*C(-1) + 0.1*LOG(C)
Y = C + G + 0.3*Y + 0.1*Y(2006)
''', modelname='plan4')
old, new, plan, diff = both(m4, df1, 2003, 2010)
same(old, new, '4. duplicates on the diagonal')
entries = int(plan.keep.sum()) + len(plan.const_slot)
print(f'entries {entries}, places {plan.nnz}')
assert entries > plan.nnz                  # some entries share a place
print('ok')

#%% 5. the plan itself: cached, remade for other periods, NaN and inf
old, new, plan, diff = both(m1b, df1, 2002, 2008)
assert diff._stacked_plan(diff.difres.index) is plan                    # same periods: the same plan
values = diff.difres.iloc[:, :-1].to_numpy(dtype=float)
# a new matrix every time, with its own arrays
again = diff._stacked_from_plan(plan, values)
assert again.data is not new.data and again.indices is not plan.indices
again.eliminate_zeros()
assert np.array_equal(diff._stacked_from_plan(plan, values).indices, plan.indices)
# NaN counts as 0, inf stays inf
vbad = values.copy()
j = int(np.flatnonzero(plan.keep[0])[0])
vbad[0, j], vbad[1, j] = np.nan, np.inf
bad = diff._stacked_from_plan(plan, vbad)
assert not np.isnan(bad.data).any() and np.isinf(bad.data).sum() == 1
# other periods: another plan
plan2 = diff._stacked_plan(df1.loc[2003:2006].index)
assert plan2 is not plan and plan2.shape == (4 * diff.nvar, 4 * diff.nvar)
# a mask of the wrong length is refused
try:
    m2.current_per = df2.loc[2002:2008].index
    newton_diff(m2, df=df2)._stacked_plan(df2.loc[2002:2008].index, is_residual_eq=np.zeros(5, dtype=bool))
    raise AssertionError('a mask of the wrong length did not raise')
except ValueError as e:
    print('As expected:', e)
print('ok')

#%% 6. bigger and timed: a ring of equations with lags and leads
nv, nyears = 300, 40
eqs = '\n'.join(f'X{i} = 0.3*X{(i+1) % nv} + 0.2*X{i}(-1) + 0.1*X{(i+7) % nv}(+1) + 0.01*X{(i+3) % nv}**2 + E{i}'
                for i in range(nv))
m6 = model(eqs, modelname='plan6')
idx6 = list(range(2000, 2000 + nyears + 4))
df6 = pd.DataFrame({**{f'X{i}': 1.0 + i / nv for i in range(nv)},
                    **{f'E{i}': 0.1 for i in range(nv)}}, index=idx6)
old, new, plan, diff = both(m6, df6, idx6[2], idx6[-3], forcenum=True)
same(old, new, '6. ring 300 x 40')

import time
values = diff.difres.iloc[:, :-1].to_numpy(dtype=float)
delattr(diff, '_stackplan')
t0 = time.perf_counter()
plan = diff._stacked_plan(diff.difres.index)
t1 = time.perf_counter()
for _ in range(10):
    new = diff._stacked_from_plan(plan, values)
t2 = time.perf_counter()
old = diff.get_diff_mat_tot_melt(df=df6)
t3 = time.perf_counter()
new = diff.get_diff_mat_tot(df=df6)
t4 = time.perf_counter()
print(f'nnz {plan.nnz}: plan once {t1 - t0:.3f} s, fill {(t2 - t1) / 10 * 1000:.2f} ms; with the evaluation: '
      f'get_diff_mat_tot_melt {t3 - t2:.3f} s, get_diff_mat_tot {t4 - t3:.3f} s')
print('ok')

#%% 7. the solvers: stacked solvers on the models above against a per-period solve or each other
def solved(m, df, start, end, cols, **kw):
    return {name: getattr(m, name)(df, start, end, silent=1, **kw).loc[start:end, cols]
            for name in ['newtonstack_ng', 'newtonstack_fbmin_ng']}


# no leads: the stacked solution is the per-period solution
m7 = model('''
C = 0.6*Y**0.9 + 0.2*C(-1)
Y = C + G
''', modelname='plan7')
ref = m7.newton_ng(df1, 2002, 2008, silent=1).loc[2002:2008, ['C', 'Y']]
for name, out in solved(m7, df1, 2002, 2008, ['C', 'Y']).items():
    print(f'{name:24} max abs difference to newton_ng {(out - ref).abs().max().max():.1e}')
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5), name      # the convergence tolerances of the solvers
# leads, a ___RES equation, a fixed period, nonlin refreshes: the equations hold at the solution
for m, df, s, e, kw in [(m1b, df1, 2002, 2008, {}), (m1b, df1, 2002, 2008, dict(nonlin=1)),
                        (m2, df2, 2002, 2008, {}), (m3, df3, 2003, 2010, {})]:      # m4 has no solution with C > 0
    for name in ['newtonstack_ng', 'newtonstack_fbmin_ng']:
        out = getattr(m, name)(df, s, e, silent=1, newton_reset=1, **kw)
        chk = m.res_ng(out, s, e, silent=1)
        endo = [v for v in m.endogene if not v.endswith('___RES')]
        resv = [v for v in m.endogene if v.endswith('___RES')]
        err = float((chk.loc[s:e, endo] - out.loc[s:e, endo]).abs().max().max())
        if resv:
            err = max(err, float(chk.loc[s:e, resv].abs().max().max()))
        print(f'{m.name:8} {name:24} {str(kw):14} max abs error in the equations {err:.1e}')
        assert err < 1e-6, (m.name, name)
# a second solve over other periods: a new plan
out = m7.newtonstack_ng(df1, 2004, 2009, silent=1)
assert np.allclose(out.loc[2004:2009, ['C', 'Y']], m7.newton_ng(df1, 2004, 2009, silent=1).loc[2004:2009, ['C', 'Y']], atol=1e-5)
assert m7.ng_newton_diff_stack._stackplan.shape[0] % 6 == 0
print('ok')
