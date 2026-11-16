"""
The stackfill option of the stacked solvers: 'plan' (the fill plan, default) and
'melt' (the former fill, get_diff_mat_tot_melt) give the same stacked jacobian
and the same solution.

Run the #%% cells in order. An assert stops at the first difference.
"""
#%% setup
import numpy as np
import pandas as pd

import modelclass
from modelclass import model
from modelnewton import newton_diff

print('modelclass imported from', modelclass.__file__)

years = list(range(2000, 2011))
n = len(years)
df1 = pd.DataFrame({'G': np.linspace(20, 30, n), 'C': 60.0, 'I': 20.0, 'Y': 100.0}, index=years)

m1 = model('''
C = 0.6*Y**0.9 + 0.2*C(-1) + 0.05*LOG(Y(+1))
I = 0.2*(Y - Y(-2)) + 0.1*I(+2)
Y = C + I + G
''', modelname='fill1')

m2 = model('''
FRML <> C = 0.6*Y + 0.2*C(-1) $
FRML <ENDO=Y> Y___RES = Y - C - G - 0.1*Y(+1) $
''', modelname='fill2')
df2 = df1.assign(Y___RES=0.0)
assert m1.normalized and not m2.normalized


def canon(a):
    a = a.tocsc().copy()
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return a


#%% 1. get_diff_mat_tot: the same matrix from both fills
for name, m, df in (('normalized', m1, df1), ('___RES', m2, df2)):
    m.current_per = df.loc[2002:2008].index
    diff = newton_diff(m, df=df, forcenum=True)
    isres = np.array([v.endswith('___RES') for v in diff.endovar], dtype=bool)
    for maskname, mask in (('no mask', None), ('per equation', isres), ('per row', np.tile(isres, 7))):
        diff.stackfill = 'plan'
        plan = canon(diff.get_diff_mat_tot(df=df, is_residual_eq=mask))
        diff.stackfill = 'melt'
        melt = diff.get_diff_mat_tot(df=df, is_residual_eq=mask)
        assert melt is diff.stacked
        melt = canon(melt)
        assert plan.shape == melt.shape
        assert np.array_equal(plan.indptr, melt.indptr) and np.array_equal(plan.indices, melt.indices), (name, maskname)
        assert np.array_equal(plan.data, melt.data), (name, maskname)
        print(f'{name:12} {maskname:13} shape {plan.shape} nnz {plan.nnz}: the same')
diff.stackfill = 'wrong'
try:
    diff.get_diff_mat_tot(df=df)
    raise AssertionError('an unknown stackfill was accepted')
except ValueError as e:
    print('unknown stackfill:', e)
print('ok')

#%% 2. the stacked solvers: the same solution with both fills
for solver in ('newtonstack_ng', 'newtonstack_fbmin_ng'):
    for name, m, df in (('normalized', m1, df1), ('___RES', m2, df2)):
        res = {}
        for fill in ('plan', 'melt'):
            res[fill] = m(df, 2002, 2008, solver=solver, stackfill=fill, newton_reset=True, silent=True)
            assert m.ng_newton_diff_stack.stackfill == fill
        # options are kept from call to call: without stackfill= the last one ('melt') is still used
        kept = m(df, 2002, 2008, solver=solver, newton_reset=True, silent=True)
        print('   without stackfill= after melt:', m.ng_newton_diff_stack.stackfill)
        maxdiff = (res['plan'] - res['melt']).abs().max().max()
        print(f'{solver:22} {name:12} max abs difference plan/melt {maxdiff:.1e}')
        assert maxdiff == 0.0, (solver, name)
        assert (kept - res['plan']).abs().max().max() == 0.0
print('ok')
