# -*- coding: utf-8 -*-
"""
Tests of variables at a fixed period in the business language: X(@2002) is the
value of X in 2002, X(@2002Q1) in a quarterly model.

* The term is parsed like a lag; its lag field is '@2002'.
* Only the ng solvers evaluate it; model(df, ...) picks one, the legacy
  solvers raise an exception.
* The generated code reads the databank row of 2002 - a constant, so the code
  is regenerated when the same columns come with another index.
* Inside the solve span per-period solvers warn; newtonstack_ng is exact
  (the derivative sits in the column of 2002 of the stacked Jacobian).

Run the #%% cells in order.  Each cell prints what it checks; an assert stops
at the first wrong number.
"""
#%% setup
import numpy as np
import pandas as pd

import modelclass
from modelclass import model
import modelpattern as pt
import modelmanipulation as mp
import modelnormalize as nz
from modelnewton import newton_diff

print('modelclass imported from', modelclass.__file__)


def close(a, b, tol=1e-8):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tol, rtol=tol)


def expect_error(f, what):
    try:
        f()
    except Exception as e:
        print(f'As expected, {what}:\n  {type(e).__name__}: {e}\n')
    else:
        raise AssertionError(f'{what} did not raise')


years = list(range(2000, 2011))
n = len(years)

#%% 1. the tokenizer
terms = pt.udtryk_parse('Y = X/X(@2002)*100 + Z(-1) + W(@2002q1)')
for t in terms:
    print(t)
varlags = [(t.var, t.lag) for t in terms if t.var]
assert ('X', '@2002') in varlags
assert ('W', '@2002Q1') in varlags
assert ('Z', '-1') in varlags
assert [pt.lag_value(l) for l in ['', '-1', '+2', '@2002']] == [0, -1, 2, '@2002']
print('ok')

#%% 2. text utilities: lagone, pastestring/stripstring, syntax check
print(mp.lagone('X/X(@2002)+Y(-1)'))
assert mp.lagone('X/X(@2002)') == 'X(-1)/X(@2002)'      # the fixed period is not shifted
pasted = mp.pastestring('X/X(@2002)+Y(-1)', '___LAG', onlylags=True)
print(pasted)
assert pasted == 'X/X___LAG___AT___2002+Y___LAG(-1)'      # sympy can read this
assert mp.stripstring(pasted, '___LAG') == 'X/X(@2002)+Y(-1)'
assert mp.check_syntax_udtryk_new('Y = X/X(@2002)*100')[0]
assert mp.check_syntax_udtryk_new('Y = X/X(@2002Q1)*100')[0]
print(mp.check_syntax_udtryk_new('Y = X/X(@2002)*(100')[1])   # a real error is still found
print('ok')

#%% 3. modelnormalize
print(nz.normal('LOG(Y/Y(@2002)) = 0.5*LOG(X)', add_add_factor=False), '\n')
print(nz.normal('DLOG(Y) = DLOG(X) + 0.1*(LOG(Y(-1))-LOG(Y(@2002)))'), '\n')
print(nz.normal('Y = X/X(@2002)*100', make_fixable=True, make_fitted=True), '\n')
print(nz.normal('a = movavg(x/x(@2002),2)', add_add_factor=False), '\n')
assert 'Y(@2002)' in nz.normal('LOG(Y/Y(@2002)) = 0.5*LOG(X)', add_add_factor=False).normalized
expect_error(lambda: nz.normal('Y = D(@2002) + X'),
             'a variable named like a function (D) at a fixed period')

#%% 4. a recursive model: model() picks xgenr_ng
df = pd.DataFrame({'X': np.linspace(10, 20, n)}, index=years)
mrec = model('Y = X/X(@2002)*100 \n Z = Y(-1) + X(@2005)', modelname='fixrec')
print(f'{mrec.fixed_terms=}\n{mrec.fixed_periods=}\n{mrec.maxlag=} {mrec.maxlead=}')
res = mrec(df, 2003, 2010, silent=1)
expected_y = df.X / df.loc[2002, 'X'] * 100
assert close(res.loc[2003:2010, 'Y'], expected_y.loc[2003:2010])
assert close(res.loc[2004:2010, 'Z'],
             res.Y.shift().loc[2004:2010] + df.loc[2005, 'X'])
print(res)
print(mrec.make_los_text)          # the generated code: values[2,...] is X in 2002

#%% 5. the same columns with another index: the row of 2002 moves, the code is regenerated
res_late = mrec(df.loc[2001:], 2003, 2010, silent=1)
assert close(res_late.loc[2003:2010, 'Y'], expected_y.loc[2003:2010])
print({k: v for k, v in vars(mrec).items() if k.startswith('ng_fixrow_')})
expect_error(lambda: mrec(df.loc[2003:], 2004, 2010, silent=1),
             '2002 is not in the dataframe')
print('ok')

#%% 6. mfcalc
df2 = df.mfcalc('IDX = X/X(@2002)*100')
assert close(df2.IDX, df.X / df.loc[2002, 'X'] * 100)
print(df2)

#%% 7. a simultaneous model: every ng solver against the old workaround
# (Y(@2002) replaced by an exogenous variable holding the 2002 value, legacy sim)
ftext = '''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(@2002)
I = 0.1*Y(-1) + IX
Y = C + I + G
'''
dfs = pd.DataFrame({'G': np.linspace(20, 30, n), 'IX': np.linspace(5, 8, n),
                    'C': 60.0, 'I': 15.0, 'Y': 100.0}, index=years)
dfs.loc[2002, 'Y'] = 95.0
msim = model(ftext, modelname='fixsim')
mref = model(ftext.replace('Y(@2002)', 'Y_AT_2002'), modelname='fixref')
ref = mref(dfs.assign(Y_AT_2002=dfs.loc[2002, 'Y']), 2005, 2010, silent=1)

for solver in ['sim_ng', 'newton_ng', 'newtonstack_ng', 'newton_fbmin_ng',
               'newtonstack_fbmin_ng']:
    out = getattr(msim, solver)(dfs, 2005, 2010, silent=1)
    diff = (out.loc[2005:2010, ['C', 'I', 'Y']]
            - ref.loc[2005:2010, ['C', 'I', 'Y']]).abs().max().max()
    print(f'{solver:22} max abs difference to the workaround: {diff:.2e}')
    assert diff < 1e-4
out = msim(dfs, 2005, 2010, silent=1)
print(f'model() used: {msim.model_solver.__name__}')

#%% 8. legacy solvers and sim1d refuse the model
expect_error(lambda: msim.sim(dfs, 2005, 2010), 'legacy sim')
expect_error(lambda: msim.res(dfs, 2005, 2010), 'legacy res')
expect_error(lambda: msim.sim1d_ng(dfs, 2005, 2010), 'sim1d_ng')

#%% 9. the fixed period inside the solve span: per-period solvers warn, stacked Newton is exact
mins = model('Y = 0.5*Y(-1) + X \n Z = Y/Y(@2006)*100', modelname='fixinspan')
dfi = pd.DataFrame({'X': np.linspace(1, 3, n), 'Y': 1.0, 'Z': 0.0}, index=years)
xg = mins.xgenr_ng(dfi, 2003, 2010, silent=1)          # prints a warning
st = mins.newtonstack_ng(dfi, 2003, 2010, silent=1)    # exact
sf = mins.newtonstack_fbmin_ng(dfi, 2003, 2010, silent=1)
exact_z = st.Y / st.loc[2006, 'Y'] * 100
print(pd.DataFrame({'xgenr_ng': xg.Z, 'newtonstack_ng': st.Z,
                    'newtonstack_fbmin_ng': sf.Z, 'exact': exact_z}).loc[2003:2010])
assert close(st.loc[2003:2010, 'Z'], exact_z.loc[2003:2010], 1e-6)
assert close(sf.loc[2003:2010, 'Z'], exact_z.loc[2003:2010], 1e-6)
# from 2006 on the per-period solver is right too
assert close(xg.loc[2006:2010, 'Z'], exact_z.loc[2006:2010], 1e-6)
print('ok')

#%% 10. the stacked Jacobian: d Z / d Y(@2006) is in the column of 2006
mins.smpl(2003, 2010, dfi)
nd = newton_diff(mins, df=st, forcenum=True)
print(nd.diff_model.equations)
jac = nd.get_diff_df_tot(df=st)
print(jac.loc[(slice(None), 'Z'), (slice(None), 'Y')].round(2))

#%% 11. a simultaneous model with the fixed period inside the span
# C depends on Y(@2006) inside the Newton core; Z is reporting (the epilog with split=1)
msim2 = model('''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(@2006)
Y = C + G
Z = Y/Y(@2006)*100
''', modelname='fixinspan2')
dfs2 = pd.DataFrame({'G': np.linspace(20, 30, n), 'C': 60.0, 'Y': 100.0, 'Z': 0.0},
                    index=years)
solutions = {
    'newtonstack_ng':         msim2.newtonstack_ng(dfs2, 2003, 2010, silent=1),
    'newtonstack_ng split=1': msim2.newtonstack_ng(dfs2, 2003, 2010, silent=1, split=1),
    'newtonstack_fbmin_ng':   msim2.newtonstack_fbmin_ng(dfs2, 2003, 2010, silent=1),
    'sim_ng (warns)':         msim2.sim_ng(dfs2, 2003, 2010, silent=1),
}
print(pd.DataFrame({k: v.Y for k, v in solutions.items()}).loc[2003:2010])
for name, sol in solutions.items():
    # the res check: each equation evaluated at the solution gives the solution back
    check = msim2.res_ng(sol, 2003, 2010, silent=1)
    err = (check - sol).loc[2003:2010, ['C', 'Y', 'Z']].abs().max().max()
    print(f'{name:24} max equation residual {err:.2e}')
    if not name.startswith('sim_ng'):
        assert err < 1e-5


#%% 12. quarterly
qidx = pd.period_range('2001Q1', '2004Q4', freq='Q')
dfq = pd.DataFrame({'X': np.linspace(10, 25, len(qidx))}, index=qidx)
mq = model('Y = X/X(@2002Q1)*100', modelname='fixq')
resq = mq(dfq, '2002Q2', '2004Q4', silent=1)
assert close(resq.loc['2002Q2':, 'Y'],
             (dfq.X / dfq.loc['2002Q1', 'X'] * 100).loc['2002Q2':])
print(resq)

#%% 13. equation values and attribution
base = mrec(df, 2003, 2010, silent=1)
alt = mrec(df.assign(X=df.X * 1.1), 2003, 2010, silent=1)   # also X in 2002 changes
mrec.basedf, mrec.lastdf = base, alt
print(mrec.get_eq_values('Y', showvar=True))
dek = mrec.dekomp('Y')     # X and X(@2002) contribute with opposite signs, Y is unchanged

#%% 14. jit (numba): the row of 2002 is a constant in the compiled code
out_nojit = msim.sim_ng(dfs, 2005, 2010, silent=1)
out_jit = msim.sim_ng(dfs, 2005, 2010, silent=1, ljit=1, stringjit=1)
assert close(out_jit.loc[2005:2010, ['C', 'I', 'Y']], out_nojit.loc[2005:2010, ['C', 'I', 'Y']])
print('ok')

#%% 15. a variable at a fixed period can not be on the left hand side
expect_error(lambda: model('Y(@2002) = X'), 'a fixed period on the left hand side')
