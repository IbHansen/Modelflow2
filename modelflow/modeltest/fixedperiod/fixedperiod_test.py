# -*- coding: utf-8 -*-
"""
Tests of variables at a fixed period in the business language: X(2002) is the
value of X in 2002, X('2002Q1') in a quarterly model, X('2002-03') in a monthly.

* A fixed period is a year, an unsigned number with at least 4 digits (a lag or
  lead always has a sign), or text: the key which indexes the dataframe, as in
  df.loc['2002Q1']. Both are legal Python.
* The term is parsed like a lag; its lag field is the period as written: '2002'
  or "'2002Q1'" (a lag or lead has a sign). The old form X(@2002) is not accepted.
* Only the ng solvers evaluate it; model(df, ...) picks one, the legacy
  solvers raise an exception.
* The generated code reads the databank row of 2002 - a constant, so the code
  is regenerated when the same columns come with another index.
* A per-period solver refuses an endogenous variable at a fixed period inside
  the solve span; res_ng, which only evaluates, allows it, and so do the
  stacked solvers, which solve it exactly (the stacked Jacobian has the
  derivative in the column of the period, cell 10) - tested in
  fixedperiod_inspan_test.py.

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
terms = pt.udtryk_parse("Y = X/X(2002)*100 + Z(-1) + W('2002q1') + V(\"2002-03\")")
for t in terms:
    print(t)
varlags = [(t.var, t.lag) for t in terms if t.var]
# the lag field is the period as written (text with single quotes)
assert ('X', '2002') in varlags
assert ('W', "'2002Q1'") in varlags
assert ('V', "'2002-03'") in varlags
assert ('Z', '-1') in varlags
assert [pt.lag_value(l) for l in ['', '-1', '+2', '2002']] == [0, -1, 2, '2002']
assert [pt.is_fixed_lag(l) for l in ['', '-1', '+2', '0', '2002', "'2002Q1'"]] == \
       [False, False, False, False, True, True]
# a year as text is the same period, a short label needs the quotes
assert pt.udtryk_parse("X/X('2002')") == pt.udtryk_parse('X/X(2002)')
assert [(t.var, t.lag) for t in pt.udtryk_parse("X('5')") if t.var] == [('X', "'5'")]
# not a fixed period: 1 to 3 unsigned digits (EViews leads, see cell 3b), the old @ form
# and a sub-annual period without quotes - all caught when a model is made (cell 16)
for text in ['X(1)', 'X(@2002)', 'X(2002Q1)']:
    assert [(t.var, t.lag) for t in pt.udtryk_parse(text) if t.var][0] == ('X', ''), text
# the text in the parentheses -> the lag field
assert [pt.lag_field(x) for x in ['-0', '-1', '2002', "'2002'", "'2002Q1'", '"2002-03"', "'5'"]] == \
       ['', '-1', '2002', '2002', "'2002Q1'", "'2002-03'", "'5'"]
print('ok')

#%% 2. text utilities: lagone, pastestring/stripstring, syntax check
print(mp.lagone('X/X(2002)+Y(-1)'))
assert mp.lagone('X/X(2002)') == 'X(-1)/X(2002)'      # the fixed period is not shifted
pasted = mp.pastestring('X/X(2002)+Y(-1)', '___LAG', onlylags=True)
print(pasted)
assert pasted == 'X/X___LAG___AT___2002+Y___LAG(-1)'      # sympy can read this
assert mp.stripstring(pasted, '___LAG') == 'X/X(2002)+Y(-1)'
pasted = mp.pastestring("X/X('2002-03')", '___LAG', onlylags=True)
print(pasted)
assert pasted == 'X/X___LAG___AT___2002_03'                  # - is not allowed in a name
assert mp.stripstring(pasted, '___LAG') == "X/X('2002-03')"
import ast
ast.parse("Y = X/X(2002)*100 + W('2002Q1')")                  # legal Python as it is
assert mp.check_syntax_udtryk_new('Y = X/X(2002)*100')[0]
assert mp.check_syntax_udtryk_new("Y = X/X('2002Q1')*100")[0]
print(mp.check_syntax_udtryk_new('Y = X/X(2002)*(100')[1])   # a real error is still found
print('ok')

#%% 3. modelnormalize
print(nz.normal('LOG(Y/Y(2002)) = 0.5*LOG(X)', add_add_factor=False), '\n')
print(nz.normal('DLOG(Y) = DLOG(X) + 0.1*(LOG(Y(-1))-LOG(Y(2002)))'), '\n')
print(nz.normal('Y = X/X(2002)*100', make_fixable=True, make_fitted=True), '\n')
print(nz.normal('a = movavg(x/x(2002),2)', add_add_factor=False), '\n')
assert 'Y(2002)' in nz.normal('LOG(Y/Y(2002)) = 0.5*LOG(X)', add_add_factor=False).normalized
norm = nz.normal("DLOG(Y) = DLOG(X) + 0.1*LOG(X('2002-03'))").normalized
print(norm)
assert "X('2002-03')" in norm
expect_error(lambda: nz.normal('Y = D(2002) + X'),
             'a variable named like a function (D) at a fixed period')

#%% 3b. EViews leads without sign: only 1 to 3 digits are leads
print(nz.fixleads("Y = ZLHP(1) + X/X(2002) + W('2002Q1')"))
assert nz.fixleads("Y = ZLHP(1) + X/X(2002) + W('2002Q1')") == "Y=ZLHP(+1)+X/X(2002)+W('2002Q1')"
norm = nz.normal('Y = ZLHP(1) + X/X(2002)', add_add_factor=False).normalized
print(norm)
assert 'ZLHP(+1)' in norm and 'X(2002)' in norm
print('ok')

#%% 4. a recursive model: model() picks xgenr_ng
df = pd.DataFrame({'X': np.linspace(10, 20, n)}, index=years)
mrec = model('Y = X/X(2002)*100 \n Z = Y(-1) + X(2005)', modelname='fixrec')
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
df2 = df.mfcalc('IDX = X/X(2002)*100')
assert close(df2.IDX, df.X / df.loc[2002, 'X'] * 100)
print(df2)

#%% 7. a simultaneous model: every ng solver against the old workaround
# (Y(2002) replaced by an exogenous variable holding the 2002 value, legacy sim)
ftext = '''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(2002)
I = 0.1*Y(-1) + IX
Y = C + I + G
'''
dfs = pd.DataFrame({'G': np.linspace(20, 30, n), 'IX': np.linspace(5, 8, n),
                    'C': 60.0, 'I': 15.0, 'Y': 100.0}, index=years)
dfs.loc[2002, 'Y'] = 95.0
msim = model(ftext, modelname='fixsim')
mref = model(ftext.replace('Y(2002)', 'Y_AT_2002'), modelname='fixref')
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

#%% 7b. symbolic derivatives with a fixed period inside (newton_ng differentiates with sympy)
# dC/dY = 0.6*P(2002)/P(2003) must keep P(2002) a fixed period - sympy's leads come back
# as X(1) and get their sign again, but only 1 to 3 digits, not the year in P(2002)
ftextp = '''
C = 0.6*Y*P(2002)/P(2003) + 0.2*C(-1)
I = 0.1*Y(-1) + IX
Y = C + I + G
'''
dfp = dfs.assign(P=np.linspace(1.0, 1.5, n))
mpx = model(ftextp, modelname='fixsympy')
nd_p = newton_diff(mpx, df=dfp, onlyendocur=True)     # symbolic, like newton_ng
print(nd_p.diffendocur['C'])
assert 'P(2002)' in nd_p.diffendocur['C']['Y'] and '(+' not in nd_p.diffendocur['C']['Y']
mrefp = model(ftextp.replace('P(2002)', 'P_AT_2002').replace('P(2003)', 'P_AT_2003'),
              modelname='fixsympyref')
refp = mrefp(dfp.assign(P_AT_2002=dfp.loc[2002, 'P'], P_AT_2003=dfp.loc[2003, 'P']),
             2005, 2010, silent=1)
outp = mpx.newton_ng(dfp, 2005, 2010, silent=1)
diff = (outp.loc[2005:2010, ['C', 'I', 'Y']] - refp.loc[2005:2010, ['C', 'I', 'Y']]).abs().max().max()
print(f'newton_ng max abs difference to the workaround: {diff:.2e}')
assert diff < 1e-4
print('ok')

#%% 8. legacy solvers and sim1d refuse the model
expect_error(lambda: msim.sim(dfs, 2005, 2010), 'legacy sim')
expect_error(lambda: msim.res(dfs, 2005, 2010), 'legacy res')
expect_error(lambda: msim.sim1d_ng(dfs, 2005, 2010), 'sim1d_ng')

#%% 9. an endogenous variable at a fixed period inside the solve span: the per-period solvers refuse
# (the stacked solvers solve it, see fixedperiod_inspan_test.py)
mins = model('Y = 0.5*Y(-1) + X \n Z = Y/Y(2006)*100', modelname='fixinspan')
dfi = pd.DataFrame({'X': np.linspace(1, 3, n), 'Y': 1.0, 'Z': 0.0}, index=years)
for solver in ['xgenr_ng', 'sim_ng', 'newton_ng']:
    expect_error(lambda: getattr(mins, solver)(dfi, 2003, 2010, silent=1),
                 f'{solver} with Y(2006) inside 2003-2010')
expect_error(lambda: mins(dfi, 2003, 2010, silent=1), 'model() with Y(2006) inside 2003-2010')
# allowed: a span after the period, where Y(2006) is data
after = mins(dfi, 2007, 2010, silent=1)
assert close(after.loc[2007:2010, 'Z'], after.Y.loc[2007:2010] / dfi.loc[2006, 'Y'] * 100)
# allowed: res only evaluates the input values
chk = mins.res_ng(dfi, 2003, 2010, silent=1)
assert close(chk.loc[2003:2010, 'Z'], dfi.Y.loc[2003:2010] / dfi.loc[2006, 'Y'] * 100)
# (an exogenous variable at a period inside the span is fine: X(2005) in cell 4)
print('ok')

#%% 10. the stacked Jacobian: d Z / d Y(2006) is in the column of 2006
mins.smpl(2003, 2010, dfi)
nd = newton_diff(mins, df=dfi, forcenum=True)
print(nd.diff_model.equations)
jac = nd.get_diff_df_tot(df=dfi)
print(jac.loc[(slice(None), 'Z'), (slice(None), 'Y')].round(2))

#%% 11. a simultaneous model with the fixed period inside the span is refused by the per-period solvers too
msim2 = model('''
C = 0.6*Y + 0.2*C(-1) + 0.1*Y(2006)
Y = C + G
Z = Y/Y(2006)*100
''', modelname='fixinspan2')
dfs2 = pd.DataFrame({'G': np.linspace(20, 30, n), 'C': 60.0, 'Y': 100.0, 'Z': 0.0},
                    index=years)
for solver in ['sim_ng', 'newton_ng', 'newton_fbmin_ng']:
    expect_error(lambda: getattr(msim2, solver)(dfs2, 2003, 2010, silent=1),
                 f'{solver}, simultaneous, Y(2006) inside 2003-2010')
print('ok')


#%% 12. quarterly
qidx = pd.period_range('2001Q1', '2004Q4', freq='Q')
dfq = pd.DataFrame({'X': np.linspace(10, 25, len(qidx))}, index=qidx)
mq = model("Y = X/X('2002Q1')*100", modelname='fixq')     # the key as in dfq.loc['2002Q1']
resq = mq(dfq, '2002Q2', '2004Q4', silent=1)
assert close(resq.loc['2002Q2':, 'Y'],
             (dfq.X / dfq.loc['2002Q1', 'X'] * 100).loc['2002Q2':])
print(resq)

#%% 12b. monthly: refused with the period inside the span, solved after it
midx = pd.period_range('2001-01', '2002-12', freq='M')
dfm = pd.DataFrame({'X': np.linspace(1, 3, len(midx)), 'Y': 1.0, 'Z': 0.0}, index=midx)
mm = model("Y = 0.5*Y(-1) + X \n Z = Y/Y('2001-06')*100", modelname='fixm')
print(mm.fixed_terms)
expect_error(lambda: mm(dfm, '2001-02', '2002-12', silent=1), "Y('2001-06') inside the span")
resm = mm(dfm, '2001-07', '2002-12', silent=1)
assert close(resm.loc['2001-07':, 'Z'], resm.Y.loc['2001-07':] / dfm.loc['2001-06', 'Y'] * 100)
nd_m = newton_diff(mm, df=dfm, forcenum=True)       # the derivative name carries the period
print(sorted(nd_m.diff_model.endogene))
assert any('___PER___2001_06' in v for v in nd_m.diff_model.endogene)
print('ok')

#%% 13. equation values and attribution
base = mrec(df, 2003, 2010, silent=1)
alt = mrec(df.assign(X=df.X * 1.1), 2003, 2010, silent=1)   # also X in 2002 changes
mrec.basedf, mrec.lastdf = base, alt
print(mrec.get_eq_values('Y', showvar=True))
dek = mrec.dekomp('Y')     # X and X(2002) contribute with opposite signs, Y is unchanged

#%% 14. jit (numba): the row of 2002 is a constant in the compiled code
out_nojit = msim.sim_ng(dfs, 2005, 2010, silent=1)
out_jit = msim.sim_ng(dfs, 2005, 2010, silent=1, ljit=1, stringjit=1)
assert close(out_jit.loc[2005:2010, ['C', 'I', 'Y']], out_nojit.loc[2005:2010, ['C', 'I', 'Y']])
print('ok')

#%% 15. a variable at a fixed period can not be on the left hand side
expect_error(lambda: model('Y(2002) = X'), 'a fixed period on the left hand side')

#%% 16. malformed lags and fixed periods are caught when the model is made
expect_error(lambda: model('Y = X/X( 2002 )*100'), 'blanks in a fixed period')
expect_error(lambda: model('Y = X(1) + Z'), 'an unsigned lead (EViews style) in model()')
expect_error(lambda: model('Y = X( -1 ) + Z'), 'blanks in a lag')
expect_error(lambda: model('Y = X/X(2002Q1)'), 'a sub-annual period without quotes')
expect_error(lambda: model('Y = X/X(@2002)'), 'the old X(@2002) form')
model('Y = X/X(2002)*100 + LOG(2) \n Z = A@B')     # a function of a number and a matrix product are fine
print('ok')
