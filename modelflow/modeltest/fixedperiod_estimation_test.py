# -*- coding: utf-8 -*-
"""
Tests of variables at a fixed period, X(2005), in the estimators.

* The estimators parse the equation with ast. X(2005) and X('2005Q1') are legal
  Python, read like a lag X(-1): a call of X, so the parser needs nothing special.
* A variable ending in C, like PC(2005), is not the coefficient C(n).
* The estimates must equal those of the workaround: an exogenous column holding
  the 2005 value.

Run the #%% cells in order.
"""
#%% setup
import numpy as np
import pandas as pd

import modelclass
from modelclass import model
from modelestimator_new import Estimate_ols, Estimate_nls_lmfit, EquationParse

rng = np.random.default_rng(1)
years = list(range(2000, 2021))
n = len(years)
df = pd.DataFrame({'X': np.linspace(10, 30, n) + rng.normal(0, 1, n),
                   'Z': rng.normal(0, 1, n)}, index=years)
df['Y'] = 2.0 + 3.0 * df.X / df.loc[2005, 'X'] + 0.5 * df.Z + rng.normal(0, 0.05, n)
df['X_AT_2005'] = df.loc[2005, 'X']


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


#%% 1. the parser keeps a period given as text
p = EquationParse(org_eq_clean="Y = C__1 + C__2*(X/X('2005Q1')) + C__3*Z", ecm=False)
print(*p.mfcalc_code, sep='\n')
assert "C__2 = X / X('2005Q1')" in p.mfcalc_code
assert {'Y', 'X', 'Z'} <= p.used_vars
print('ok')

#%% 2. the parser: mfcalc lines with X(2005), used variables without the period
p = EquationParse(org_eq_clean='Y = C__1 + C__2*(X/X(2005)) + C__3*Z', ecm=False)
print(*p.mfcalc_code, sep='\n')
assert 'C__2 = X / X(2005)' in p.mfcalc_code
print(p.used_vars)
assert {'Y', 'X', 'Z'} <= p.used_vars
print('ok')

#%% 3. OLS: the same estimates as the workaround
ols = Estimate_ols(org_eq='Y = C(1) + C(2)*(X/X(2005)) + C(3)*Z', input_df=df,
                   smpl=(2001, 2020), ecm=False)
ref = Estimate_ols(org_eq='Y = C(1) + C(2)*(X/X_AT_2005) + C(3)*Z', input_df=df,
                   smpl=(2001, 2020), ecm=False)
print(ols.coef_estimate_dict, '\n', ref.coef_estimate_dict)
assert close(list(ols.coef_estimate_dict.values()), list(ref.coef_estimate_dict.values()))
print('ok')

#%% 4. NLS (lmfit): the same estimates as the workaround
nls = Estimate_nls_lmfit(org_eq='Y = C(1) + C(2)*(X/X(2005)) + C(3)*Z', input_df=df,
                         smpl=(2001, 2020), ecm=False)
nref = Estimate_nls_lmfit(org_eq='Y = C(1) + C(2)*(X/X_AT_2005) + C(3)*Z', input_df=df,
                          smpl=(2001, 2020), ecm=False)
print(nls.coef_estimate_dict, '\n', nref.coef_estimate_dict)
assert close(list(nls.coef_estimate_dict.values()), list(nref.coef_estimate_dict.values()), 1e-6)
print('ok')

#%% 5. a variable ending in C is not the coefficient C(n)
df['PC'] = df.X
pc = Estimate_ols(org_eq='Y = C(1) + C(2)*(PC/PC(2005)) + C(3)*Z', input_df=df,
                  smpl=(2001, 2020), ecm=False)
print(pc.coef_estimate_dict)
assert sorted(pc.coef_estimate_dict) == ['C__1', 'C__2', 'C__3']
assert close(list(pc.coef_estimate_dict.values()), list(ref.coef_estimate_dict.values()))
print('ok')

#%% 6. Makemodel: estimation and normalization (fixleads) keep X(2005) a fixed period
from modelconstruct_estimation import Makemodel
mm = Makemodel('''FRML <estimator=ols,smpl=2001 2020> Y = C(1) + C(2)*(X/X(2005)) + C(3)*Z $
FRML <> W = Y/Y(2005)*100 $''', input_df=df, type_input='modelflow')
print(mm.normal_frml)
assert 'X/X(2005)' in mm.normal_frml and 'Y/Y(2005)' in mm.normal_frml
# W uses the endogenous Y in 2005: not allowed with 2005 inside the solve span (for now)
expect_error(lambda: mm.mmodel(df, 2001, 2020, silent=1), 'Y(2005) inside the solve span')
mres = mm.mmodel(df, 2006, 2020, silent=1)          # after 2005: Y(2005) is data
assert close(mres.loc[2006:2020, 'W'], mres.Y.loc[2006:2020] / df.loc[2005, 'Y'] * 100)
print('ok')

#%% 7. quarterly: the period as text, as for indexing the dataframe
qidx = pd.period_range('2001Q1', '2005Q4', freq='Q')
nq = len(qidx)
dfq = pd.DataFrame({'X': np.linspace(10, 30, nq) + rng.normal(0, 1, nq),
                    'Z': rng.normal(0, 1, nq)}, index=qidx)
dfq['Y'] = 2.0 + 3.0 * dfq.X / dfq.loc['2002Q1', 'X'] + 0.5 * dfq.Z + rng.normal(0, 0.05, nq)
dfq['X_AT'] = dfq.loc['2002Q1', 'X']
olsq = Estimate_ols(org_eq="Y = C(1) + C(2)*(X/X('2002Q1')) + C(3)*Z", input_df=dfq,
                    smpl=('2001Q2', '2005Q4'), ecm=False)
refq = Estimate_ols(org_eq='Y = C(1) + C(2)*(X/X_AT) + C(3)*Z', input_df=dfq,
                    smpl=('2001Q2', '2005Q4'), ecm=False)
print(olsq.coef_estimate_dict, '\n', refq.coef_estimate_dict)
assert close(list(olsq.coef_estimate_dict.values()), list(refq.coef_estimate_dict.values()))
print('ok')
