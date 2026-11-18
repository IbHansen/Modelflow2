# -*- coding: utf-8 -*-
"""Tests of modelestimator_r against a real R.

R lives in its own conda environment (conda-forge's Windows R can not share
book314b with cvxopt):

    conda create -n mfr -c conda-forge --override-channels r-base=4.5 r-sandwich r-lmtest r-ivreg r-minpack.lm

Point MODELFLOW_RSCRIPT at that environment (done below if it is not set).
Every result is checked against a Python reference: Estimate_ols, numpy 2SLS
and Estimate_nls_lmfit.
"""
#%% setup
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MODELFLOW_RSCRIPT", str(Path.home() / "miniforge3" / "envs" / "mfr"))

import modelestimator_r as mr
from modelestimator_new import Estimate_ols, Estimate_nls_lmfit
from modelestimator_r import Estimate_r, Estimate_r_lm, Estimate_r_nls

print("Rscript:", mr.find_rscript())

rng = np.random.default_rng(7)
years = list(range(1970, 2024))
n = len(years)
gov = 100 * np.exp(np.cumsum(0.02 + rng.normal(0, 0.03, n)))
inc = np.empty(n)
con = np.empty(n)
inc[0], con[0] = 100.0, 90.0
for t in range(1, n):
    # income growth driven by government growth plus a shock that also
    # hits consumption -> DLOG(INC) is endogenous in the CON equation
    common = rng.normal(0, 0.01)
    inc[t] = inc[t - 1] * np.exp(0.01 + 0.6 * np.log(gov[t] / gov[t - 1])
                                 + common + rng.normal(0, 0.005))
    dl = (0.01 + 0.7 * np.log(inc[t] / inc[t - 1])
          - 0.3 * (np.log(con[t - 1]) - np.log(inc[t - 1]) - np.log(0.9))
          + 0.8 * common + rng.normal(0, 0.004))
    con[t] = con[t - 1] * np.exp(dl)
df = pd.DataFrame({"CON": con, "INC": inc, "GOV": gov}, index=years)
SMPL = (1975, 2023)

EQ = "DLOG(CON) = C(1) + C(2)*DLOG(INC) + C(3)*(LOG(CON(-1)) - LOG(INC(-1)))"


def close(a, b, tol=1e-7):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    d = (a - b.reindex(a.index)).abs().max()
    assert d < tol, f"difference {d}\n{a}\n{b}"
    return d


#%% 1. lm matches Estimate_ols
ols = Estimate_ols(org_eq=EQ, input_df=df, smpl=SMPL)
rlm = Estimate_r_lm(org_eq=EQ, input_df=df, smpl=SMPL, method="lm")
print("lm coef diff", close(rlm.coef_ser, ols.coef_ser))
print("lm t diff   ", close(rlm.tvalues, ols.tvalues, 1e-6))
assert rlm.residual_check_max_diff < 1e-10
print(rlm.regression_model.diagnostics)

#%% 2. hac: same coefficients, Newey-West standard errors
hac = Estimate_r(EQ, input_df=df, smpl=SMPL, method="hac", hac_lag=2)
close(hac.coef_ser, ols.coef_ser)
import statsmodels.api as sm
X = ols.estimation_df.drop(columns="LHS")
ref = sm.OLS(ols.estimation_df["LHS"], X).fit(
    cov_type="HAC", cov_kwds={"maxlags": 2, "use_correction": False})
# sandwich adjust=TRUE scales by n/(n-k); statsmodels' small-sample
# correction off -> scale it the same way
k = X.shape[1]
nobs = len(X)
ref_se = pd.Series(ols._map_statsmodels_params(
    (ref.bse * np.sqrt(nobs / (nobs - k))).to_dict()))
print("hac se diff ", close(hac.regression_model.bse, ref_se, 1e-8))
hac_auto = Estimate_r(EQ, input_df=df, smpl=SMPL, method="hac")
close(hac_auto.coef_ser, ols.coef_ser)

#%% 3. ivreg vs numpy 2SLS, constant written as +C(1) and as -C(1)
INST = "DLOG(GOV) DLOG(GOV(-1)) LOG(CON(-1)) - LOG(INC(-1))"
INST_LIST = ["DLOG(GOV)", "DLOG(GOV(-1))", "LOG(CON(-1)) - LOG(INC(-1))"]


def tsls(est):
    """numpy 2SLS on the frame the estimator used; constant columns are
    instruments too."""
    d = est.estimation_df
    Xr = d[[c for c in est._regressors]].to_numpy()
    const = [c for c in est._regressors
             if np.all(d[c].to_numpy() == d[c].to_numpy()[0])]
    Z = np.column_stack([d[c].to_numpy() for c in const]
                        + [d[c].to_numpy() for c in est._instrument_columns])
    y = d["LHS"].to_numpy()
    Pz = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    b = np.linalg.solve(Xr.T @ Pz @ Xr, Xr.T @ Pz @ y)
    return pd.Series(b, index=[est._r_param_to_token(c) for c in est._regressors])


iv = Estimate_r(EQ, input_df=df, smpl=SMPL, method="ivreg", instruments=INST_LIST)
print(iv.regression_model.call)
print("ivreg diff  ", close(iv.coef_ser, tsls(iv)))
# one string: blanks separate, except next to an operator
split = mr._split_instruments
assert split(INST) == INST_LIST, split(INST)
assert split("DLOG(GOV), LOG(CON(-1)) -LOG(INC(-1)); X") == [
    "DLOG(GOV)", "LOG(CON(-1)) -LOG(INC(-1))", "X"]
assert split("a+ b c *d  e ** 2 f") == ["A+ B", "C *D", "E ** 2", "F"]
assert split("y c x @ z1 z2") == ["Y", "C", "X", "Z1", "Z2"]
assert split(["dlog(gov)", " x "]) == ["DLOG(GOV)", "X"]
iv_str = Estimate_r(EQ, input_df=df, smpl=SMPL, method="ivreg", instruments=INST)
close(iv_str.coef_ser, iv.coef_ser, 1e-12)
iv_ratio = Estimate_r(EQ, input_df=df, smpl=SMPL, method="ivreg", instruments=INST.replace(
    "LOG(CON(-1)) - LOG(INC(-1))", "LOG(CON(-1)/INC(-1))"))
close(iv_ratio.coef_ser, iv.coef_ser)
print("instrument string splitting ok")

EQ_NEG = "DLOG(CON) = -C(1) + C(2)*DLOG(INC) + C(3)*(LOG(CON(-1)) - LOG(INC(-1)))"
iv_neg = Estimate_r(EQ_NEG, input_df=df, smpl=SMPL, method="ivreg",
                    instruments=INST_LIST)
print(iv_neg.regression_model.call)
assert "`C__1`" in iv_neg.regression_model.call.split("|")[1], "constant not an instrument"
# same fit, opposite sign of C__1
close(iv_neg.coef_ser * pd.Series({"C__1": -1, "C__2": 1, "C__3": 1}), iv.coef_ser)
print("ivreg -C(1) ok; diagnostics:")
print(iv.regression_model.diagnostics)

#%% 4. nls / nlsLM vs lmfit, with ST. start values and a fixed parameter
EQ_NL = ("DLOG(CON) = C(1) + C(2)*DLOG(INC) - C(3)*(LOG(CON(-1)) - C(4)*LOG(INC(-1)))"
         " ST. C(3)~0.2; C(4)~1")
lm_ref = Estimate_nls_lmfit(org_eq=EQ_NL, input_df=df, smpl=SMPL)
for method in ("nls", "nlsLM"):
    e = Estimate_r(EQ_NL, input_df=df, smpl=SMPL, method=method)
    print(method, "diff", close(e.coef_ser, lm_ref.coef_ser, 1e-5),
          "converged", e.regression_model.stats.get("converged"))
    assert e.residual_check_max_diff < 1e-10

fixed = Estimate_r_nls(org_eq=EQ_NL.replace("C(4)~1", "C(4)=1"), input_df=df,
                       smpl=SMPL)
assert fixed.coef_ser["C__4"] == 1.0
bounded = Estimate_r_nls(org_eq=EQ_NL + "; C(3)=[0, 0.25]", input_df=df, smpl=SMPL)
assert 0 <= bounded.coef_ser["C__3"] <= 0.25 + 1e-12
print("fixed and bounded ok:", fixed.coef_ser.round(4).to_dict(),
      bounded.coef_ser.round(4).to_dict())

#%% 5. 'expr' is rejected also when it comes from default_params
for kw in ({"default_params": {"C__4": {"expr": "C__3"}}},
           {"constraints": {"C__4": {"expr": "C__3"}}}):
    try:
        Estimate_r_nls(org_eq=EQ_NL, input_df=df, smpl=SMPL, **kw)
    except ValueError as e:
        assert "not supported" in str(e)
    else:
        raise AssertionError(f"expr not rejected for {kw}")
print("expr rejected ok")

#%% 6. backend: auto means rscript on Windows; missing package message
assert rlm.regression_model.backend == ("rscript" if os.name == "nt" else rlm.regression_model.backend)
try:
    mr.run_r_script('.mf_need <- function(pkg) if (!requireNamespace(pkg, quietly=TRUE)) '
                    'stop(paste0("MODELFLOW_MISSING_PACKAGE:", pkg), call.=FALSE)\n'
                    '.mf_need("noSuchPackageX")', Path(os.environ["TEMP"]))
except mr.REstimationError as e:
    assert "install.packages('noSuchPackageX')" in str(e)
print("backend and error message ok")

#%% 7. Makemodel with <estimator=r_lm> and <estimator=r_nls>, and a factory
from modelconstruct_estimation import Makemodel
mm = Makemodel(f'''<estimator=r_lm,smpl=1975 2023> {EQ}
<estimator=ols,smpl=1975 2023> {EQ.replace("CON", "CON2")}''',
               input_df=df.assign(CON2=df.CON), type_input='modelflow')
print(mm.normal_frml)
rhac = Estimate_r.with_defaults(input_df=df, method="hac", hac_lag=2)
mm2 = Makemodel(f'''<estimator=rhac,smpl=1975 2023> {EQ}''',
                input_df=df, type_input='modelflow')
print(mm2.normal_frml)
mm3 = Makemodel(f'''<estimator=r_nls,smpl=1975 2023> {EQ_NL}''',
                input_df=df, type_input='modelflow')
print(mm3.normal_frml)
import re
c4 = float(re.search(r"-\(([0-9.]+)\)\*LOG\(INC\(-1\)\)", mm3.normal_frml).group(1))
assert abs(c4 - lm_ref.coef_ser["C__4"]) < 1e-5, c4
print("ALL OK")
