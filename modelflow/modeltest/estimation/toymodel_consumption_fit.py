"""
The consumption equation of the toy model (the last one in the consumption segment
of C:\\estimator\\mf-next generation\\ToyModel estimation.ipynb, lambda fixed at 0.3)
estimated on the Nepal data, and the fitted values extracted. The data is in the
script, so it runs on its own.

Run the #%% cells in order.
"""
#%% setup
from functools import partial
from io import StringIO

import pandas as pd

from modelconstruct_estimation import Makemodel
from modelestimator_new import Estimate_nls

#%% data, the same as the %%dataframe cell in the notebook
data = """
year        Y     CON    GOV     INV     EXP     IMP
2010  1507633     nan    nan  376664     nan     nan
2011  1559222 1320302 127814  373939  121715  444232
2012  1632040 1360376 128860  381170  146293  457746
2013  1689572 1396402 124255  414557  165235  522448
2014  1791141 1438904 138533  466473  194706  632208
2015  1862357 1476067 154467  536417  199215  692795
2016  1870424 1537410 135991  570679  164739  714626
2017  2038337 1549515 165119  702408  179327  916470
2018  2193706 1645118 168507  785371  193125 1090956
2019  2339743 1779022 184955  874481  203831 1154398
2020  2284300 1842997 192011  796389  171458  913728
2021  2394818 1989827 188832  874412  134904 1085637
2022  2529243 2125755 207023  907774  180852 1249408
2023  2576251 2144255 202054  809228  190884 1034756
2024  2737770 2187140 213861  847178  223748  998743
2025  2873409 2311319 229992  896806  258987 1088281
2026  3044418 2469903 247718  970059  284928 1192775
2027  3247138 2644779 265841 1060412  309374 1297854
2028  3469278 2825068 283920 1161461  332639 1398396
2029  3692901 2999393 301261 1265387  355073 1492799
2030  3905964 3161294 317423 1366512  376928 1580778
"""
npl = pd.read_csv(StringIO(data), sep=r'\s+', index_col='year').astype(float)
npl.index.name = None
npl['YDISC'] = npl['Y'] - (npl['CON'] + npl['INV'] + npl['GOV'] + npl['EXP'] - npl['IMP'])
npl['GDE'] = npl['CON'] + npl['GOV'] + npl['INV']

var_description = {
    'Y': 'GDP',
    'CON': 'Private Consumption',
}

ls = Estimate_nls.with_defaults(
    smpl=(2011, 2019),
    var_description=var_description,
    input_df=npl,
    param_names=['lambda', 'beta_shortterm', 'beta_longterm'],
)

makeeq = partial(
    Makemodel,
    estimator_classes={'ls': ls},
    var_description=var_description,
    type_input='single',
)

#%% estimate the consumption equation
fcon = """<stoc,fit,est=ls>
dlog(con)= -lambda*(log(con(-1))-log(y(-1))-log(abs(beta_longterm(1))))+
beta_shortterm(10)*dlog(y) st. lambda=0.3"""

mcon = makeeq(fcon)
# mcon.shownormal_frml
# mcon.shownormal_fit
est = mcon.estimation_records[0]['estimator_object']

#%% fitted values in left-hand-side units, dlog(con), estimation sample
print(est.af_df)
print(est.residuals_df)

#%% fitted values of CON itself (one-step fit), estimation sample
print(est.af_level_df)

#%% fitted values of CON for the whole data period, from the model
mfit = mcon.mmodel
base = mfit.insertModelVar(npl)
res = mfit(base, 2012, 2030, solver='res_ng')
fitted = pd.DataFrame({'CON__Actual': npl.loc[2012:2030, 'CON'],
                       'CON__Fitted': res.loc[2012:2030, 'CON_FITTED']})
fitted['CON__Residuals'] = fitted['CON__Actual'] - fitted['CON__Fitted']
print(fitted)
