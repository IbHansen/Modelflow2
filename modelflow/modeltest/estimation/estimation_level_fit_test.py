"""
Actual, fitted and residuals of the endogenous variable itself (af_level_df,
Makemodel.actual_fitted, the extra plot in the report) when the left-hand side
is a transformation such as DLOG(CON).

Run the #%% cells in order. An assert stops at the first difference. The last cell
writes the reports to the folder html next to this file.
"""
#%% setup
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

try:
    HERE = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    HERE = Path(r'C:\modelflow2\modelflow\modeltest\estimation')

from modelconstruct_estimation import Makemodel
from modelestimator_new import plot_actual_vs_fitted

rng = np.random.default_rng(1)
years = list(range(1990, 2024))
y = 100 * np.exp(np.cumsum(0.02 + 0.01 * rng.standard_normal(len(years))))
con = np.empty(len(years))
con[0] = 60
for i in range(1, len(years)):
    con[i] = con[i - 1] * np.exp(0.6 * np.log(y[i] / y[i - 1])
                                 - 0.2 * (np.log(con[i - 1]) - np.log(0.6 * y[i - 1]))
                                 + 0.005 * rng.standard_normal())
df = pd.DataFrame({'Y': y, 'CON': con, 'INV': 0.2 * y + rng.standard_normal(len(years))}, index=years)

text = '''
FRML <est=nls_lmfit, stoc, smpl=1995 2020> dlog(con) = c(1)*dlog(y) + c(2)*(log(con(-1)) - log(y(-1))) + c(3) $
FRML <est=ols, stoc, smpl=1995 2020> inv = c(1) + c(2)*y $
'''
mm = Makemodel(text, eq_start='FRML', eq_end='$', input_df=df)
est_con, est_inv = [rec['estimator_object'] for rec in mm.estimation_records]

#%% 1. the frame of one equation
lev_prefixed = est_con.af_level_df
print(lev_prefixed.loc[1995:1999].round(4))
assert est_con.lhs_is_transformed and not est_inv.lhs_is_transformed
assert list(lev_prefixed.columns) == ['CON__Actual', 'CON__Fitted', 'CON__Residuals']
lev = lev_prefixed.rename(columns=lambda c: c.removeprefix('CON__'))
assert lev.index[0] == 1995 and lev.index[-1] == 2020
assert np.array_equal(lev['Actual'].to_numpy(), df.loc[1995:2020, 'CON'].to_numpy())
# the level fit is the fit of dlog(con) put on top of the actual lag
by_hand = df['CON'].shift(1).loc[1995:2020] * np.exp(est_con.af_df['Fitted'])
assert np.allclose(lev['Fitted'], by_hand, rtol=1e-8), abs(lev['Fitted'] - by_hand).max()
assert np.allclose(lev['Residuals'], lev['Actual'] - lev['Fitted'])
# log(actual/fitted) is the residual of the estimated equation
assert np.allclose(np.log(lev['Actual'] / lev['Fitted']), est_con.residuals_df['Residuals'], atol=1e-8)
# a left-hand side which is the variable: the same numbers as af_df / residuals_df
assert est_inv.af_level_df.equals(est_inv.af_df.join(est_inv.residuals_df).add_prefix('INV__'))
print('ok')

#%% 2. the frame of the whole model
both = mm.actual_fitted()
assert list(both.columns.get_level_values(0).unique()) == ['CON', 'INV']
assert both['CON'].equals(lev)
lhs = mm.actual_fitted(level=False)
assert np.allclose(lhs['CON']['Fitted'], est_con.af_df['Fitted'])
assert abs(lhs['CON']['Actual']).max() < 1 < both['CON']['Actual'].min()      # dlog against level
print(both.loc[2018:2020].round(3))
print('ok')

#%% 3. the plot and the report
from io import BytesIO
buf = BytesIO()
plot_actual_vs_fitted(est_con, level=True, save_to=buf, format='png')
assert buf.getbuffer().nbytes > 1000
html_con = est_con.get_html_report(plot_format='png')
html_inv = est_inv.get_html_report(plot_format='png')
assert html_con.count('<img') == 2 and 'CON itself (one-step fit)' in html_con
assert html_inv.count('<img') == 1 and 'itself' not in html_inv
assert mm.report(plot_format='png')._repr_html_().count('CON itself (one-step fit)') == 1      # the model report
print('ok')

#%% 4. the html files to look at
print(mm.report(title='Level fit test').save(path=str(HERE / 'html')))            # the model report
mm.estimation_report(path=str(HERE / 'html'), filename='estimation_report.html')  # the estimation summary
print(HERE / 'html' / 'estimation_report.html')
