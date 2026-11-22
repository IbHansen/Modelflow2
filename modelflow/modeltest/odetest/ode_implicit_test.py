"""Implicit tableaux and simultaneous auxiliaries in modelode (experimental): each time step holds a simultaneous
block, solved with solver='newton_ng'. Checked against scipy solve_ivp (Radau, tight tolerance).
Run:  python ode_implicit_test.py"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the modelflow folder
from modelclass import model
from modelode import ode_transform, is_implicit, BUTCHER, OdeTransformError

# %% stiff scalar equation: dY/dt = -K*(Y - 1/(1+T)), K = 1000, Y(0) = 2
stiff = '''
FRML <ODE> Y = -K*(Y - 1/(1+T)) $
FRML <ODE> T = 1 $
'''
K, DT, N = 1000.0, 0.01, 100
ref = solve_ivp(lambda t, y: [-K * (y[0] - 1 / (1 + t))], (0, 1), [2.0], method='Radau', rtol=1e-12, atol=1e-12)
y_ref = ref.y[0, -1]

def run(method, solver='newton_ng'):
    m = model(ode_transform(stiff, method=method))
    df = pd.DataFrame(0.0, index=range(N + 1), columns=sorted(m.allvar))
    df['DT'], df['K'] = DT, K
    df.loc[0, 'Y'] = 2.0
    return m(df, 1, N, silent=1, **({'solver': solver} if solver else {})).loc[N, 'Y']

print(f'reference Y(1) = {y_ref:.10f}')
for method in ['euler', 'rk4', 'backward_euler', 'trapezoid', 'implicit_midpoint', 'radau3']:
    assert is_implicit(BUTCHER[method]) == (method not in ('euler', 'rk4'))
    y = run(method) if is_implicit(BUTCHER[method]) else run(method, solver=None)
    print(f'{method:18} Y(1) = {y: .10g}   error {abs(y - y_ref):.1e}')
    if is_implicit(BUTCHER[method]):
        assert abs(y - y_ref) < 1e-5, method

# %% simultaneous auxiliaries: dX/dt = -P*X with P and Q determined together
simaux = '''
FRML <ODE> X = -P*X $
FRML <> P = 0.5 + 0.1*Q $
FRML <> Q = X + 0.2*P $
'''
# P = 0.5 + 0.1*(X + 0.2*P)  ->  P = (0.5 + 0.1*X)/0.98
ref2 = solve_ivp(lambda t, x: [-(0.5 + 0.1 * x[0]) / 0.98 * x[0]], (0, 5), [1.0], method='DOP853',
                 rtol=1e-12, atol=1e-12)
for method, solver in [('rk4', 'newton_ng'), ('heun', 'newton_ng'), ('rk4', None)]:
    m = model(ode_transform(simaux, method=method))
    df = pd.DataFrame(0.0, index=range(501), columns=sorted(m.allvar))
    df['DT'] = 0.01
    df.loc[0, 'X'] = 1.0
    x = m(df, 1, 500, silent=1, **({'solver': solver} if solver else {})).loc[500, 'X']
    print(f'simultaneous aux, {method}, solver {solver or "default"}: X(5) = {x:.10f}  error {abs(x - ref2.y[0, -1]):.1e}')
    assert abs(x - ref2.y[0, -1]) < (1e-8 if method == 'rk4' else 1e-5)

# %% a tableau of the wrong shape and an unknown method are refused
for bad in [dict(a=[[1]], b=[0.5, 0.5], c=[0, 1]), 'rk5']:
    try:
        ode_transform(stiff, method=bad)
        raise AssertionError('not refused')
    except OdeTransformError as e:
        print('refused:', e)
print('ok')
