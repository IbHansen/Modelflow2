# -*- coding: utf-8 -*-
"""
Tests of the derivatives of functions sympy does not know (modelnewton.Diff_function).

* An equation with MAX, MIN, ABS, LOGIT, LOGIT_INVERSE, ERF, NORMCDF, RECODE, SQRT, TANH
  or SUM_EXCEL gets an expression as derivative, not a numerical difference.
* The values equal the derivatives worked out by hand in numpy.
* Close to a kink the expression is exact, where the numerical difference is wrong.
* At a tie max and min take the first argument, abs gives 1 at zero.
* A user function (funks=) gets its derivative from a subclass of Diff_function;
  without a subclass, or with None as partial, the numerical derivative is used.
* The key of the stored expressions (diff_expressions) follows the functions used.
* A model with these functions solved by Newton gives the same as Gauss-Seidel.

The references are computed by hand in numpy, not with modelflow.

Run the #%% cells in order.  Each cell prints what it checks; an assert stops
at the first wrong number.
"""
#%% setup
import numpy as np
import pandas as pd
from math import erf, exp, sqrt, tanh, pi, log

import modelclass
from modelclass import model
import modelnewton
from modelnewton import newton_diff, Diff_function

print('modelclass  imported from', modelclass.__file__)
print('modelnewton imported from', modelnewton.__file__)
print('Functions with a derivative:', ' '.join(sorted(Diff_function.registry)))


def close(a, b, tol=1e-12):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tol, rtol=tol)


def is_numerical(expression):
    ''' A numerical derivative is (f(x+0.0025)-f(x-0.0025))/0.005 '''
    return expression.endswith('/0.005') and '0.0025' in expression


def derivatives(fmodel, df, funks=[], **kwargs):
    ''' Returns the newton_diff instance and the values as {(var, pvar, lag, date): value}'''
    m = model(fmodel, funks=funks)
    m.lastdf = df
    m.current_per = df.index[1:]
    nd = newton_diff(m, df=df, **kwargs)
    melted = nd.get_diff_melted_var(df=df)
    values = {(r.var, r.pvar, r.lag, r.dates): r.value for r in melted.itertuples()}
    return nd, values


def phi(z):
    ''' the standard normal density '''
    return exp(-z*z/2)/sqrt(2*pi)


def sigmoid(u):
    return 1/(1+exp(-u))


# A, B and C are endogenous, and the Y variables are the functions tested.
# The data makes the branches of max, min, abs and recode differ between the years.
ftest = '''
frml <> a      = 0.5*b(-1) + 0.1*c $
frml <> b      = 0.3*a + 0.1 $
frml <> c      = 0.2*a + 0.1*b $
frml <> ymax   = max(a, 2*b) $
frml <> ymin   = min(a*c, b) $
frml <> ymax3  = max(a, b, c) $
frml <> ymin3  = min(a, b, c) $
frml <> yabs   = abs(a - b) $
frml <> ylogit = logit(a*b) $
frml <> yinv   = logit_inverse(a - 2*c) $
frml <> yerf   = erf(a*b) $
frml <> ynorm  = normcdf(a, b, c) $
frml <> ynorm1 = normcdf(a*b) $
frml <> ysqrt  = sqrt(a + b*b) $
frml <> ytanh  = tanh(a*c) $
frml <> ysum   = sum_excel(a, b*b, c) $
frml <> ymix   = recode(a > b, a**2, b*c) + max(a, c)*b $
frml <> ylag   = max(a(-1), b) + min(b(-1), c)*a $
'''
years = [2020, 2021, 2022, 2023]
df = pd.DataFrame({'A': [0.40, 0.50, 0.20, 0.70],
                   'B': [0.30, 0.20, 0.60, 0.45],
                   'C': [0.50, 0.70, 0.40, 0.10]}, index=years)
for name in 'YMAX YMIN YMAX3 YMIN3 YABS YLOGIT YINV YERF YNORM YNORM1 YSQRT YTANH YSUM YMIX YLAG'.split():
    df[name] = 0.0

#%% The derivatives are expressions, not numerical differences
nd, values = derivatives(ftest, df)
for var, d in nd.diffendocur.items():
    for pvar, expression in d.items():
        print(f'd{var}/d{pvar} = {expression}')
        assert not is_numerical(expression), f'd{var}/d{pvar} is a numerical difference'
        assert 'DERIVATIVE' not in expression.upper()
nd_num, values_num = derivatives(ftest, df, forcenum=True)
assert all(is_numerical(e) for d in nd_num.diffendocur.values() for e in d.values())
print('\nOk: all are expressions, and forcenum=True still gives numerical differences')

#%% The values equal the derivatives worked out by hand
def reference(date):
    ''' The derivatives by hand: {(var, pvar, lag): value} '''
    a, b, c = df.loc[date, ['A', 'B', 'C']]
    al, bl = df.loc[date-1, ['A', 'B']]
    r = {}
    r['YMAX', 'A', 0], r['YMAX', 'B', 0] = (1, 0) if a >= 2*b else (0, 2)
    r['YMIN', 'A', 0], r['YMIN', 'B', 0], r['YMIN', 'C', 0] = (c, 0, a) if a*c <= b else (0, 1, 0)
    largest, smallest = max(a, b, c), min(a, b, c)
    for name, value in zip('ABC', (a, b, c)):
        r['YMAX3', name, 0] = 1.0 if value == largest else 0.0
        r['YMIN3', name, 0] = 1.0 if value == smallest else 0.0
    sign = 1 if a - b >= 0 else -1
    r['YABS', 'A', 0], r['YABS', 'B', 0] = sign, -sign
    u = a*b
    r['YLOGIT', 'A', 0], r['YLOGIT', 'B', 0] = b/(u*(1-u)), a/(u*(1-u))
    s = sigmoid(a - 2*c)
    r['YINV', 'A', 0], r['YINV', 'C', 0] = s*(1-s), -2*s*(1-s)
    d_erf = 2/sqrt(pi)*exp(-u*u)
    r['YERF', 'A', 0], r['YERF', 'B', 0] = b*d_erf, a*d_erf
    z = (a-b)/c
    r['YNORM', 'A', 0], r['YNORM', 'B', 0], r['YNORM', 'C', 0] = phi(z)/c, -phi(z)/c, -z*phi(z)/c
    r['YNORM1', 'A', 0], r['YNORM1', 'B', 0] = b*phi(u), a*phi(u)
    root = sqrt(a + b*b)
    r['YSQRT', 'A', 0], r['YSQRT', 'B', 0] = 1/(2*root), b/root
    d_tanh = 1 - tanh(a*c)**2
    r['YTANH', 'A', 0], r['YTANH', 'C', 0] = c*d_tanh, a*d_tanh
    r['YSUM', 'A', 0], r['YSUM', 'B', 0], r['YSUM', 'C', 0] = 1, 2*b, 1
    r['YMIX', 'A', 0] = (2*a if a > b else 0) + (b if a >= c else 0)
    r['YMIX', 'B', 0] = (0 if a > b else c) + max(a, c)
    r['YMIX', 'C', 0] = (0 if a > b else b) + (0 if a >= c else b)
    r['YLAG', 'A', -1] = 1 if al >= b else 0
    r['YLAG', 'B', 0] = 0 if al >= b else 1
    r['YLAG', 'B', -1] = a if bl <= c else 0
    r['YLAG', 'C', 0] = 0 if bl <= c else a
    r['YLAG', 'A', 0] = min(bl, c)
    return r


checked = 0
for date in years[1:]:
    for (var, pvar, lag), expected in reference(date).items():
        got = values[var, pvar, lag, date]
        assert close(got, expected), f'd{var}/d{pvar}({lag}) in {date}: {got} expected {expected}'
        checked += 1
tested = {var for var, *_ in values if var.startswith('Y')}
assert tested == {var for var, *_ in reference(years[1])}, 'a test equation has no reference'
print(f'Ok: {checked} derivatives equal the hand-made ones (1e-12)')

worst = max(abs(values[key] - values_num[key]) for key in values)
print(f'Largest difference to the numerical derivatives: {worst:.2e} (the error of the numerical difference)')
assert worst < 1e-3       # only a sanity check: logit is curved enough to give 2.6e-4 with a step of 0.005

#%% The derivative model evaluated with ng and with jit gives the same
for kwargs in [dict(ng=1), dict(ljit=1)]:
    _, values_other = derivatives(ftest, df, **kwargs)
    assert values_other.keys() == values.keys()
    assert all(close(values_other[key], values[key]) for key in values), kwargs
    print(f'Ok: {kwargs} gives the same values')

#%% Close to a kink the expression is exact, the numerical difference is not
fkink = '''
frml <> a    = 0.5*b $
frml <> b    = 0.3*a + 0.1 $
frml <> ymax = max(a, b) $
frml <> yabs = abs(a - b) $
frml <> yrec = recode(a > b, 3*a, b) $
'''
dfkink = pd.DataFrame({'A': [0.5, 0.501], 'B': [0.5, 0.5], 'YMAX': 0.0, 'YABS': 0.0, 'YREC': 0.0},
                      index=[2020, 2021])        # A is 0.001 over B: inside the step of 0.0025
_, vkink = derivatives(fkink, dfkink)
_, vkink_num = derivatives(fkink, dfkink, forcenum=True)
expected = {('YMAX', 'A'): 1, ('YMAX', 'B'): 0, ('YABS', 'A'): 1, ('YABS', 'B'): -1,
            ('YREC', 'A'): 3, ('YREC', 'B'): 0}
for (var, pvar), value in expected.items():
    got, got_num = vkink[var, pvar, 0, 2021], vkink_num[var, pvar, 0, 2021]
    print(f'd{var}/d{pvar}: expression {got:8.3f}   numerical {got_num:8.3f}   true {value}')
    assert close(got, value)
assert not close(vkink_num['YMAX', 'A', 0, 2021], 1, tol=0.01), 'the numerical derivative was expected to be wrong here'
print('Ok: exact inside the step of the numerical difference')

#%% At a tie the first argument is taken, and abs gives 1 at zero
dftie = dfkink.copy()
dftie.loc[2021, 'A'] = 0.5
_, vtie = derivatives(fkink, dftie)
assert close([vtie['YMAX', 'A', 0, 2021], vtie['YMAX', 'B', 0, 2021]], [1, 0])
assert close([vtie['YABS', 'A', 0, 2021], vtie['YABS', 'B', 0, 2021]], [1, -1])
print('Ok: max(a,b) with a == b has the derivative of a, abs(0) has 1')

#%% Conditions sympy can decide itself
fdecided = '''
frml <> a  = 0.5*b $
frml <> b  = 0.3*a + 0.1 $
frml <> y1 = max(2, 3)*a + min(2, 3)*b $
frml <> y2 = max(a, a) + abs(3)*b $
frml <> y3 = max(a, 0) + min(b, 1) $
'''
dfdecided = pd.DataFrame({'A': [0.5, 0.4], 'B': [0.5, 0.3], 'Y1': 0.0, 'Y2': 0.0, 'Y3': 0.0}, index=[2020, 2021])
nd_decided, vdecided = derivatives(fdecided, dfdecided)
for var in ['Y1', 'Y2', 'Y3']:
    print(var, nd_decided.diffendocur[var])
    assert not any(is_numerical(e) for e in nd_decided.diffendocur[var].values())
assert close([vdecided['Y1', 'A', 0, 2021], vdecided['Y1', 'B', 0, 2021]], [3, 2])
assert close([vdecided['Y2', 'A', 0, 2021], vdecided['Y2', 'B', 0, 2021]], [1, 3])
assert close([vdecided['Y3', 'A', 0, 2021], vdecided['Y3', 'B', 0, 2021]], [1, 1])
print('Ok: no True/False ends up in the expressions')

#%% A user function gets its derivative from a subclass of Diff_function
def cube(x):
    return x**3


def scaled(x, k):
    return k*x*x


def plain(x):
    return x*x


class CUBE(Diff_function):
    @classmethod
    def partials(cls, u):
        return [3*u**2]


class SCALED(Diff_function):
    ''' Only the derivative with respect to the first argument is given '''
    @classmethod
    def partials(cls, x, k):
        return [2*k*x, None]


fuser = '''
frml <> a  = 0.5*b $
frml <> b  = 0.3*a + 0.1 $
frml <> y1 = cube(a*b) $
frml <> y2 = scaled(a, 4) $
frml <> y3 = scaled(a, b) $
frml <> y4 = plain(a) $
'''
dfuser = pd.DataFrame({'A': [0.5, 0.4], 'B': [0.5, 0.3], 'Y1': 0.0, 'Y2': 0.0, 'Y3': 0.0, 'Y4': 0.0},
                      index=[2020, 2021])
try:
    nd_user, vuser = derivatives(fuser, dfuser, funks=[cube, scaled, plain])
finally:
    for name in ['CUBE', 'SCALED']:            # the registry is shared: the test functions are taken out again
        Diff_function.registry.pop(name, None)
for var in ['Y1', 'Y2', 'Y3', 'Y4']:
    print(var, nd_user.diffendocur[var])
a, b = 0.4, 0.3
assert not any(is_numerical(e) for e in nd_user.diffendocur['Y1'].values())
assert close([vuser['Y1', 'A', 0, 2021], vuser['Y1', 'B', 0, 2021]], [3*(a*b)**2*b, 3*(a*b)**2*a])
assert not is_numerical(nd_user.diffendocur['Y2']['A'])
assert close(vuser['Y2', 'A', 0, 2021], 8*a)
assert is_numerical(nd_user.diffendocur['Y3']['B']), 'None as partial should give the numerical derivative'
assert close(vuser['Y3', 'B', 0, 2021], a*a, tol=1e-6)
assert is_numerical(nd_user.diffendocur['Y4']['A']), 'a function without a subclass should give the numerical derivative'
assert close(vuser['Y4', 'A', 0, 2021], 2*a, tol=1e-6)
print('Ok: subclass used, None and no subclass fall back to the numerical derivative')

#%% The key of the stored expressions follows the functions used
assert Diff_function.used_in('A = XMAX(-1) + LOGIT_INVERSE(B) + RELOGIT(-1)') == ['LOGIT_INVERSE']
assert Diff_function.used_in('A = MAX(B,C)+RECODE(B>C,1,2)') == ['MAX', 'RECODE']
assert Diff_function.used_in('A = B + C(-1)') == []


def diff_key(fmodel, registry=None):
    ''' The key of the stored expressions, optionally with another set of registered functions '''
    saved = dict(Diff_function.registry)
    try:
        if registry is not None:
            Diff_function.registry.clear()
            Diff_function.registry.update(registry)
        m = model(fmodel)
        dfkey = pd.DataFrame(1.0, index=[2020, 2021], columns=sorted(m.allvar))
        m.lastdf = dfkey
        return newton_diff(m, df=dfkey).diff_key
    finally:
        Diff_function.registry.clear()
        Diff_function.registry.update(saved)


fmax = 'frml <> a = max(b, 1) $ frml <> b = 0.5*a $'
frec = 'frml <> a = recode(b > 1, b, 1) $ frml <> b = 0.5*a $'
only_recode = {'RECODE': Diff_function.registry['RECODE']}
assert diff_key(fmax) != diff_key(fmax, only_recode), 'expressions stored before MAX had a derivative would be reused'
assert diff_key(frec) == diff_key(frec, only_recode), 'a model with only RECODE should keep its stored expressions'
print('Ok: a function which gets a derivative gives a new key, a model with only RECODE keeps its key')

#%% The stored expressions are reused
m = model(fmax)
dfkey = pd.DataFrame(1.0, index=[2020, 2021], columns=['A', 'B'])
m.lastdf = dfkey
first = newton_diff(m, df=dfkey)
second = newton_diff(m, df=dfkey)
assert second.diffendocur is first.diffendocur
print('Ok: the second newton_diff takes the expressions from model.diff_expressions')

#%% Newton solves a model with these functions as Gauss-Seidel does
fsolve = '''
frml <> c    = 0.6*max(y - t, 0.2) + 0.1*abs(y(-1) - 1) $
frml <> t    = 0.3*y*logit_inverse(y - 1) $
frml <> i    = min(0.2*y, 0.35) + 0.05*tanh(c) + 0.02*normcdf(y, 1, 0.5) $
frml <> y    = c + i + g $
frml <> risk = recode(y > 1.3, sqrt(y), logit(0.5*t + 0.2)) $
'''
dfsolve = pd.DataFrame({'G': [0.2, 0.25, 0.4, 0.6, 0.3, 0.1]}, index=range(2020, 2026))
for name in ['C', 'T', 'I', 'RISK']:
    dfsolve[name] = 0.5
dfsolve['Y'] = 1.0
msolve = model(fsolve)
gauss = msolve(dfsolve, 2021, 2025, solver='sim_ng', silent=1, max_iterations=500, relconv=1e-12)
shown = ['C', 'T', 'I', 'Y', 'RISK']
print(gauss.loc[2021:2025, shown].round(4))
assert (gauss.loc[2021:2025, 'Y'] > 1.3).any() and (gauss.loc[2021:2025, 'Y'] <= 1.3).any(), 'both branches of recode should be used'
for solver in ['newton_ng', 'newtonstack_ng']:
    # nonlin=1: the Jacobian is found again in every iteration, as the kinks change it
    newton = model(fsolve)(dfsolve, 2021, 2025, solver=solver, silent=1, forcenum=False, nonlin=1)
    worst = (newton.loc[2021:2025, shown] - gauss.loc[2021:2025, shown]).abs().max().max()
    print(f'{solver:16} largest difference to sim_ng: {worst:.2e}')
    assert worst < 1e-6, solver
print('Ok: the same solution')

#%% done
print('\nAll tests of the derivatives of functions passed')
