# -*- coding: utf-8 -*-
"""
Tests of recode written as if else in the generated code (modelpattern.inline_recode).

recode(c,a,b) was a function call, so both a and b were evaluated: recode(x>0, log(x), 0)
failed when x<=0.  The code writers now read allvar[v]['recodeterms'], where it is

    ( ( a ) if ( c ) else ( b ) )

so only the chosen branch is evaluated.  allvar[v]['terms'] is unchanged.

* The rewrite itself: parentheses, nested recode, recode in a function, lags, wrong arguments.
* terms is untouched, and recodeterms is the same list when the equation has no recode.
* A recode in funks= is the user's own: no rewrite (this is the behaviour from before).
* Guards: log, 1/x and sqrt of values outside their domain in the branch not chosen,
  in all solvers, also in the derivatives (forcenum=False). The reference is numpy by hand.
* The same guards fail with the user's own recode: the test can tell the difference.
* Where both branches are valid the results equal those with the user's own recode.
* PAK: the same solution as with the user's own recode.

Run the #%% cells in order.  Each cell prints what it checks; an assert stops
at the first wrong number.
"""
#%% setup
import contextlib
import io
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import modelclass
from modelclass import model
import modelpattern as pt
import modeluserfunk

print('modelclass   imported from', modelclass.__file__)
print('modelpattern imported from', pt.__file__)

try:
    HERE = Path(__file__).resolve().parent
except NameError:                       # run cell by cell
    HERE = Path.cwd()
DATA = HERE.parent / 'data'


def recode(condition, yes, no):
    ''' The user's own recode for funks=, an ordinary function: both yes and no are evaluated'''
    return yes if condition else no


def text(terms):
    ''' terms back to text, as the code writers do, with the variables as names'''
    return ''.join(t.op.lower() or t.number or (t.var + (f'({t.lag})' if t.lag else ''))
                   for t in terms)


def inlined(expression):
    return text(pt.inline_recode(pt.udtryk_parse(expression)))


def quiet(function, *args, **kwargs):
    ''' Calls function without its print. Returns (result, exception)'''
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return function(*args, **kwargs), None
        except Exception as e:
            return None, e


#%% the rewrite
cases = {
    'a = recode(x>0, y, z) $':
        'A=((Y) if (X>0) else (Z))$',
    # the outer pair keeps the meaning with an operator before and after
    'a = 2*recode(x>0, y, z)**2 $':
        'A=2*((Y) if (X>0) else (Z))**2$',
    'a = -recode(x>0, y, z) - recode(x>1, y, z) $':
        'A=-((Y) if (X>0) else (Z))-((Y) if (X>1) else (Z))$',
    # the pair around each argument: + binds weaker than if else
    'a = recode(x>0, y+1, z+2) $':
        'A=((Y+1) if (X>0) else (Z+2))$',
    # commas in a function inside recode, and lags
    'a = recode(x(-1)>0, max(y, z(+1)), min(y(-2), z)) $':
        'A=((max(Y,Z(+1))) if (X(-1)>0) else (min(Y(-2),Z)))$',
    # recode in recode, in all three places
    'a = recode(recode(d, x, y)>0, recode(x>1, 1, 2), recode(z==1, 3, b)) $':
        'A=((((1) if (X>1) else (2))) if (((X) if (D) else (Y))>0) else (((3) if (Z==1) else (B))))$',
    # recode in another function
    'a = log(max(recode(x>0, x, 1), y)) $':
        'A=log(max(((X) if (X>0) else (1)),Y))$',
    # no recode
    'a = max(x, y) + z(-1) $':
        'A=max(X,Y)+Z(-1)$',
}
for expression, expected in cases.items():
    result = inlined(expression)
    assert result == expected, f'\n{expression}\n{result}\n{expected}'
    print(f'{expression:75} {result}')

same = pt.udtryk_parse('a = max(x, y) $')
assert pt.inline_recode(same) is same, 'without recode the same list should be returned'

for wrong in ['a = recode(x>0, y) $', 'a = recode(x>0, y, z, 1) $', 'a = recode(x>0, , z) $']:
    _, error = quiet(inlined, wrong)
    assert error is not None and 'recode takes 3 arguments' in str(error), wrong
    _, error = quiet(model, 'frml <> ' + wrong)
    assert error is not None and 'recode takes 3 arguments' in str(error), wrong
print('Ok: the rewrite, and an exception with the wrong number of arguments')

#%% terms is untouched, recodeterms is the same list without recode, the user's own recode is not rewritten
ftwo = '''
frml <> a = recode(x > 0, log(x), 0) $
frml <> b = 2*a + x $
'''
m = model(ftwo)
assert text(m.allvar['A']['terms']) == 'A=recode(X>0,log(X),0)$'
assert text(m.allvar['A']['recodeterms']) == 'A=((log(X)) if (X>0) else (0))$'
assert m.allvar['B']['recodeterms'] is m.allvar['B']['terms']
assert m.allvar['A']['recodeterms'][m.allvar['A']['assigpos']] == m.aequalterm, 'assigpos should still point at ='

mown = model(ftwo, funks=[recode])
assert mown.allvar['A']['recodeterms'] is mown.allvar['A']['terms']
print("Ok: terms unchanged, recodeterms only differs with recode, a recode in funks= is not rewritten")

#%% guards: the branch not chosen is outside the domain of log, 1/x and sqrt
# W and V are simultaneous, so the Newton solvers differentiate recode(v>0, sqrt(v), 0) with
# respect to V: recode(v>0, 1/(2*sqrt(v)), 0), also guarded. Q only depends on the exogenous E.
fguard = '''
frml <> w = recode(v > 0, sqrt(v), 0) $
frml <> v = 0.5*w + x $
frml <> u = recode(v > 0, log(v), -99) $
frml <> r = recode(v != 0, 1/v, 0) + recode(v >= 0, v**0.5, -1) $
frml <> q = recode(e > 0, log(e), -99) + recode(e != 0, 1/e, 0) + recode(e >= 0, sqrt(e), -1) $
'''
years = range(2020, 2025)
dfguard = pd.DataFrame({'X': [1.0, -1.0, -0.5, 2.0, 6.0],
                        'E': [1.0, -1.0, 0.0, 2.0, 6.0]}, index=years)
for name in ['W', 'V']:
    dfguard[name] = 1.0
for name in ['U', 'R', 'Q']:
    dfguard[name] = 0.0

# by hand: x<=0 gives w=0 and v=x. x>0 gives w = sqrt(v) and v = 0.5*w+x, so w**2-0.5*w-x = 0
expected = pd.DataFrame(index=years[1:], columns=['W', 'V', 'U', 'R', 'Q'], dtype=float)
for year in years[1:]:
    x, e = dfguard.loc[year, ['X', 'E']]
    w = (0.5 + sqrt(0.25 + 4*x))/2 if x > 0 else 0.0
    v = 0.5*w + x
    expected.loc[year] = [w, v,
                          log(v) if v > 0 else -99,
                          (1/v if v != 0 else 0) + (v**0.5 if v >= 0 else -1),
                          (log(e) if e > 0 else -99) + (1/e if e != 0 else 0) + (sqrt(e) if e >= 0 else -1)]
print(expected.round(4))

runs = {
    'sim (legacy)':            dict(ng=False, solver='sim'),
    'sim (legacy) jit':        dict(ng=False, solver='sim', ljit=1, stringjit=True),
    'sim1d (legacy)':          dict(ng=False, solver='sim1d'),
    'sim_ng':                  dict(solver='sim_ng'),
    'sim_ng jit':              dict(solver='sim_ng', ljit=1, stringjit=True),
    'sim1d_ng':                dict(solver='sim1d_ng'),
    'newton_ng':               dict(solver='newton_ng', forcenum=False, nonlin=1),
    'newton_ng forcenum':      dict(solver='newton_ng', forcenum=True, nonlin=1),
    'newton_fbmin_ng':         dict(solver='newton_fbmin_ng', forcenum=False, nonlin=1),
    'newtonstack_ng':          dict(solver='newtonstack_ng', forcenum=False, nonlin=1),
    'newtonstack_ng jit':      dict(solver='newtonstack_ng', forcenum=False, nonlin=1, ljit=1, stringjit=True),
    'newtonstack_fbmin_ng':    dict(solver='newtonstack_fbmin_ng', forcenum=False, nonlin=1),
}


def solve_guard(funks=[], ng=True, **options):
    mguard = model(fguard, funks=funks, ng=ng)
    return mguard(dfguard, 2021, 2024, silent=1, max_iterations=500, **options)


for label, options in runs.items():
    result, error = quiet(solve_guard, **options)
    assert error is None, f'{label} raised {type(error).__name__}: {error}'
    worst = (result.loc[2021:2024, expected.columns] - expected).abs().max().max()
    print(f'{label:24} largest difference to the values by hand: {worst:.2e}')
    assert worst < 1e-6, label
print('Ok: only the chosen branch is evaluated, in the equations and in the derivatives')

#%% the same guards with the user's own recode fail: both branches are evaluated
for label in ['sim (legacy)', 'sim_ng', 'newton_ng']:
    result, error = quiet(solve_guard, funks=[recode], **runs[label])
    failed = error is not None or not np.isfinite(result.loc[2021:2024, expected.columns].to_numpy(dtype=float)).all()
    print(f'{label:24} with recode as a function: '
          + (f'{type(error).__name__}' if error is not None else 'nan in the result'))
    assert failed, label
print('Ok: as a function call recode fails on these equations')

#%% where both branches are valid: the same numbers as with the user's own recode
fvalid = '''
frml <> y1 = recode(a > b, a**2, b*c) + 2*recode(d, a, b)**2 $
frml <> y2 = -recode(a > 1, recode(b > 1, 1, 2), max(a, b)) + log(recode(c > a, c, a)) $
frml <> y3 = recode(y1(-1) > y2, exp(a - b), sqrt(c)) * recode(d == 1, 0.5*y3(-1), y1) $
'''
rng = np.random.default_rng(1)
dfvalid = pd.DataFrame(rng.uniform(0.5, 2.0, size=(12, 3)), index=range(2010, 2022), columns=['A', 'B', 'C'])
dfvalid['D'] = (rng.uniform(size=12) > 0.5).astype(float)       # a 0/1 variable as condition
for name in ['Y1', 'Y2', 'Y3']:
    dfvalid[name] = 1.0
shown = ['Y1', 'Y2', 'Y3']
for label, options, tolerance in [('sim_ng', dict(solver='sim_ng'), 0.0),
                                  ('sim (legacy)', dict(ng=False, solver='sim'), 0.0),
                                  ('sim_ng jit', dict(solver='sim_ng', ljit=1, stringjit=True), 1e-12)]:
    ng = options.pop('ng', True)
    new = model(fvalid, ng=ng)(dfvalid, 2011, 2021, silent=1, **options)
    # with jit the user's function has to be one numba knows: the jitted recode of modeluserfunk
    own = modeluserfunk.recode if options.get('ljit') else recode
    old = model(fvalid, ng=ng, funks=[own])(dfvalid, 2011, 2021, silent=1, **options)
    worst = (new.loc[2011:2021, shown] - old.loc[2011:2021, shown]).abs().max().max()
    print(f'{label:24} largest difference to recode as a function: {worst:.2e}')
    assert worst <= tolerance, label
    assert np.isfinite(new.loc[2011:2021, shown].to_numpy()).all()
print('Ok: the same results where both branches are valid')

#%% PAK: the same solution as with recode as a function
pakfile = DATA / 'pak.pcim'
if pakfile.exists():
    mpak, pakdf = model.modelload(str(pakfile), run=0)
    with_recode = [v for v in mpak.endogene if mpak.allvar[v]['recodeterms'] is not mpak.allvar[v]['terms']]
    print(f'PAK: {len(with_recode)} of {len(mpak.endogene)} equations have recode')
    start, end = mpak.current_per[0], mpak.current_per[-1]
    new = mpak(pakdf, start, end, solver='sim_ng', silent=1, reset_options=True)
    mold = model(mpak.equations, funks=[recode])
    old = mold(pakdf, start, end, solver='sim_ng', silent=1)
    endo = sorted(mpak.endogene)
    worst = (new.loc[start:end, endo] - old.loc[start:end, endo]).abs().max().max()
    print(f'PAK sim_ng {start}-{end}: largest difference to recode as a function: {worst:.2e}')
    assert worst == 0.0
    print('Ok: PAK is solved to the same numbers')
else:
    print(f'{pakfile} not found, PAK is not tested')

#%% done
print('\nAll tests of recode as if else passed')
