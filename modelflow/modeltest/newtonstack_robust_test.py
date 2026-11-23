# -*- coding: utf-8 -*-
"""Smoke test of the robustness options of newtonstack_ng (branch faster-newton, design note
experiments/factorize_update/newtonstack_robust_design.md):

    newton_linesearch, newton_stall_refresh, newton_raise, model.ng_solve_status, the stop on a
    non-finite residual and the exceptions SolveError / NotConvergedError / DivergedError / EquationError.

Run from the modelflow folder:  python modeltest/newtonstack_robust_test.py
The default path is checked bit-identical against the old iterate separately (regression on PAK, FRB/US).
"""
#%% setup
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modelclass import model                                         # noqa: E402
from modelsolver_ng import (SolveError, NotConvergedError, DivergedError,  # noqa: E402
                            EquationError)

INDEX = range(2000, 2011)
BASE = dict(solver='newtonstack_ng', silent=1, reset_options=True, ljit=False)
results = {}


def check(name, ok, detail=''):
    results[name] = ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:55} {detail}", flush=True)


def cubic():
    """y**3 = X, solved from y = 1. With the Jacobian of the start kept (no nonlin) the chord steps
    overshoot and grow."""
    m = model('FRML <> Y___RES = Y**3 - X $')
    df = pd.DataFrame({'Y': 1.0, 'X': 8.0, 'Y___RES': 0.0}, index=INDEX)
    return m, df


#%% the default: chord on the Jacobian of the start diverges (as before)
m, df = cubic()
r = m(df, 2001, 2010, **BASE, max_iterations=30)
st = m.ng_solve_status
check('default: chord diverges, stops at the non-finite residual', not st['converged']
      and st['reason'] == 'non-finite residual' and st['iterations'] < 30, str(st))

#%% newton_raise: NotConvergedError or DivergedError with the result attached
m, df = cubic()
try:
    m(df, 2001, 2010, **BASE, max_iterations=30, newton_raise=True)
    check('newton_raise raises', False, 'no exception')
except (NotConvergedError, DivergedError) as e:
    check('newton_raise raises', isinstance(e, SolveError) and isinstance(e, ArithmeticError)
          and e.result is not None and e.status['converged'] is False, f'{type(e).__name__}: {e}')

#%% line search alone: no step makes max |residual| grow, but on the Jacobian of the start (3, the
# derivative at the solution is 12) the progress is slow -- the cure is a new Jacobian (next cells)
m, df = cubic()
r = m(df, 2001, 2010, **BASE, max_iterations=30, newton_linesearch=True)
st = m.ng_solve_status
check('line search alone: no divergence', st['reason'] in ('converged', 'max_iterations')
      and st['max_residual'] < 7.0, f"{st}")

#%% stall refresh, no line search: the first step is on a Jacobian from the start, so it is taken as
# before; later steps on an older Jacobian that lower max |residual| by less than half go back and
# renew the Jacobian there (with the default ratio 1.0 only a growing step does, and the slow chord
# steps after the first refresh all fall a little)
m, df = cubic()
r = m(df, 2001, 2010, **BASE, max_iterations=30, newton_stall_refresh=10, newton_stall_ratio=0.5)
st = m.ng_solve_status
check('stall refresh converges', st['converged'] and np.allclose(r.loc[2001:2010, 'Y'], 2.0),
      f"{st['iterations']} iterations, {st['refreshes']} refreshes")

#%% line search + stall refresh: the first step is halved (Jacobian from that point), the next growing
# step renews the Jacobian
m, df = cubic()
r = m(df, 2001, 2010, **BASE, max_iterations=30, newton_stall_refresh=10, newton_linesearch=True)
st = m.ng_solve_status
check('line search + stall refresh converge', st['converged'] and np.allclose(r.loc[2001:2010, 'Y'], 2.0),
      f"{st['iterations']} iterations, {st['refreshes']} refreshes, {st['linesearch_steps']} halvings")

#%% a stored Jacobian from an earlier solve: the second solve renews it only when it stops working
m, df = cubic()
m(df.assign(X=1.0), 2001, 2010, **BASE)                 # solves at y = 1: the stored J is 3
r = m(df.assign(X=27.0), 2001, 2010, **BASE, max_iterations=30, newton_stall_refresh=3,
      newton_linesearch=True)
st = m.ng_solve_status
check('stored Jacobian + line search + refresh', st['converged'] and st['refreshes'] >= 1
      and np.allclose(r.loc[2001:2010, 'Y'], 3.0), str(st))

#%% the same without the options: the stored Jacobian is kept (as before)
m, df = cubic()
m(df.assign(X=1.0), 2001, 2010, **BASE)
r = m(df.assign(X=27.0), 2001, 2010, **BASE, max_iterations=30)
st = m.ng_solve_status
check('stored Jacobian, no options: not converged', not st['converged'], str(st))

#%% an equation that raises in the Newton loop: EquationError with newton_raise, the original
# exception without. Exact derivatives (forcenum=False): the numerical ones would raise in prepare
m = model('FRML <> Y___RES = LOG(Y) - X $')
df = pd.DataFrame({'Y': -1.0, 'X': 0.0, 'Y___RES': 0.0}, index=INDEX)
try:
    m(df, 2001, 2010, **BASE, forcenum=False)
    check('LOG(-1) raises the original exception', False, 'no exception')
except SolveError as e:
    check('LOG(-1) raises the original exception', False, f'got {type(e).__name__}')
except Exception as e:
    check('LOG(-1) raises the original exception', True, f'{type(e).__name__}: {e}')
try:
    m(df, 2001, 2010, **BASE, forcenum=False, newton_raise=True)
    check('LOG(-1) with newton_raise: EquationError', False, 'no exception')
except EquationError as e:
    check('LOG(-1) with newton_raise: EquationError', isinstance(e, ValueError)
          and e.__cause__ is not None and m.ng_solve_status['reason'] == 'evaluation error',
          f'{e} / cause {type(e.__cause__).__name__}')
except Exception as e:
    check('LOG(-1) with newton_raise: EquationError', False, f'got {type(e).__name__}: {e}')

#%% summary
bad = [k for k, v in results.items() if not v]
print(f"\n{len(results) - len(bad)} of {len(results)} passed" + (f'; failed: {bad}' if bad else ''))
sys.exit(1 if bad else 0)
