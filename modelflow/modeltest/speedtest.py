# -*- coding: utf-8 -*-
"""
Speed test of the ng solvers on the PAK carbon-tax scenario (2020-2100).

Each solver gets two timed runs, with and without jit:

* cold -- code regenerated and recompiled (``transpile_reset=True``) and
  Jacobians / Newton structures rebuilt (``newton_reset=True``);
* warm -- the same call again, everything reused.

The model is ``modeltest/data/pak.pcim`` (a copy of the mfdemo Pakistan model).
Model loading is not timed.  Times are wall-clock time of the whole model call.
Each solution is compared with the warm non-jit ``sim_ng`` solution.

Run the #%% cells in order.  The setup cell changes the working folder to the
modelflow source folder and puts it first on ``sys.path``, so the repo version
is used, not an installed modelflow.  If
``modelclass`` is already imported from somewhere else in this session,
restart the kernel first.
"""
#%% setup
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    MODELTEST = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")
MODELFLOW = MODELTEST.parent
DATA = MODELTEST / 'data'               # pak.pcim copied from mfdemo
# if str(MODELFLOW) not in sys.path:
#     sys.path.insert(0, str(MODELFLOW))
# # the jit solvers are written to ./modelsource and imported as package
# # 'modelsource' -- run from the modelflow folder so both are the same folder
# os.chdir(MODELFLOW)

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)

#%% settings
START, END = 2020, 2100
LJIT = [False, True]             # run every solver without and with jit

newton_opts = dict(nonlin=6)
SOLVERS = {                      # ng solver name (without _ng): extra options
    'sim': {'use_fbmin':True},
    'sim_hoist': {},
    'sim1d': {},
    'newton': newton_opts,
    'newtonstack': newton_opts,
    'newton_fbmin': newton_opts,
    'newtonstack_fbmin': newton_opts,
}

#%% load PAK and make the scenario (not timed)
mpak, baseline = model.modelload(str(DATA / 'pak.pcim'), run=1, use_fbmin=True,
                                 ljit=False)
alternative = baseline.upd("<2020 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30")
endo = sorted(mpak.endogene)


def timed_run(solver, ljit, cold):
    """Solve the scenario with *solver*; returns (seconds, result)."""
    reset = dict(transpile_reset=True, newton_reset=True) if cold else {}
    t0 = time.perf_counter()
    res = mpak(alternative, START, END, solver=f'{solver}_ng', silent=1,
               reset_options=True, ljit=ljit, **SOLVERS[solver], **reset)
    return time.perf_counter() - t0, res


#%% run: cold then warm for every solver, without and with jit
rows, results = [], {}
for ljit in LJIT:
    for solver in SOLVERS:
        label = f'{solver}_ng' + (' jit' if ljit else '')
        try:
            t_cold, _ = timed_run(solver, ljit, cold=True)
            t_warm, res = timed_run(solver, ljit, cold=False)
        except Exception as e:
            print(f'{label:28s} raised {type(e).__name__}: {e}')
            rows.append(dict(solver=f'{solver}_ng', ljit=ljit, cold=np.nan,
                             warm=np.nan, error=type(e).__name__))
            continue
        results[(solver, ljit)] = res
        rows.append(dict(solver=f'{solver}_ng', ljit=ljit, cold=t_cold,
                         warm=t_warm, error=''))
        print(f'{label:28s} cold {t_cold:9.3f} s   warm {t_warm:9.3f} s')

#%% table: times and difference from the warm non-jit sim_ng solution
ref = results.get(('sim', False))
for row in rows:
    res = results.get((row['solver'][:-3], row['ljit']))
    if ref is None or res is None:
        row['max rel diff'] = np.nan
        continue
    absdiff = (res[endo] - ref[endo]).abs()
    row['max rel diff'] = float((absdiff / (ref[endo].abs() + 1e-8)).max().max())

table = pd.DataFrame(rows).set_index(['solver', 'ljit'])
with pd.option_context('display.float_format', '{:,.4g}'.format,
                       'display.width', 120):
    print(table)

if hasattr(mpak, 'ng_hoist_stats'):
    s = mpak.ng_hoist_stats
    share = 1 - s['ops_core_after'] / s['ops_core_before'] if s['ops_core_before'] else 0
    print(f"\nsim_hoist: {s['slots']} slots, operations per core sweep "
          f"{s['ops_core_before']} -> {s['ops_core_after']} ({share:.1%} hoisted)")
