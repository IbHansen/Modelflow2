# -*- coding: utf-8 -*-
"""
Tests of the transpiled jit file (ljit=True, stringjit=False) of the legacy
solvers, which now go through modelsolver_ng.import_transpiled like the ng
solvers.

The generated code is written to modelsource/<model>_<solver>_jit_jitsolver.py
and imported from there.  It has the databank column positions built in, so the
file must be rewritten whenever the generated text changes.  Before the change
the legacy makelos

* imported an existing file without checking it (wrong column positions after
  a change of model or data layout, cell 5),
* raised UnboundLocalError when the data layout changed in a session (cell 4),
* did not reload an already imported module after transpile_reset (cell 6).

Legacy 'sim' and ng 'sim_ng' write the same file and store the functions under
the same model attributes (cell 7).

The model is modeltest/data/pak.pcim with the carbon-tax scenario of
speedtest.py.  The reference is legacy 'sim' without jit.  The jit files are
written to modeltest/jitwork/modelsource (git-ignored); the working directory
is changed to modeltest/jitwork.

Run the #%% cells in order.  Each cell prints what it checks; an assert stops
at the first failure.  The first jit solve compiles PAK (a minute or so).
"""
#%% setup
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    MODELTEST = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")
DATA = MODELTEST / 'data'
WORK = MODELTEST / 'jitwork'

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)

# start without jit files, in a folder of its own.  A modelsource package
# imported earlier in this session (maybe from another folder) is forgotten.
WORK.mkdir(exist_ok=True)
shutil.rmtree(WORK / 'modelsource', ignore_errors=True)
os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
for name in [n for n in sys.modules if n == 'modelsource' or n.startswith('modelsource.')]:
    del sys.modules[name]

START, END = 2020, 2100

mpak, baseline = model.modelload(str(DATA / 'pak.pcim'), run=1, ljit=False)
alternative = baseline.upd("<2020 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30")
endo = sorted(mpak.endogene)

ref = mpak(alternative, START, END, solver='sim', ljit=False,
           silent=1, reset_options=True)[endo]


def maxdiff(res):
    """Largest difference to the reference, relative where the reference is
    large."""
    a = res[endo].loc[START:END].to_numpy(dtype=float)
    b = ref.loc[START:END].to_numpy(dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def check(res, what, tol=1e-8):
    d = maxdiff(res)
    print(f'{what}: max relative difference to sim without jit {d:.2e}')
    assert d < tol, f'{what}: the result differs from the reference'


def jitname(solvename='sim'):
    return f'{mpak.name}_{solvename}_jit'


def jitfile(solvename='sim'):
    safename = re.sub(r'\W', '_', jitname(solvename))
    return Path('modelsource') / f'{safename}_jitsolver.py'


def stamp(solvename='sim'):
    """(modification time, text) of the jit file."""
    f = jitfile(solvename)
    return f.stat().st_mtime_ns, f.read_text(encoding='utf-8')


def functions(solvename='sim'):
    """The compiled functions stored on the model."""
    return tuple(getattr(mpak, f'{p}_{jitname(solvename)}', None) for p in ('pro', 'core', 'epi'))


def new_session(solvename='sim'):
    """Forget the compiled functions and the imported module, as in a new
    Python session.  The file on disk stays."""
    for p in ('pro', 'core', 'epi'):
        if hasattr(mpak, f'{p}_{jitname(solvename)}'):
            delattr(mpak, f'{p}_{jitname(solvename)}')
    sys.modules.pop(f'modelsource.{jitfile(solvename).stem}', None)


def solve(df=None, solver='sim', **kwargs):
    df = alternative if df is None else df
    return mpak(df, START, END, solver=solver, ljit=True, stringjit=False,
                silent=1, reset_options=True, **kwargs)


print('jit file:', jitfile())

#%% 1. the first jit solve writes the file
assert not jitfile().exists()
res = solve()
assert jitfile().is_file(), 'no jit file written'
check(res, 'first jit solve')
stamp1 = stamp()
print('ok')

#%% 2. the same solve again reuses the compiled functions, the file is untouched
before = functions()
res = solve()
check(res, 'second jit solve')
assert all(a is b for a, b in zip(before, functions())), 'the functions were rebuilt'
assert stamp() == stamp1, 'the file was rewritten'
print('ok')

#%% 3. a new session with an unchanged model: the file is checked and reused
new_session()
res = solve()
check(res, 'new session, same model')
assert stamp() == stamp1, 'an unchanged file was rewritten (numba would recompile)'
print('ok')

#%% 4. another column layout in the same session: the file is rewritten
# columns reversed plus an extra column -> other column positions in the code.
# Before the change this raised UnboundLocalError (pro_jit not assigned).
other_layout = alternative[alternative.columns[::-1]].assign(JITTEST_EXTRA=0.0)
res = solve(other_layout)
check(res, 'other column layout')
stamp4 = stamp()
assert stamp4[1] != stamp1[1], 'the code was not regenerated for the new layout'
print('ok')

#%% 5. a new session with the first layout: the file from cell 4 is stale
# Before the change this file was imported as it was: the column positions of
# other_layout used on alternative -> wrong numbers.
new_session()
res = solve()
check(res, 'new session, stale file')
assert stamp()[1] == stamp1[1], 'the stale file was not rewritten'
print('ok')

#%% 6. transpile_reset rewrites the file and reloads the module
before = functions()
mtime_before = stamp()[0]
res = solve(transpile_reset=True)
check(res, 'transpile_reset')
assert stamp()[0] > mtime_before, 'the file was not rewritten'
assert all(a is not b for a, b in zip(before, functions())), 'the old functions are still used'
print('ok')

#%% 7. legacy sim and ng sim_ng share the file and the stored functions
# A new session, so sim_ng generates its own text and compares it with the
# legacy file.  Same text: the file is kept.  Other text: sim_ng rewrites it,
# and the next legacy solve after a new session writes it back.
new_session()
legacy_text = stamp()[1]
res = solve(solver='sim_ng')
# another implementation of the same Gauss-Seidel: equal within the convergence tolerance
check(res, 'sim_ng with the shared file', tol=1e-5)
same = stamp()[1] == legacy_text
print('legacy sim and sim_ng generate the same code' if same else
      'legacy sim and sim_ng generate different code: switching between them recompiles')
new_session()
res = solve()
check(res, 'legacy sim after sim_ng')
assert stamp()[1] == stamp1[1]
print('ok')

#%% 8. legacy res solver (used by model.test_model): file written and checked
# res only calculates the equations once, so compare with res without jit.
ref_res = mpak(alternative, START, END, solver='res', ljit=False,
               silent=1, reset_options=True)[endo]
res = solve(solver='res')
assert jitfile('res').is_file(), 'no jit file for res'
d = float(np.max(np.abs(res[endo].loc[START:END].to_numpy(dtype=float) -
                        ref_res.loc[START:END].to_numpy(dtype=float)) /
                 np.maximum(1.0, np.abs(ref_res.loc[START:END].to_numpy(dtype=float)))))
print(f'res with jit vs without: max relative difference {d:.2e}')
assert d < 1e-8
print('ok')

#%% done
print('all jit file tests passed')
