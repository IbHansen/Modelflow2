# -*- coding: utf-8 -*-
"""Tests of GridCore in modelwidget_core (phase 1 of experiments/gridwidget/gridwidget_design.md)

Run with ``python modeltest/gridtest/gridcore_test.py`` from any folder, or cell by cell.
"""
#%% imports and a small model
import numpy as np
import pandas as pd

from modelclass import model
from pathlib import Path

def find_pak():
    """modeltest/data/pak.pcim, found from this file or from the folder the cells run in"""
    try:
        here = [Path(__file__).resolve().parent]
    except NameError:          # run cell by cell
        here = []
    cwd = Path.cwd()
    for folder in here + [cwd, cwd / 'modeltest' / 'gridtest', cwd / 'gridtest']:
        pak = folder.parent / 'data' / 'pak.pcim'
        if pak.exists():
            return str(pak)
    raise FileNotFoundError('modeltest/data/pak.pcim not found, run from modelflow, modeltest or modeltest/gridtest')
from modelwidget_core import GridCore, ScenarioRunner

# C fixable with add factor and fitted, I with an add factor only, G exogenous, Y plain endogenous
small_eq = '''
FRML <> C = (0.6*Y + C_A)*(1-C_D) + C_X*C_D $
FRML <> C_FITTED = 0.6*Y $
FRML <> I = 0.2*Y(-1) + I_A $
FRML <> Y = C + I + G $
'''
msmall = model.from_eq(small_eq, silent=True)
df0 = pd.DataFrame({'G': 100.0, 'Y': 400.0}, index=range(2020, 2031))
df0 = msmall.insertModelVar(df0)
base_small = msmall(df0, 2021, 2030, silent=True)
print('fix_endo', msmall.fix_endo, 'exogene', sorted(msmall.exogene))

def solve_small(df):
    return msmall(df, 2021, 2030, silent=True)

#%% roles and lines
with msmall.set_smpl(2024, 2028):
    core = GridCore.from_model(msmall, 'C_X G I Y', df=base_small, show_fitted=True)
kinds = {g.var: (g.role, [l.kind for l in g.lines]) for g in core.groups}
print(kinds)
assert [g.var for g in core.groups] == ['C', 'G', 'I', 'Y'], 'C_X is folded into C'
assert kinds['C'] == ('fixable', ['value', 'fix', 'flag', 'add', 'fitted', 'diff'])
assert kinds['G'] == ('exogenous', ['value'])
assert kinds['I'] == ('add', ['value', 'add'])
assert kinds['Y'] == ('endogenous', ['value'])
assert list(core.periods) == list(range(2024, 2029))
print(core.table())

#%% auto dummy, set and clear
core.set_cell('C_X', 2025, 500)
assert core.edits[('C_D', 2025)] == 1.0
core.clear_cell('C_X', 2025)
assert not core.edits, core.edits
core.set_cell('C_D', '2026', True)          # periods can come as text from a front end
core.set_cell('C_X', 2026, 510)
core.clear_cell('C_X', 2026)
assert core.edits == {('C_D', 2026): 1.0}, 'a flag set by the user stays'
core.reset()
core.set_cell('C', 2025, 301.5)             # a known value of a fixable variable: goes to C_X
assert core.edits == {('C_X', 2025): 301.5, ('C_D', 2025): 1.0}, core.edits
core.clear_cell('C', 2025)
assert not core.edits
with msmall.set_smpl(2024, 2028):
    nofix = GridCore.from_model(msmall, 'C', df=base_small, value_fixes=False)
for bad in [lambda: nofix.set_cell('C', 2025, 1), lambda: core.set_cell('Y', 2025, 1), lambda: core.set_cell('I', 2025, 1),
            lambda: core.set_cell('G', 2035, 1), lambda: core.set_cell('XX', 2025, 1)]:
    try:
        bad()
        raise AssertionError('should fail')
    except ValueError as e:
        print('ok, refused:', e)

#%% exogenous edits give the same solution as upd
core.apply_tool('G', (2025, 2027), '+', 10)
core.set_cell('G', 2028, 95)
grid_df = base_small.copy()
core.update_df(grid_df)
upd_df = base_small.upd('<2025 2027> G + 10\n<2028 2028> G = 95')
assert grid_df.equals(upd_df)
assert solve_small(grid_df).equals(solve_small(upd_df))
print(core.as_upd())

#%% fix to baseline is the same as model.fix
core.reset()
core.fix_to_baseline('C', (2025, 2027))
grid_df = base_small.copy()
core.update_df(grid_df)
fix_df = msmall.fix(base_small, 'C', 2025, 2027, silent=1)
assert np.array_equal(grid_df.values, fix_df.values)
assert solve_small(grid_df).equals(solve_small(fix_df))

#%% as_upd round trip, with gaps and several lines
core.reset()
core.set_cell('C_X', 2024, 300.123456789)
core.set_cell('C_X', 2026, 310.1)
core.set_cell('C_X', 2027, 1 / 3)
core.apply_tool('C_A', (2025, 2028), '+', 1.5)
core.apply_tool('G', (2025, 2028), '%growth', 2.5)
core.apply_tool('I_A', (2024, 2026), '=impulse', 7)
text = core.as_upd()
print(text)
grid_df = base_small.copy()
core.update_df(grid_df)
assert np.array_equal(grid_df.values, base_small.upd(text).values)

#%% fitted and diff lines
core.reset()
diff = core.derived(solve_small(base_small))['C|DIFF']
assert np.allclose(diff, 0, atol=1e-3), diff                 # no add factor, no fix (solver tolerance)
core.set_cell('C_A', 2025, 5.0)
sol = solve_small(core.current())
diff = core.derived(sol)['C|DIFF']
assert abs(diff[2025] - 5.0) < 1e-3, diff                    # the add factor
core.reset()
core.fix_to_baseline('C', [2026])
core.set_cell('C_X', 2026, base_small.loc[2026, 'C'] + 20)
diff = core.derived(solve_small(core.current()))['C|DIFF']
print(diff)
assert diff[2026] != 0
with msmall.set_smpl(2024, 2028):
    pct = GridCore.from_model(msmall, 'C', df=base_small, show_fitted=True, fitted_diff='pct')
assert np.allclose(pct.derived(sol)['C|DIFF'], 100 * (sol.loc[2024:2028, 'C'] / sol.loc[2024:2028, 'C_FITTED'] - 1))
with msmall.set_smpl(2024, 2028):
    nofit = GridCore.from_model(msmall, 'C', df=base_small)
assert [l.kind for l in nofit.groups[0].lines] == ['value', 'fix', 'flag', 'add'], 'show_fitted is off by default'
with msmall.set_smpl(2024, 2028):
    some = GridCore.from_model(msmall, 'C', df=base_small, lines={'c': ['value', 'add']})
assert [l.kind for l in some.groups[0].lines] == ['value', 'add']

#%% history columns and the growth line
with msmall.set_smpl(2024, 2028):
    gr = GridCore.from_model(msmall, 'G C', df=base_small, show_growth=True, history=3)
assert list(gr.hist_periods) == [2021, 2022, 2023]
assert [l.kind for l in gr.groups[0].lines] == ['value', 'growth']
gr.set_cell('G', 2025, 110)
sh = gr.shown()
print(sh)
assert list(sh.columns) == list(range(2021, 2029))
assert abs(sh.loc['G|GROWTH', 2025] - 10) < 1e-12 and abs(sh.loc['G|GROWTH', 2026] + 100 / 11) < 1e-9
assert sh.loc['C_X', 2021] == 0.0

#%% PAK: headline variables, fix to baseline and an exogenous shock through ScenarioRunner
mpak, bline = model.modelload(find_pak(), run=1, silent=1)
fixvar = 'PAKNECONPRVTKN'
assert fixvar in mpak.fix_endo
with mpak.set_smpl(2025, 2030):
    gpak = GridCore.from_model(mpak, f'#Headline {fixvar} PAKGGREVCO2CER', show_fitted=True)
print({g.var: g.role for g in gpak.groups})
assert {g.var: g.role for g in gpak.groups}[fixvar] == 'fixable'

gpak.fix_to_baseline(fixvar, (2025, 2027))
gpak.apply_tool('PAKGGREVCO2CER', (2025, 2030), '+', 20)
grid_df = mpak.basedf.copy()
gpak.update_df(grid_df)
ref_df = mpak.fix(mpak.basedf, fixvar, 2025, 2027, silent=1).upd('<2025 2030> PAKGGREVCO2CER + 20')
assert np.array_equal(grid_df.values, ref_df.values)
assert np.array_equal(mpak.basedf.upd(gpak.as_upd()).values, grid_df.values)

runner = ScenarioRunner(mpak)
with mpak.set_smpl(2020, 2030):
    runner.set_period()
    runner.run(gpak, 'grid')
    ref = mpak(ref_df, start=2020, end=2030, progressbar=0)
kept = mpak.keep_solutions['grid']          # keep_solutions holds the model variables only
assert kept.equals(ref[kept.columns]), 'ScenarioRunner with the grid = the same change solved directly'
d = gpak.derived(mpak.keep_solutions['grid'])[f'{fixvar}|DIFF']
print(gpak.table(mpak.keep_solutions['grid']).loc[fixvar])
print('All GridCore tests passed')
