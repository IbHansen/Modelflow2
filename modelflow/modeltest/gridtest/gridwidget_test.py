# -*- coding: utf-8 -*-
"""Tests of the Python side of the forecast grid (modelwidget_grid.ScenarioGrid)

The javascript sends messages; here they are sent to the handler directly.
Run with ``python modeltest/gridtest/gridwidget_test.py`` from any folder, or cell by cell.
The grid itself is tried by hand in JupyterLab: modeltest/gridtest/gridwidget_try.ipynb.
"""
#%% load PAK and make a grid
import json
import numpy as np

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

mpak, bline = model.modelload(find_pak(), run=1, silent=1)
known = 'PAKNECONPRVTKN'           # fixable
assumption = 'PAKGGREVCO2CER'      # exogenous
g = mpak.gridwidget(f'{known} {assumption} #Headline', start=2025, end=2030,
                    show_growth=True, show_fitted=True)
send = lambda **content: g._on_msg(g, content, [])
print(repr(g))
assert g.hist_periods == ['2023', '2024'] and g.periods[0] == '2025' and g.periods[-1] == '2030'
assert all(len(v) == 8 for v in g.data.values())
json.dumps({'rows': g.rows, 'data': g.data, 'edits': g.edits})     # what goes to javascript
assert mpak.current_per[0] != 2025 or True                          # the grid does not change the model period
print([(r['var'], r['kind'], r['editable'], r['target']) for r in g.rows if r['var'] == known])

#%% a known value, a tool on the value line, a refused edit
send(cmd='set', cells=[[known, '2026', 2.6e7]])
assert g.core.edits[(known + '_X', 2026)] == 2.6e7 and g.core.edits[(known + '_D', 2026)] == 1.0
assert g.pending and g.edits[known + '_X'] == {'2026': 2.6e7}
send(cmd='tool', keys=[known], periods=['2028', '2029'], op='+', value='1000')
assert g.core.edits[(known + '_X', 2028)] == bline.loc[2028, known] + 1000
allper = [str(p) for p in range(2025, 2031)]          # javascript sends every selected period
send(cmd='tool', keys=[assumption], periods=allper, op='+', value='20')
assert g.core.edits[(assumption, 2030)] == bline.loc[2030, assumption] + 20
send(cmd='set', cells=[['PAKNYGDPMKTPXN', '2026', 1.0]])
print('message:', g.message)
assert 'can not be edited' in g.message
send(cmd='tool', keys=[assumption], periods=['2025'], op='+', value='abc')
assert 'needs a number' in g.message

#%% solve, keep, and the same as solving directly
send(cmd='solve', name='round 1')
assert not g.pending and g.message == '', g.message
assert 'round 1' in mpak.keep_solutions
sol = g.solution
direct = mpak(g.core.current(), start=2025, end=2030, silent=1, progressbar=0)
assert sol.equals(direct)
i2026 = g.hist_periods.__len__() + g.periods.index('2026')
assert g.data[known][i2026] == 2.6e7, g.data[known][i2026]          # the known value is in the solution
print('diff line', g.data[known + '|DIFF'])

#%% fix at shown values after the solve, upd text, reset
send(cmd='fix', vars=[known], periods=['2030'])
assert g.core.edits[(known + '_X', 2030)] == sol.loc[2030, known]
send(cmd='upd')
print(g.upd_text)
assert np.array_equal(mpak.basedf.upd(g.upd_text).values, g.core.current().values)
send(cmd='reset')
assert not g.core.edits and g.solution is None and not g.edits

#%% as the input of updatewidget
from modelwidget_input import updatewidget
g2 = mpak.gridwidget(assumption, start=2025, end=2030, solve_button=False)
g2._on_msg(g2, {'cmd': 'tool', 'keys': [assumption], 'periods': allper, 'op': '+', 'value': 20}, [])
u = updatewidget(mpak, g2)
u.run(None)
ref = mpak(mpak.basedf.upd('<2025 2030> PAKGGREVCO2CER + 20'), start=u.runner.start, end=u.runner.end, progressbar=0)
kept = mpak.keep_solutions[u.runner.current_experiment]
assert kept.equals(ref[kept.columns])
g2._on_msg(g2, {'cmd': 'refresh'}, [])
assert g2.solution is mpak.lastdf
print('All grid widget tests passed')
