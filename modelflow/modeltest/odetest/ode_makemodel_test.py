#%% Three ways of writing the same N-body model must give the same Runge-Kutta model, bit for bit:
#  1. the documented markdown template with doable lines (modelode.nbody_template) through ode_makemodel
#  2. the same model lines as a plain string (type_input='modelflow')
#  3. explicit DO/ENDDO loops through ode_transform (no Makemodel)
# Checked on the figure-8 three-body orbit. The ODE code is modelode.py in the ModelFlow folder; nothing else is changed.
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the modelflow folder
from modelclass import model
from modelode import ode_makemodel, ode_transform, OdeTransformError
from modelode import nbody_template, make_df, pos, vel

three = ['A', 'B', 'C']
markdown_text = nbody_template(three)
# the model lines only: '>' lines, a '>>' line joined to the line before it, markers dropped
plain_text = '\n'.join(l for l in markdown_text.splitlines() if l.startswith('>')).replace('\n>>', ' ')
plain_text = '\n'.join(l[1:] for l in plain_text.splitlines())
do_text = '''
LIST BODIES = BODIES : A B C $
LIST DIMS = DIMS : X Y $
LIST DD = DD : X Y $
DO BODIES $
  DO DIMS $
    FRML <ODE> POS__{BODIES}__{DIMS} = V__{BODIES}__{DIMS} $
    FRML <ODE> V__{BODIES}__{DIMS} = G*sum(NO{BODIES}, M__{NO{BODIES}}*(POS__{NO{BODIES}}__{DIMS} - POS__{BODIES}__{DIMS})
                                      / (sum(DD, (POS__{NO{BODIES}}__{DD} - POS__{BODIES}__{DD})**2))**1.5) $
  ENDDO $
ENDDO $
FRML <ODE> T = 1 $
'''
m_md, mm_md = ode_makemodel(markdown_text, 'rk4')
m_plain, mm_plain = ode_makemodel(plain_text, 'rk4', type_input='modelflow')
m_do = model(ode_transform(do_text, 'rk4'), modelname='DO/ENDDO route')
print(mm_md.expanded_frml)
print(f'endogenous: markdown {len(m_md.endogene)}, plain {len(m_plain.endogene)}, DO/ENDDO {len(m_do.endogene)}')
assert len(m_md.endogene) == len(m_plain.endogene) == len(m_do.endogene) == 117

#%% solve the figure-8 on all three and compare
masses = {'A': 1.0, 'B': 1.0, 'C': 1.0}
pos0 = {'A': (-0.97000436, 0.24308753), 'B': (0.0, 0.0), 'C': (0.97000436, -0.24308753)}
vel0 = {'A': (0.4662036850, 0.4323657300), 'B': (-0.93240737, -0.86473146), 'C': (0.4662036850, 0.4323657300)}
dt, n = 0.005, 2530
runs = [m(make_df(m, three, masses, pos0, vel0, 1.0, dt, n * dt), 1, n, silent=1) for m in (m_md, m_plain, m_do)]
cols = [pos(B, c) for B in three for c in 'XY'] + [vel(B, c) for B in three for c in 'XY']
d1 = max((runs[1][c] - runs[0][c]).abs().max() for c in cols)
d2 = max((runs[2][c] - runs[0][c]).abs().max() for c in cols)
print(f'max |plain - markdown| {d1:.1e}   max |DO/ENDDO - markdown| {d2:.1e}')
assert d1 == 0.0 and d2 == 0.0, 'the three routes must be bit-identical'
i1 = int(round(6.32591398 / dt))
back = np.hypot(runs[0].loc[i1, pos('A', 'X')] - pos0['A'][0], runs[0].loc[i1, pos('A', 'Y')] - pos0['A'][1])
print(f'A back at its start after one period within {back:.1e}')
assert back < 1e-3

#%% the Makemodel keeps the documented, untransformed model; bad input fails fast
assert 'POS__A__X__K4' in m_md.equations and 'POS__A__X__K4' not in mm_md.expanded_frml
assert '$$' in mm_md.markdown_model or 'dot' in mm_md.original_statements     # the LaTeX documentation is kept
try:
    ode_makemodel(markdown_text.replace('= v__{bodies}__{dims}\n', '= v__{bodies}__{dims}(-2)\n', 1), 'rk4')
    raise AssertionError('lag -2 was not caught')
except OdeTransformError as e:
    print('lag -2 caught:', e)
print('ode_makemodel_test: all ok')
