#%% Makemodel(scheme='rk4'): the N-body laws of motion written once, integrated with RK4.
# Three routes must give the same model: Makemodel modelflow input with doable lines, Makemodel markdown
# input, and modelode.ode_transform on a DO/ENDDO template. Checked on the figure-8 three-body orbit.
import numpy as np
import pandas as pd
from modelclass import model
from modelconstruct_estimation import Makemodel
from modelode import ode_transform, exclusion_lists, OdeTransformError

doable_text = '''
list bodies = bodies : a b c
doable r3__{bodies}__{NO{bodies}} = ((x__{NO{bodies}} - x__{bodies})**2 + (y__{NO{bodies}} - y__{bodies})**2)**1.5
doable <ODE> x__{bodies}  = vx__{bodies}
doable <ODE> y__{bodies}  = vy__{bodies}
doable <ODE> vx__{bodies} = G*sum(NO{bodies}, m__{NO{bodies}}*(x__{NO{bodies}} - x__{bodies})/r3__{bodies}__{NO{bodies}})
doable <ODE> vy__{bodies} = G*sum(NO{bodies}, m__{NO{bodies}}*(y__{NO{bodies}} - y__{bodies})/r3__{bodies}__{NO{bodies}})
<ODE> t = 1
'''
markdown_text = '# Three bodies\n' + '\n'.join('>' + l for l in doable_text.strip().splitlines())
template_text = '''
LIST BODIES = BODIES : A B C $
DO BODIES $
  DO NO{BODIES} $
    FRML <> R3__{BODIES}__{NO{BODIES}} = ((X__{NO{BODIES}} - X__{BODIES})**2 + (Y__{NO{BODIES}} - Y__{BODIES})**2)**1.5 $
  ENDDO $
  FRML <ODE> X__{BODIES}  = VX__{BODIES} $
  FRML <ODE> Y__{BODIES}  = VY__{BODIES} $
  FRML <ODE> VX__{BODIES} = G*sum(NO{BODIES}, M__{NO{BODIES}}*(X__{NO{BODIES}} - X__{BODIES})/R3__{BODIES}__{NO{BODIES}}) $
  FRML <ODE> VY__{BODIES} = G*sum(NO{BODIES}, M__{NO{BODIES}}*(Y__{NO{BODIES}} - Y__{BODIES})/R3__{BODIES}__{NO{BODIES}}) $
ENDDO $
FRML <ODE> T = 1 $
'''

mm_flow = Makemodel(doable_text, type_input='modelflow', scheme='rk4')
mm_md = Makemodel(markdown_text, scheme='rk4')
m_flow, m_md = mm_flow.mmodel, mm_md.mmodel
m_tpl = model(ode_transform(template_text, 'rk4'), modelname='template route')
print(f'endogenous: modelflow {len(m_flow.endogene)}, markdown {len(m_md.endogene)}, template {len(m_tpl.endogene)}')
assert len(m_flow.endogene) == len(m_md.endogene) == len(m_tpl.endogene) == 147

#%% solve the figure-8 on all three and compare
three = ['A', 'B', 'C']
pos0 = {'A': (-0.97000436, 0.24308753), 'B': (0.0, 0.0), 'C': (0.97000436, -0.24308753)}
vel0 = {'A': (0.4662036850, 0.4323657300), 'B': (-0.93240737, -0.86473146), 'C': (0.4662036850, 0.4323657300)}
dt, n = 0.005, 2530

def solve(m):
    df = pd.DataFrame(0.0, index=range(n + 1), columns=sorted(m.allvar))
    df['DT'], df['G'] = dt, 1.0
    for B in three:
        df[f'M__{B}'] = 1.0
        df.loc[0, [f'X__{B}', f'Y__{B}']] = pos0[B]
        df.loc[0, [f'VX__{B}', f'VY__{B}']] = vel0[B]
    return m(df, 1, n, silent=1)

r_flow, r_md, r_tpl = solve(m_flow), solve(m_md), solve(m_tpl)
cols = [f'{v}__{B}' for B in three for v in ('X', 'Y', 'VX', 'VY')]
d1 = max((r_flow[c] - r_tpl[c]).abs().max() for c in cols)
d2 = max((r_md[c] - r_tpl[c]).abs().max() for c in cols)
print(f'max |modelflow route - template| {d1:.1e}   max |markdown route - template| {d2:.1e}')
assert d1 == 0.0 and d2 == 0.0, 'the three routes must be bit-identical'
# back near the start after one period (6.3259)
i1 = int(round(6.32591398 / dt))
back = np.hypot(r_flow.loc[i1, 'X__A'] - pos0['A'][0], r_flow.loc[i1, 'Y__A'] - pos0['A'][1])
print(f'A back at its start after one period within {back:.1e}')
assert back < 1e-3

#%% stages are visible, scheme=None leaves the model alone, bad input fails fast
assert mm_flow.post_ode != mm_flow.post_sum and 'X__A__K4' in mm_flow.post_ode
mm_none = Makemodel(doable_text.replace('NO{bodies}', 'bodies').replace('doable r3', 'doable r3x'), type_input='modelflow')
assert mm_none.post_ode == mm_none.post_sum
try:
    Makemodel(doable_text.replace('vx__{bodies}\n', 'vx__{bodies}(-2)\n', 1), type_input='modelflow', scheme='rk4')
    raise AssertionError('lag -2 was not caught')
except OdeTransformError as e:
    print('lag -2 caught:', e)
print('ode_makemodel_test: all ok')
