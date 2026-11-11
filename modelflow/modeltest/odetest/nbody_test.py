"""The documented doable template (modelode.nbody_template, inlined distance, dummy index dd) through the
overlay, against the hand-generated probe (nbody_rk4_probe) on the figure-8 orbit. Names differ
(POS__A__X vs A_X), the numbers must not."""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the modelflow folder
from modelclass import model
from nbody_rk4_probe import nbody_equations, rhs_numpy
from modelode import nbody_template, nbody_makemodel, make_df, energy, pos, vel

three = ['A', 'B', 'C']
print(nbody_template(three))
m_dsl, mm = nbody_makemodel(three)
print('--- expanded by Makemodel (before the RK4 transform) ---')
print(mm.expanded_frml)
m_probe = model(nbody_equations(three), modelname='three-body, probe')
print(f'endogenous: template {len(m_dsl.endogene)}  probe {len(m_probe.endogene)}')

masses = {'A': 1.0, 'B': 1.0, 'C': 1.0}
pos0 = {'A': (-0.97000436, 0.24308753), 'B': (0.0, 0.0), 'C': (0.97000436, -0.24308753)}
vel0 = {'A': (0.4662036850, 0.4323657300), 'B': (-0.93240737, -0.86473146), 'C': (0.4662036850, 0.4323657300)}
dt, t_max = 0.005, 2 * 6.3259
n = int(round(t_max / dt))
r_dsl = m_dsl(make_df(m_dsl, three, masses, pos0, vel0, 1.0, dt, t_max), 1, n, silent=1)

dfp = pd.DataFrame(0.0, index=range(n + 1), columns=sorted(m_probe.allvar))
dfp['DT'], dfp['G'] = dt, 1.0
for B in three:
    dfp[f'M_{B}'] = masses[B]
    dfp.loc[0, [f'{B}_X', f'{B}_Y']] = pos0[B]
    dfp.loc[0, [f'{B}_VX', f'{B}_VY']] = vel0[B]
r_probe = m_probe(dfp, 1, n, silent=1)
pairs = [(pos(B, c), f'{B}_{c}') for B in three for c in 'XY'] + [(vel(B, c), f'{B}_V{c}') for B in three for c in 'XY']
diff = max((r_dsl[a] - r_probe[b]).abs().max() for a, b in pairs)
print(f'max |template - probe| over {n} steps: {diff:.1e}')

y0 = np.concatenate([np.ravel([pos0[B] for B in three]), np.ravel([vel0[B] for B in three])])
sol = solve_ivp(rhs_numpy(three, [1.0] * 3, 1.0), (0, t_max), y0, method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
ref = sol.sol(r_dsl['T'].to_numpy())
err = max(np.abs(r_dsl[pos(B, c)].to_numpy() - ref[2 * i + j]).max() for i, B in enumerate(three) for j, c in enumerate('XY'))
e = energy(r_dsl, three, masses, 1.0)
print(f'figure-8: max |pos err| vs solve_ivp {err:.2e}, energy drift {abs(e.iloc[-1] / e.iloc[0] - 1):.2e}')
i1 = int(round(6.3259 / dt))
print(f'after one period, A is back within {np.hypot(r_dsl.loc[i1, pos("A", "X")] - pos0["A"][0], r_dsl.loc[i1, pos("A", "Y")] - pos0["A"][1]):.1e}')

m4, _ = nbody_makemodel(['SUN', 'EARTH', 'MOON', 'MARS'], dims=3)
print(f'4 bodies in 3-D: {len(m4.endogene)} endogenous equations')
