"""Probe: hand-written RK4 N-body model in ModelFlow's DSL.

Generates the flat FRML text an ODE->RK4 preprocessor would emit, builds it with
model(), solves, and checks against scipy solve_ivp and energy conservation.
Run from the projecttile folder:  python nbody_rk4_probe.py
"""
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the modelflow folder
from modelclass import model

COORDS = ['X', 'Y']
RK4 = dict(a=[[], [0.5], [0, 0.5], [0, 0, 1]], b=[1/6, 1/3, 1/3, 1/6])   # Butcher tableau


def nbody_equations(bodies, tableau=RK4, dt='DT', g='G'):
    """FRML text for an N-body RK4 step. States: {B}_{C} (position), {B}_V{C} (velocity).
    Exogenous: DT, G, M_{B}."""
    a, b = tableau['a'], tableau['b']
    nst = len(b)
    eq = []
    pairs = [(p, q) for i, p in enumerate(bodies) for q in bodies[i + 1:]]
    states = [f'{B}_{C}' for B in bodies for C in COORDS] + [f'{B}_V{C}' for B in bodies for C in COORDS]
    for s in range(1, nst + 1):
        # stage values of the states: y(-1) + dt * sum_j a_sj k_j
        for S in states:
            terms = ' + '.join(f'{aij}*{S}__K{j + 1}' for j, aij in enumerate(a[s - 1]) if aij)
            rhs = f'{S}(-1)' + (f' + {dt}*({terms})' if terms else '')
            eq.append(f'FRML <> {S}__S{s} = {rhs} $')
        # auxiliaries at this stage: pair distances cubed, accelerations
        for p, q in pairs:
            d2 = ' + '.join(f'({q}_{C}__S{s} - {p}_{C}__S{s})**2' for C in COORDS)
            eq.append(f'FRML <> R3_{p}_{q}__S{s} = ({d2})**1.5 $')
        for B in bodies:
            for C in COORDS:
                terms = []
                for O in bodies:
                    if O == B:
                        continue
                    p, q = (B, O) if bodies.index(B) < bodies.index(O) else (O, B)
                    terms.append(f'M_{O}*({O}_{C}__S{s} - {B}_{C}__S{s})/R3_{p}_{q}__S{s}')
                eq.append(f'FRML <> {B}_A{C}__S{s} = {g}*(' + ' + '.join(terms) + ') $')
        # derivatives k_s: dx/dt = v, dv/dt = a
        for B in bodies:
            for C in COORDS:
                eq.append(f'FRML <> {B}_{C}__K{s} = {B}_V{C}__S{s} $')
                eq.append(f'FRML <> {B}_V{C}__K{s} = {B}_A{C}__S{s} $')
    # update
    for S in states:
        terms = ' + '.join(f'{bi}*{S}__K{i + 1}' for i, bi in enumerate(b))
        eq.append(f'FRML <> {S} = {S}(-1) + {dt}*({terms}) $')
    eq.append(f'FRML <> T = T(-1) + {dt} $')
    return '\n'.join(eq)


def rhs_numpy(bodies, masses, g):
    n = len(bodies)
    def f(t, y):
        pos, vel = y[:2 * n].reshape(n, 2), y[2 * n:].reshape(n, 2)
        acc = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = pos[j] - pos[i]
                    acc[i] += g * masses[j] * d / np.linalg.norm(d) ** 3
        return np.concatenate([vel.ravel(), acc.ravel()])
    return f


def energy(res, bodies, masses, g):
    n = len(bodies)
    kin = sum(0.5 * masses[B] * (res[f'{B}_VX'] ** 2 + res[f'{B}_VY'] ** 2) for B in bodies)
    pot = 0
    for i, p in enumerate(bodies):
        for q in bodies[i + 1:]:
            r = np.hypot(res[f'{q}_X'] - res[f'{p}_X'], res[f'{q}_Y'] - res[f'{p}_Y'])
            pot = pot - g * masses[p] * masses[q] / r
    return kin + pot


def run(bodies, masses, pos0, vel0, g, dt, t_max, ljit=False, label=''):
    text = nbody_equations(bodies)
    t0 = time.perf_counter()
    m = model(text, modelname=f'N-body RK4 {label}')
    t_build = time.perf_counter() - t0
    n = int(round(t_max / dt))
    df = pd.DataFrame(0.0, index=range(n + 1), columns=sorted(m.allvar))
    df['DT'], df['G'] = dt, g
    for B in bodies:
        df[f'M_{B}'] = masses[B]
        df.loc[0, [f'{B}_X', f'{B}_Y']] = pos0[B]
        df.loc[0, [f'{B}_VX', f'{B}_VY']] = vel0[B]
    t0 = time.perf_counter()
    res = m(df, 1, n, silent=1, ljit=ljit)
    t_solve = time.perf_counter() - t0
    # reference
    y0 = np.concatenate([np.ravel([pos0[B] for B in bodies]), np.ravel([vel0[B] for B in bodies])])
    sol = solve_ivp(rhs_numpy(bodies, [masses[B] for B in bodies], g), (0, t_max), y0,
                    method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
    ref = sol.sol(res['T'].to_numpy())
    err = 0.0
    for i, B in enumerate(bodies):
        err = max(err, np.abs(res[f'{B}_X'].to_numpy() - ref[2 * i]).max(),
                  np.abs(res[f'{B}_Y'].to_numpy() - ref[2 * i + 1]).max())
    e = energy(res, bodies, masses, g)
    print(f'{label:22} N={len(bodies)} eqs={len(m.allvar)-len(m.exogene):4d} steps={n:6d} dt={dt:<7g} '
          f'build {t_build:5.2f}s solve {t_solve:6.2f}s  max|pos err| {err:.2e}  '
          f'energy drift {abs(e.iloc[-1]/e.iloc[0]-1):.2e}')
    return m, res, err


if __name__ == '__main__':
    # 1. Two-body circular orbit (Kepler), analytic period 2*pi
    two = ['SUN', 'EARTH']
    m2 = {'SUN': 1.0, 'EARTH': 1e-6}
    p2 = {'SUN': (0, 0), 'EARTH': (1, 0)}
    v2 = {'SUN': (0, 0), 'EARTH': (0, 1)}
    for dt in [0.1, 0.05, 0.01]:
        run(two, m2, p2, v2, g=1.0, dt=dt, t_max=4 * np.pi, label=f'kepler dt={dt}')

    # 2. Figure-8 three-body choreography (Chenciner-Montgomery), period ~6.3259
    three = ['A', 'B', 'C']
    m3 = {'A': 1.0, 'B': 1.0, 'C': 1.0}
    p3 = {'A': (-0.97000436, 0.24308753), 'B': (0.0, 0.0), 'C': (0.97000436, -0.24308753)}
    v3 = {'A': (0.4662036850, 0.4323657300), 'B': (-0.93240737, -0.86473146),
          'C': (0.4662036850, 0.4323657300)}
    errs = {}
    for dt in [0.02, 0.01, 0.005, 0.0025]:
        *_, errs[dt] = run(three, m3, p3, v3, g=1.0, dt=dt, t_max=2 * 6.3259, label=f'figure-8 dt={dt}')
    h = np.array(list(errs)); e = np.array(list(errs.values()))
    print(f'estimated order (figure-8): {np.polyfit(np.log(h), np.log(e), 1)[0]:.2f}')

    # 3. Speed with jit, and a bigger system (10 bodies, random)
    run(three, m3, p3, v3, g=1.0, dt=0.005, t_max=2 * 6.3259, ljit=True, label='figure-8 jit')
    rng = np.random.default_rng(0)
    ten = [f'B{i}' for i in range(10)]
    m10 = {B: 1.0 for B in ten}
    p10 = {B: tuple(rng.normal(0, 3, 2)) for B in ten}
    v10 = {B: tuple(rng.normal(0, .3, 2)) for B in ten}
    run(ten, m10, p10, v10, g=1.0, dt=0.001, t_max=2.0, label='10 bodies')
    run(ten, m10, p10, v10, g=1.0, dt=0.001, t_max=2.0, ljit=True, label='10 bodies jit')
