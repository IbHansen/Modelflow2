"""Phase 2 tests for modelode.ode_transform on the projectile model.

1. explicit-Euler input -> method 'euler' reproduces the input model (expected to rounding).
2. 'heun' against the notebook's hand-written Heun model, and both against solve_ivp.
3. convergence order of euler / heun / rk4 against solve_ivp (DOP853, 1e-12).
4. derivative-form input gives the same text as Euler-form input.
5. diagnostics: lag -2, DT in an auxiliary, self-referencing auxiliary.
Run from the projecttile folder with PYTHONPATH set to the modelflow repo.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the modelflow folder
from modelclass import model
from modelode import ode_transform, OdeTransformError

GRAVITY, AIR_DENSITY, BC, ANGLE, V0 = 9.81, 1.225, 2709.0, 45.0, 300.0

euler_explicit = '''
FRML <> DRAG_FACTOR       = AIR_DENSITY * SPEED(-1) / (2 * BALLISTIC_COEF)                       $
FRML <> VELOCITY_X        = VELOCITY_X(-1) - TIME_STEP * DRAG_FACTOR * VELOCITY_X(-1)            $
FRML <> VELOCITY_Y        = VELOCITY_Y(-1) - TIME_STEP * (GRAVITY + DRAG_FACTOR * VELOCITY_Y(-1)) $
FRML <> SPEED             = SQRT(VELOCITY_X**2 + VELOCITY_Y**2)                                  $
FRML <> DISTANCE          = DISTANCE(-1) + TIME_STEP * VELOCITY_X(-1)                            $
FRML <> HEIGHT            = HEIGHT(-1)   + TIME_STEP * VELOCITY_Y(-1)                            $
FRML <> DRAG_DECELERATION = AIR_DENSITY * SPEED**2 / (2 * BALLISTIC_COEF)                        $
FRML <> ELAPSED_TIME      = ELAPSED_TIME(-1) + TIME_STEP                                         $
'''
# the notebook's semi-implicit Euler: same ODE, different lag pattern -> same transformed model
euler_semi = euler_explicit.replace('TIME_STEP * DRAG_FACTOR * VELOCITY_X(-1)', 'TIME_STEP * DRAG_FACTOR * VELOCITY_X') \
                           .replace('TIME_STEP * VELOCITY_X(-1)', 'TIME_STEP * VELOCITY_X') \
                           .replace('TIME_STEP * VELOCITY_Y(-1)', 'TIME_STEP * VELOCITY_Y')
derivative_form = '''
FRML <>    DRAG_FACTOR       = AIR_DENSITY * SPEED / (2 * BALLISTIC_COEF)     $
FRML <ODE> VELOCITY_X        = -DRAG_FACTOR*VELOCITY_X                        $
FRML <ODE> VELOCITY_Y        = -DRAG_FACTOR*VELOCITY_Y - GRAVITY              $
FRML <>    SPEED             = SQRT(VELOCITY_X**2 + VELOCITY_Y**2)            $
FRML <ODE> DISTANCE          = VELOCITY_X                                     $
FRML <ODE> HEIGHT            = VELOCITY_Y                                     $
FRML <>    DRAG_DECELERATION = AIR_DENSITY * SPEED**2 / (2 * BALLISTIC_COEF)  $
FRML <ODE> ELAPSED_TIME      = 1                                              $
'''
heun_hand = '''
FRML <> DRAG_FACTOR       = AIR_DENSITY * SPEED(-1) / (2 * BALLISTIC_COEF)                     $
FRML <> VELOCITY_X_PRED   = VELOCITY_X(-1) - TIME_STEP * DRAG_FACTOR * VELOCITY_X(-1)            $
FRML <> VELOCITY_Y_PRED   = VELOCITY_Y(-1) - TIME_STEP * (GRAVITY + DRAG_FACTOR * VELOCITY_Y(-1)) $
FRML <> SPEED_PRED        = SQRT(VELOCITY_X_PRED**2 + VELOCITY_Y_PRED**2)                      $
FRML <> DRAG_FACTOR_PRED  = AIR_DENSITY * SPEED_PRED / (2 * BALLISTIC_COEF)                    $
FRML <> VELOCITY_X        = VELOCITY_X(-1) - TIME_STEP * (DRAG_FACTOR * VELOCITY_X(-1)
                                                         + DRAG_FACTOR_PRED * VELOCITY_X_PRED) / 2 $
FRML <> VELOCITY_Y        = VELOCITY_Y(-1) - TIME_STEP * (GRAVITY + (DRAG_FACTOR * VELOCITY_Y(-1)
                                                         + DRAG_FACTOR_PRED * VELOCITY_Y_PRED) / 2) $
FRML <> SPEED             = SQRT(VELOCITY_X**2 + VELOCITY_Y**2)                                $
FRML <> DISTANCE          = DISTANCE(-1) + TIME_STEP * (VELOCITY_X + VELOCITY_X(-1)) / 2       $
FRML <> HEIGHT            = HEIGHT(-1)   + TIME_STEP * (VELOCITY_Y + VELOCITY_Y(-1)) / 2       $
FRML <> DRAG_DECELERATION = AIR_DENSITY * SPEED**2 / (2 * BALLISTIC_COEF)                      $
FRML <> ELAPSED_TIME      = ELAPSED_TIME(-1) + TIME_STEP                                       $
'''
EXO = ['TIME_STEP', 'GRAVITY', 'AIR_DENSITY', 'BALLISTIC_COEF']


def run(m, dt, t_max=40.0):
    n = int(round(t_max / dt))
    df = pd.DataFrame(0.0, index=range(n + 1), columns=sorted(m.allvar))
    df[EXO] = dt, GRAVITY, AIR_DENSITY, BC
    th = np.radians(ANGLE)
    df.loc[0, ['VELOCITY_X', 'VELOCITY_Y', 'SPEED']] = V0 * np.cos(th), V0 * np.sin(th), V0
    return m(df, 1, n, silent=1)


def reference(t_max=40.0):
    def rhs(t, s):
        x, y, vx, vy = s
        f = AIR_DENSITY * np.hypot(vx, vy) / (2 * BC)
        return [vx, vy, -f * vx, -GRAVITY - f * vy]
    th = np.radians(ANGLE)
    return solve_ivp(rhs, (0, t_max), [0, 0, V0 * np.cos(th), V0 * np.sin(th)],
                     method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)


def maxerr(res, ref):
    y = ref.sol(res['ELAPSED_TIME'].to_numpy())
    return max(np.abs(res['DISTANCE'].to_numpy() - y[0]).max(), np.abs(res['HEIGHT'].to_numpy() - y[1]).max())


def reldiff(r1, r2, cols):
    return max((r1[c] - r2[c]).abs().max() / max(r1[c].abs().max(), 1e-300) for c in cols)


if __name__ == '__main__':
    ref = reference()
    cols = ['DISTANCE', 'HEIGHT', 'VELOCITY_X', 'VELOCITY_Y']
    print('=' * 70, '\n1. euler input -> euler output')
    t_euler = ode_transform(euler_explicit, 'euler', dt='TIME_STEP', report=True)
    print(t_euler)
    m_in, m_out = model(euler_explicit), model(t_euler)
    print(f'   explicit Euler input vs transformed, dt=0.01: max rel diff {reldiff(run(m_in, .01), run(m_out, .01), cols):.1e}')
    print('   semi-implicit input gives the same text:', ode_transform(euler_semi, 'euler', dt='TIME_STEP') == t_euler)
    print('   derivative-form input gives the same text:', ode_transform(derivative_form, 'euler', dt='TIME_STEP') == t_euler)
    print('   rk4 from Euler form == rk4 from derivative form:',
          ode_transform(euler_semi, 'rk4', dt='TIME_STEP') == ode_transform(derivative_form, 'rk4', dt='TIME_STEP'))

    print('=' * 70, '\n2. heun vs the hand-written Heun model, dt=0.01')
    m_heun, m_hand = model(ode_transform(euler_semi, 'heun', dt='TIME_STEP')), model(heun_hand)
    r_heun, r_hand = run(m_heun, .01), run(m_hand, .01)
    print(f'   max rel diff transformed vs hand: {reldiff(r_heun, r_hand, cols):.1e}')
    print(f'   max |pos err| vs solve_ivp: transformed {maxerr(r_heun, ref):.2e}   hand {maxerr(r_hand, ref):.2e}')

    print('=' * 70, '\n3. convergence order vs solve_ivp')
    for method in ['euler', 'heun', 'midpoint', 'ralston', 'rk4']:
        # rk4 reaches the 1e-12 reference floor (~1e-8 m) below dt=0.05, so it is measured on coarser steps
        dts = [1.0, 0.5, 0.2, 0.1, 0.05] if method == 'rk4' else [0.2, 0.1, 0.05, 0.02, 0.01]
        m = model(ode_transform(derivative_form, method, dt='TIME_STEP'))
        errs = [maxerr(run(m, dt), ref) for dt in dts]
        order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        print(f'   {method:9} eqs {len(m.endogene):3}  errors {" ".join(f"{e:.1e}" for e in errs)}   order {order:.2f}')

    print('=' * 70, '\n4. diagnostics')
    bad = {
        'lag -2':       derivative_form.replace('SQRT(VELOCITY_X**2', 'SQRT(VELOCITY_X(-2)**2'),
        'DT in aux':    derivative_form.replace('SPEED**2 /', 'SPEED**2 * TIME_STEP /'),
        'self-ref aux': derivative_form.replace('FRML <ODE> ELAPSED_TIME      = 1', 'FRML <> ELAPSED_TIME = ELAPSED_TIME(-1) + 1'),
        'odd DT use':   euler_explicit.replace('ELAPSED_TIME(-1) + TIME_STEP', 'ELAPSED_TIME(-1) * EXP(TIME_STEP)'),
    }
    for name, text in bad.items():
        try:
            ode_transform(text, 'rk4', dt='TIME_STEP')
            print(f'   {name}: NOT CAUGHT')
        except OdeTransformError as e:
            print(f'   {name}: caught ({e})')
