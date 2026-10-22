# -*- coding: utf-8 -*-
"""
Is the model option ``ng=True`` ready to be the default?

``ng=True`` (``model(..., ng=True)``, ``modelload(..., ng=True)`` or
``mmodel.ng = True``) makes a call without ``solver=`` dispatch to the ng
solvers (xgenr_ng, sim_ng, newton_ng, newtonstack_ng) instead of the legacy
ones.  This test solves every model in ``CASES`` twice, once on a model loaded
with ``ng=False`` and once on a model loaded with ``ng=True``, both without
``solver=``, and compares the solutions:

* the solver the dispatch picked;
* cold time (first call, code generated and compiled) and warm time (the same
  call again);
* the largest relative and absolute difference over the solved periods and
  the variable where it is; the number of values outside
  ``np.isclose(rtol=RTOL, atol=ATOL)``.

The downstream cell then runs the features that call the model without
``solver=`` or read what a solve leaves behind (keep, dekomp, fix, invert,
lastdf/basedf, the add-factor model) on fresh PAK models with ``ng=False`` and
``ng=True``, and reports which raise and whether the results agree.  Fresh
models, so a legacy attribute left by an earlier legacy solve can not hide a
missing one.

The models in ``modeltest/data`` (PAK, FRB/US MCE) and the small models below
always run; the models in ``EXTRA_MODELS`` run if their file exists.

Run the #%% cells in order or ``%runfile`` the file.  Jit code is compiled in
memory (``stringjit=True``), so no ``modelsource/`` files are written.  If
``modelclass`` is already imported from somewhere else in this session,
restart the kernel first.
"""
#%% setup
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    MODELTEST = Path(__file__).resolve().parent
except NameError:                       # interactive cell without __file__
    MODELTEST = Path(r"C:\modelflow2\modelflow\modeltest")
DATA = MODELTEST / 'data'

import modelclass
from modelclass import model

print('modelclass imported from', modelclass.__file__)

#%% settings
LJIT = False                     # True: compare the jit versions
RTOL, ATOL = 1e-5, 1e-6          # np.isclose tolerance for "same solution"
# collect garbage before each timed call and keep the collector off while the
# clock runs
GC_QUIET = True
all_opts = dict(stringjit=True)  # options for every call (no solver=)

# models outside modeltest; a missing file is skipped.  label: (pcim file, options)
EXTRA_MODELS = {
    'bolivia':  (r'C:\mfdemo\Pakistan\WorldbankModels\models\Bolivia\data\bol.pcim', {}),
    'croatia':  (r'C:\mfdemo\Pakistan\WorldbankModels\models\Croatia\data\hrv.pcim', {}),
    'iraq':     (r'C:\mfdemo\Pakistan\WorldbankModels\models\Iraq\data\irq.pcim', {}),
    'nepal':    (r'C:\mfdemo\Pakistan\WorldbankModels\models\Nepal\data\NPL.pcim', {}),
    'turkiye':  (r'C:\mfdemo\Pakistan\WorldbankModels\models\Türkiye\data\tur.pcim', {}),
    'chile':    (r'C:\mfmodeller_raw\chile\pensionsmodel 2\data\chilepens.pcim', {}),
    'dk5':      (r'C:\mfmodeller_raw\tony\DK_5eq\data\dk5_baseline.pcim', {}),
    'frbus var': (r'C:\mfmodeller_raw\frbus_okt_2021\FrbUS\models\frbus_var.pcim',
                  dict(max_iterations=300)),
}

# PAK carbon-tax scenario, as in speedtest.py
PAK_PER = (2020, 2100)
PAK_SCENARIO = "<2020 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30"
# FRB/US MCE: the federal funds rate shock of frbus_mce_test.py
FRBUS_SCENARIO = '<2020q1 2020q1> rffintay_aerr + 1 '
FRBUS_OPTS = dict(nonlin=15, max_iterations=300)

# downstream cell (PAK): fixed spending as in the interactive notebook, and a
# target-instrument problem (GDP 1% above the scenario in 2021-2023, private
# consumption add factor as instrument)
GGEXP = ('PAKGGREVOTHRCN PAKGGEXPCAPTCN PAKGGEXPGNFSCN PAKGGEXPTRNSCN '
         'PAKGGEXPOTHRCN PAKNEGDIFGOVXN PAKBXFSTREMTCD PAKBMFSTREMTCD')
DEKOMP_VAR = 'PAKNYGDPMKTPKN'
INV_TARGET = 'PAKNYGDPMKTPKN'
INV_INSTRUMENT = 'PAKNECONPRVTKN_A'
INV_PER = (2021, 2023)

#%% the cases: label -> build(ng) returning (model, input frame, (start, end) or None, options)


# saved solve options a pcim may carry that this test sets itself
NOT_SAVED = {'solver', 'ljit', 'silent', 'stringjit', 'transpile_reset', 'reset_options'}


def pcim_case(path, scenario=None, per=None, **opts):
    """A model loaded from *path*; *scenario* is an upd string applied to the
    saved frame (None: solve the saved frame), *per* the periods (None: the
    saved ones).  The solve options saved in the file are used (except the
    solver), *opts* on top."""
    def build(ng):
        m, df = model.modelload(str(path), run=0, ng=ng)
        saved = {k: v for k, v in m.oldkwargs.items() if k not in NOT_SAVED}
        return m, (df.upd(scenario) if scenario else df), per, {**saved, **opts}
    return build


SMALL_PER = (2001, 2048)
SMALL_INDEX = range(2000, 2051)
SMALL_MODELS = {
    # recursive: xgenr / xgenr_ng
    'small acyclic': '''
        FRML <> A = 2*B $
        FRML <> C = A + B(-1) $
        FRML <> D = 0.5*C + 0.5*D(-1) $
    ''',
    # simultaneous: sim / sim_ng
    'small simultaneous': '''
        FRML <> C = 10 + 0.3*C(-1) + 0.5*Y $
        FRML <> I = 20 + 0.1*(Y-Y(-1)) $
        FRML <> Y = C + I + G $
    ''',
    # leads: newtonstack / newtonstack_ng
    'small forward': '''
        FRML <> Y = 0.4*Y(-1) + 0.4*Y(+1) + G $
        FRML <> C = 0.6*Y + 0.2*C(+1) $
    ''',
}


def small_case(text):
    """A small model on made-up data: G = 100, a 10% rise from 2010, every
    endogenous variable 100 to start from (and as terminal values)."""
    def build(ng):
        m = model(text, ng=ng)
        df = pd.DataFrame(100.0, index=SMALL_INDEX, columns=sorted(m.allvar))
        df.loc[2010:, [v for v in m.exogene if v in df.columns]] *= 1.1
        return m, df, SMALL_PER, {}
    return build


CASES = {
    'pak':            pcim_case(DATA / 'pak.pcim', PAK_SCENARIO, PAK_PER),
    'frbus mce':      pcim_case(DATA / 'frbusmcetest.pcim', FRBUS_SCENARIO, None, **FRBUS_OPTS),
    **{label: small_case(text) for label, text in SMALL_MODELS.items()},
    **{label: pcim_case(path, None, None, **opts)
       for label, (path, opts) in EXTRA_MODELS.items() if Path(path).exists()},
}
print('cases:', ', '.join(CASES))
print('skipped (file not found):',
      ', '.join(l for l, (p, _) in EXTRA_MODELS.items() if not Path(p).exists()) or 'none')


def call(m, df, per, opts, **extra):
    """One model call without solver=; returns (seconds, result)."""
    span = list(per) if per else []
    if GC_QUIET:
        gc.collect()
        gc.disable()
    try:
        t0 = time.perf_counter()
        res = m(df, *span, **{**opts, **all_opts, 'silent': 1, 'reset_options': True,
                              'ljit': LJIT, **extra})
        return time.perf_counter() - t0, res
    finally:
        if GC_QUIET:
            gc.enable()


def compare(a, b, per):
    """Differences of *b* from *a* over the periods *per* and the columns both
    have: (max rel diff, max abs diff, where, values outside the tolerance).
    Frames that can not be compared give NaN and the reason in *where*."""
    try:
        return _compare(a, b, per)
    except Exception as e:
        return np.nan, np.nan, f'not compared, {type(e).__name__}: {e}'[:60], -1


def _compare(a, b, per):
    cols = [c for c in a.columns if c in b.columns]
    x = a.loc[per, cols].astype(float)
    y = b.loc[per, cols].astype(float)
    absdiff = (y - x).abs()
    rel = absdiff / (x.abs() + 1e-8)
    var = rel.max().idxmax()
    where = f'{var} {rel[var].idxmax()}'
    outside = int((~np.isclose(y.to_numpy(), x.to_numpy(), rtol=RTOL, atol=ATOL,
                               equal_nan=True)).sum())
    return float(rel.max().max()), float(absdiff.max().max()), where, outside


#%% run: every case with ng=False and ng=True
rows, results = [], {}
for label, build in CASES.items():
    row = dict(case=label)
    for ng in (False, True):
        tag = 'ng' if ng else 'legacy'
        try:
            m, df, per, opts = build(ng)
            t_cold, _ = call(m, df, per, opts)
            t_warm, res = call(m, df, per, opts)
        except Exception as e:
            print(f'{label:20s} ng={ng!s:5s} raised {type(e).__name__}: {e}')
            row[f'{tag} solver'] = f'{type(e).__name__}'
            continue
        results[(label, ng)] = (res, m.current_per)
        row.update({f'{tag} solver': m.model_solver.__name__,
                    f'{tag} cold': t_cold, f'{tag} warm': t_warm})
        print(f'{label:20s} ng={ng!s:5s} {m.model_solver.__name__:28s} '
              f'cold {t_cold:8.3f} s  warm {t_warm:8.3f} s')
    if (label, False) in results and (label, True) in results:
        (a, per), (b, _) = results[(label, False)], results[(label, True)]
        rel, absd, where, outside = compare(a, b, per)
        row.update({'max rel diff': rel, 'max abs diff': absd, 'where': where,
                    'outside tol': outside})
    rows.append(row)

table = pd.DataFrame(rows).set_index('case')
with pd.option_context('display.float_format', '{:,.4g}'.format,
                       'display.width', 200, 'display.max_columns', 20):
    print(table)

failed = [r['case'] for r in rows
          if r.get('outside tol', 1) != 0 or not isinstance(r.get('ng warm'), float)]
print(f'\nng=True gives the ng=False solution within rtol={RTOL}, atol={ATOL}: '
      + ('all cases' if not failed else f'NOT for {", ".join(failed)}'))

#%% downstream: features that call the model or read what a solve leaves behind (PAK)
# label: function(model, scenario frame) -> a frame to compare (or None)


def check_solve(m, alt):
    return m(alt, *PAK_PER, silent=1, reset_options=True, ljit=LJIT, **all_opts)


def check_keep(m, alt):
    m(alt, *PAK_PER, silent=1, reset_options=True, ljit=LJIT, keep='carbon tax', **all_opts)
    return m.keep_solutions['carbon tax']


def check_lastdf_basedf(m, alt):
    res = check_solve(m, alt)
    assert m.lastdf.equals(res), 'lastdf is not the last solution'
    return m.basedf


def check_dekomp(m, alt):
    check_solve(m, alt)
    return m.dekomp(DEKOMP_VAR, *PAK_PER, lprint=False).att_level


def add_factor_solver(m):
    """The solver the add-factor model used last."""
    am = m.calc_add_factor_model
    return getattr(getattr(am, 'model_solver', None), '__name__', 'not run')


def check_fix(m, alt):
    """Fixed spending: the dummies are set, so the call also runs the
    add-factor model."""
    res = m(m.fix(alt, GGEXP, *PAK_PER, silent=1), *PAK_PER, silent=1,
            reset_options=True, ljit=LJIT, **all_opts)
    if m.split_calc_add_factor:
        print(f'    ng={m.ng}: add-factor model ng={m.calc_add_factor_model.ng}, '
              f'solver {add_factor_solver(m)}')
    return res


def check_invert(m, alt):
    if INV_INSTRUMENT not in m.exogene:
        raise ValueError(f'{INV_INSTRUMENT} is not exogenous in {m.name}, change INV_INSTRUMENT')
    sol = check_solve(m, alt)
    targets = sol.loc[INV_PER[0]:INV_PER[1], [INV_TARGET]] * 1.01
    return m.invert(sol, targets, [INV_INSTRUMENT], progressbar=False)


def check_add_factor_model(m, alt):
    """The add factors of the scenario solution, calculated directly with the
    add-factor model (no fixing needed)."""
    if not m.split_calc_add_factor:
        print('    PAK has no CALC_ADD_FACTOR equations, nothing to check')
        return None
    sol = check_solve(m, alt)
    am = m.calc_add_factor_model
    res = am(sol, *PAK_PER, silent=1, reset_options=True, ljit=LJIT, **all_opts)
    print(f'    ng={m.ng}: add-factor model ng={am.ng}, solver {add_factor_solver(m)}')
    return res


DOWNSTREAM = {
    'solve':            check_solve,
    'keep':             check_keep,
    'lastdf / basedf':  check_lastdf_basedf,
    'dekomp':           check_dekomp,
    'fix':              check_fix,
    'invert':           check_invert,
    'add-factor model': check_add_factor_model,
}

down_rows, down_res = [], {}
for label, check in DOWNSTREAM.items():
    row = dict(feature=label)
    for ng in (False, True):
        tag = 'ng' if ng else 'legacy'
        # a fresh model for every check: nothing left by an earlier solve
        m, baseline = model.modelload(str(DATA / 'pak.pcim'), run=0, ng=ng)
        alt = baseline.upd(PAK_SCENARIO)
        try:
            down_res[(label, ng)] = check(m, alt)
            row[tag] = f'ok ({getattr(getattr(m, "model_solver", None), "__name__", "-")})'
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'.replace('\n', ' ')
            row[tag] = msg if len(msg) <= 100 else f'{msg[:40]} ... {msg[-55:]}'
    a, b = down_res.get((label, False)), down_res.get((label, True))
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        per = a.index.intersection(b.index)
        rel, absd, where, outside = compare(a, b, per)
        row.update({'max rel diff': rel, 'where': where, 'outside tol': outside})
    down_rows.append(row)
    print(f'{label:18s} legacy: {row["legacy"]:40s} ng: {row["ng"]}')

with pd.option_context('display.float_format', '{:,.4g}'.format,
                       'display.width', 250, 'display.max_columns', 10,
                       'display.max_colwidth', 100):
    print(pd.DataFrame(down_rows).set_index('feature'))
