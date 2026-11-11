"""ODE -> Runge-Kutta preprocessor for ModelFlow models.

Reads a model whose state equations are either

    FRML <ODE> S = f $                      the derivative dS/dt = f, or
    FRML <>    S = S(-1) + DT*f $           an Euler step (explicit or semi-implicit)

and returns FRML text for the same ODE system integrated with a one-step method given by a
Butcher tableau (euler, heun, midpoint, ralston, rk4). Every generated equation is recursive
within the period, so ModelFlow solves one time step per row in a single ordered sweep.

Rules (see nbody_rk4_design.md):
- inside f and inside the auxiliaries needed by f, X and X(-1) both mean "X at the current stage";
  the lag pattern of an Euler model is discarded. A lag of 2 or more, a lead or a fixed period is an error.
- DT is the name of a dataframe column (default 'DT'); it may vary by row.
- an exogenous variable E is read at the stage time c_i: c=0 -> E(-1), c=1 -> E, else interpolated.
- auxiliaries not reachable from any derivative are kept once, verbatim (reporting only).

    text = ode_transform(frml_text, method='rk4', dt='DT')
    m = model(text)

or through Makemodel, which runs exclusion_lists before its doable/DO expansion and ode_transform after the
sums are expanded:

    mm = Makemodel(text, type_input='modelflow', scheme='rk4')     # or %%Makemymodel name scheme=rk4
    m = mm.mmodel                                                   # mm.showpost_ode prints the generated text

NO{list} inside a DO, a doable line or a sum means 'all the other elements' (exclusion_lists).
"""
import re
from fractions import Fraction

import networkx as nx
import sympy as sp

import modelpattern as pt
import modelmanipulation as mp

BUTCHER = {
    'euler':    dict(a=[[]],                              b=[1],                      c=[0]),
    'heun':     dict(a=[[], [1]],                         b=[0.5, 0.5],               c=[0, 1]),
    'midpoint': dict(a=[[], [0.5]],                       b=[0, 1],                   c=[0, 0.5]),
    'ralston':  dict(a=[[], [Fraction(2, 3)]],            b=[0.25, 0.75],             c=[0, Fraction(2, 3)]),
    'rk4':      dict(a=[[], [0.5], [0, 0.5], [0, 0, 1]],  b=[Fraction(1, 6), Fraction(1, 3),
                                                             Fraction(1, 3), Fraction(1, 6)], c=[0, 0.5, 0.5, 1]),
}

_FUNK_OP = re.compile(r'^[A-Z_][A-Z0-9_]*$')      # an op token that is a function name (SQRT, EXP, ...)


class OdeTransformError(Exception):
    pass


def _coef(x):
    """A tableau coefficient as model text: 1 -> '', 0.5 -> '0.5*', Fraction(1, 6) -> '(1/6)*'."""
    if isinstance(x, Fraction):
        return '' if x == 1 else f'({x.numerator}/{x.denominator})*'
    return '' if x == 1 else f'{x!r}*'


def _stage_time(c):
    return c if isinstance(c, Fraction) else Fraction(c).limit_denominator(1000)


class _Eq:
    """One parsed equation: name options, lhs, rhs tokens."""

    def __init__(self, frml, funks):
        _, _, self.frmlname, expr = pt.split_frml(frml)
        expr = expr.strip().rstrip('$').strip()
        lhs, rhs = expr.split('=', 1)
        self.lhs = lhs.strip().upper()
        self.rhs = rhs.strip()
        self.frml = frml.strip()
        self.tokens = pt.udtryk_parse(self.rhs, funks=funks)
        self.vars = {t.var for t in self.tokens if t.var}
        self.kind = None          # 'state' | 'aux' | 'report'
        self.f_tokens = None      # derivative tokens for a state


def _retext(tokens, var_map):
    """Tokens back to text, with variables replaced through var_map(var, lag) -> text."""
    out = []
    for t in tokens:
        if t.var:
            out.append(var_map(t.var, t.lag))
        elif t.op:
            out.append(t.op)
        else:
            out.append(t.number)
    return ''.join(out)


def _check_lags(eq, errors):
    for t in eq.tokens:
        if t.var and t.lag not in ('', '-1'):
            errors.append(f'lag/lead {t.var}({t.lag}) is not allowed (only X and X(-1); no delays):\n   {eq.frml}')
            return


def _sympy_of(tokens, dt):
    """Sympy expression of the tokens with lags collapsed (X(-1) -> X). Functions stay undefined."""
    syms = {t.var: sp.Symbol(t.var) for t in tokens if t.var}
    funcs = {t.op: sp.Function(t.op) for t in tokens if t.op and _FUNK_OP.match(t.op)}
    text = _retext(tokens, lambda v, lag: v)
    try:
        return sp.parse_expr(text, local_dict={**syms, **funcs}), syms
    except Exception as e:
        raise OdeTransformError(f'cannot parse for ODE detection: {text}\n   {e}')


def _classify(eqs, dt, funks, errors):
    """Mark each equation state / aux. For Euler-form states extract f with sympy."""
    DT = sp.Symbol(dt)
    for eq in eqs:
        if re.search(r'\bODE\b', eq.frmlname.upper()):
            eq.kind = 'state'
            if dt in eq.vars:
                errors.append(f'{dt} inside an <ODE> equation (write the derivative, not a step):\n   {eq.frml}')
            eq.f_tokens = eq.tokens
            continue
        if dt not in eq.vars:
            eq.kind = 'aux'
            if eq.lhs in eq.vars:
                errors.append(f'auxiliary {eq.lhs} refers to itself; with lags collapsed it is simultaneous. '
                              f'If it is a state write it as {eq.lhs} = {eq.lhs}(-1) + {dt}*f or tag it <ODE>:\n   {eq.frml}')
            continue
        # DT present: must be S = S(-1) + DT*f
        expr, syms = _sympy_of(eq.tokens, dt)
        S = syms.get(eq.lhs)
        if S is None:
            errors.append(f'{dt} in an equation that is not a step {eq.lhs} = {eq.lhs}(-1) + {dt}*f:\n   {eq.frml}')
            continue
        g = sp.cancel(sp.expand((expr - S) / DT))
        if g.has(DT):
            errors.append(f'{dt} appears other than as {eq.lhs}(-1) + {dt}*f:\n   {eq.frml}')
            continue
        eq.kind = 'state'
        ftext = sp.sstr(g, full_prec=False)
        eq.f_tokens = pt.udtryk_parse(ftext, funks=funks)


# LIST L = L : a b c [/ sub : ...]  ended by $ or by the end of the line (Makemodel style)
_LIST_PAT = re.compile(r'^[ \t]*(LIST\s+(\w+)\s*=\s*(\w+)\s*:\s*([^/$\n]*))((?:/[^$\n]*)?)(\$?)[ \t]*$',
                       re.IGNORECASE | re.MULTILINE)
# DOABLE [<options>] lhs = rhs  on one line, Makemodel style
_DOABLE_PAT = re.compile(r'^[ \t]*DOABLE\s+(<[^>]*>)?\s*([^=\n]+?)\s*=\s*(.+?)\s*(\$?)[ \t]*$',
                         re.IGNORECASE | re.MULTILINE)


def exclusion_lists(text):
    """Make 'all the other elements' writable in the DSL.

    For a list L used as NO{L} in the text - inside a DO over L, or in a Makemodel DOABLE line - this

    1. appends a sublist  / NO : NOA NOB NOC  to  LIST L = L : A B C
    2. adds one list per element with the other elements:  LIST NOA = NOA : B C  etc.
    3. rewrites the nested spelling, which format_map cannot take, into the escaped form that works:
       {NO{L}} -> {{{NO}}}   (the element of the exclusion list)
       NO{L}   -> {NO}       (the list name, in DO NO{L} and sum(NO{L}, ...))
    After the outer DO over L these read {NOA} and NOA, and the inner DO / sum expand them.
    4. a DOABLE line with {NO{L}} on its left-hand side (one equation per pair, e.g. a distance) becomes an
       explicit  DO L $ DO NO{L} $ FRML ... $ ENDDO $ ENDDO $ , since Makemodel's doable cannot loop over a
       list whose name comes from the outer loop. Other DOABLE lines are left to Makemodel.
    """
    # Makemodel: either every statement ends with $ or none does - keep the style of the input
    end = ' $' if '$' in text else ''
    for m in list(_LIST_PAT.finditer(text)):
        whole = m.group(0)
        head, lname, first, elems, subs, dollar = m.groups()
        nested = re.compile(r'\{NO\{' + re.escape(lname) + r'\}\}', re.IGNORECASE)
        bare = re.compile(r'(?<![{\w])NO\{' + re.escape(lname) + r'\}', re.IGNORECASE)
        if not (nested.search(text) or bare.search(text)):
            continue
        if lname.upper() != first.upper():
            raise OdeTransformError(f'NO{{{lname}}}: the first sublist must be the list name itself: {whole}')
        if re.search(r'/\s*NO\s*:', subs, re.IGNORECASE):
            raise OdeTransformError(f'list {lname} already has a sublist NO: {whole}')
        items = elems.split()
        no_names = [f'NO{e}' for e in items]
        taken = {mm.group(2).upper() for mm in _LIST_PAT.finditer(text)}
        if clash := [n for n in no_names if n.upper() in taken]:
            raise OdeTransformError(f'exclusion list name(s) already defined: {" ".join(clash)}')
        new_lists = [f'LIST {n} = {n} : {" ".join(o for o in items if o != e)}{end}' for n, e in zip(no_names, items)]
        new_stmt = f'{head.rstrip()} / NO : {" ".join(no_names)} {subs.strip()}'.rstrip() + end

        def doable_pair(dm):
            opts, lhs, rhs, _ = dm.groups()
            if not nested.search(lhs):
                return dm.group(0)
            return (f'DO {lname}{end}\n  DO NO{{{lname}}}{end}\n    FRML {opts or "<>"} {lhs} = {rhs}{end}\n'
                    f'  ENDDO{end}\nENDDO{end}')
        text = _DOABLE_PAT.sub(doable_pair, text)
        text = text.replace(whole, new_stmt + '\n' + '\n'.join(new_lists), 1)
        text = nested.sub('{{{NO}}}', text)
        text = bare.sub('{NO}', text)
    return text


def ode_transform(frml_text, method='rk4', dt='DT', funks=[], report=False):
    """FRML text of the model integrated with `method` (a key of BUTCHER or a dict a/b/c)."""
    tab = BUTCHER[method.lower()] if isinstance(method, str) else method
    a, b, c = tab['a'], tab['b'], [_stage_time(ci) for ci in tab['c']]
    nst = len(b)
    dt = dt.upper()

    expanded = mp.explode(exclusion_lists(frml_text))
    eqs = [_Eq(f, funks) for f in pt.find_frml(expanded)]
    errors = []
    lhs_seen = {}
    for eq in eqs:
        if eq.lhs in lhs_seen:
            errors.append(f'{eq.lhs} is determined twice:\n   {lhs_seen[eq.lhs].frml}\n   {eq.frml}')
        lhs_seen[eq.lhs] = eq
    _classify(eqs, dt, funks, errors)
    states = [eq for eq in eqs if eq.kind == 'state']
    auxs = [eq for eq in eqs if eq.kind == 'aux']
    state_names = {eq.lhs for eq in states}
    aux_names = {eq.lhs for eq in auxs}
    endo = state_names | aux_names
    if not states:
        errors.append('no state equation found (S = S(-1) + DT*f or <ODE> S = f)')

    # which auxiliaries does some derivative need (backwards through the aux graph)?
    g = nx.DiGraph()
    g.add_nodes_from(aux_names)
    for eq in auxs:
        for v in eq.vars & aux_names:
            g.add_edge(v, eq.lhs)                      # v is used by eq.lhs
    needed = set()
    for eq in states:
        for v in {t.var for t in eq.f_tokens if t.var} & aux_names:
            needed |= {v} | nx.ancestors(g, v)
    cycles = [cyc for cyc in nx.simple_cycles(g.subgraph(needed))]
    if cycles:
        errors.append('auxiliaries needed by a derivative are simultaneous: ' +
                      ', '.join(' -> '.join(cyc) for cyc in cycles))
    for eq in auxs:
        eq.kind = 'aux' if eq.lhs in needed else 'report'
        if eq.kind == 'aux':
            _check_lags(eq, errors)
    for eq in states:
        _check_lags(eq, errors)
        if dt in {t.var for t in eq.f_tokens if t.var}:
            errors.append(f'{dt} inside the derivative of {eq.lhs}:\n   {eq.frml}')

    # name collisions
    allvars = set().union(*(eq.vars for eq in eqs)) | endo
    suffixes = [f'__S{i}' for i in range(1, nst + 1)] + [f'__K{i}' for i in range(1, nst + 1)]
    clash = [v + s for v in endo for s in suffixes if v + s in allvars]
    if clash:
        errors.append('generated names already exist: ' + ' '.join(clash))

    if errors:
        print('ode_transform: the model is not an ODE system that can be transformed:')
        for e in errors:
            print(' *', e)
        raise OdeTransformError(f'{len(errors)} problem(s), see above')

    if report:
        print(f'{"variable":30} {"class":8} {"derivative / expression"}')
        for eq in eqs:
            rhs = _retext(eq.f_tokens, lambda v, lag: v) if eq.kind == 'state' else eq.rhs
            print(f'{eq.lhs:30} {eq.kind:8} {rhs}')
        exo = sorted(allvars - endo - {dt})
        print(f'exogenous: {" ".join(exo)}   step: {dt}   method: {method}')

    by_name = {eq.lhs: eq for eq in auxs}
    needed_order = [by_name[n] for n in nx.topological_sort(g.subgraph(needed))]

    def stage_map(i):
        ci = c[i - 1]
        def f(v, lag):
            if v in state_names or v in needed:
                return f'{v}__S{i}'
            if ci == 0:
                return f'{v}(-1)'
            if ci == 1:
                return v
            return f'((1-{float(ci)!r})*{v}(-1)+{float(ci)!r}*{v})'
        return f

    out = [f'£ generated by ode_transform, method {method}, step {dt}']
    for i in range(1, nst + 1):
        out.append(f'£ stage {i}')
        for eq in states:
            terms = [f'{_coef(aij)}{eq.lhs}__K{j + 1}' for j, aij in enumerate(a[i - 1]) if aij]
            rhs = f'{eq.lhs}(-1)' + (f' + {dt}*({" + ".join(terms)})' if terms else '')
            out.append(f'FRML <> {eq.lhs}__S{i} = {rhs} $')
        for eq in needed_order:
            out.append(f'FRML <> {eq.lhs}__S{i} = {_retext(eq.tokens, stage_map(i))} $')
        for eq in states:
            out.append(f'FRML <> {eq.lhs}__K{i} = {_retext(eq.f_tokens, stage_map(i))} $')
    out.append('£ update')
    for eq in states:
        terms = [f'{_coef(bi)}{eq.lhs}__K{i + 1}' for i, bi in enumerate(b) if bi]
        out.append(f'FRML <> {eq.lhs} = {eq.lhs}(-1) + {dt}*({" + ".join(terms)}) $')
    out.append('£ auxiliaries at the end of the step, and reporting equations')
    for eq in auxs:
        if eq.kind == 'aux':
            out.append(f'FRML <> {eq.lhs} = {_retext(eq.tokens, lambda v, lag: v)} $')
        else:   # reporting only: lags kept as written
            out.append(f'FRML {eq.frmlname} {eq.lhs} = {_retext(eq.tokens, lambda v, lag: v + (f"({lag})" if lag else ""))} $')
    return '\n'.join(out)
