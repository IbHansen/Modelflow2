"""ODE equations in ModelFlow: a preprocessor from laws of motion to Runge-Kutta FRML text, and the N-body problem.

An overlay: it builds ordinary ModelFlow models and needs no change in the core.

ODE -> Runge-Kutta
------------------
Reads a model whose state equations are either

    FRML <ODE> S = f $                      the derivative dS/dt = f, or
    FRML <>    S = S(-1) + DT*f $           an Euler step (explicit or semi-implicit)

and returns FRML text for the same ODE system integrated with a one-step method given by a
Butcher tableau (euler, heun, midpoint, ralston, rk4). Every generated equation is recursive
within the period, so ModelFlow solves one time step per row in a single ordered sweep.

Rules (see experiments/projecttile/nbody_rk4_design.md):
- inside f and inside the auxiliaries needed by f, X and X(-1) both mean "X at the current stage";
  the lag pattern of an Euler model is discarded. A lag of 2 or more, a lead or a fixed period is an error.
- DT is the name of a dataframe column (default 'DT'); it may vary by row.
- an exogenous variable E is read at the stage time c_i: c=0 -> E(-1), c=1 -> E, else interpolated.
- auxiliaries not reachable from any derivative are kept once, verbatim (reporting only).

    text = ode_transform(frml_text, method='rk4', dt='DT')
    m = model(text)

or, to use Makemodel's doable lines and markdown input, through the overlay wrapper, which runs
exclusion_lists before Makemodel and ode_transform on Makemodel's expanded text:

    m, mm = ode_makemodel(text, method='rk4', type_input='modelflow')

NO{list} inside a DO, a doable line or a sum means 'all the other elements' (exclusion_lists).
A LIST of dimensions (LIST DIMS = DIMS : X Y Z) with DO DIMS / sum(DIMS, ...) writes the coordinate
equations once; see nbody_template.

N-body
------
The laws of motion for point masses written once as a documented markdown model (nbody_template),
the model builders nbody_makemodel / nbody_model, a dataframe builder (make_df) and measurements
(zero_momentum, energy, orbital_period).

Variable names follow Makemodel's doable convention name__{index}: POS__EARTH__X, V__EARTH__X, M__EARTH.
"""
import re
from fractions import Fraction

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp

import modelmanipulation as mp      # before modelpattern: the two import each other, and only this order works cold
import modelpattern as pt

# =====================================================================================================
# ODE -> Runge-Kutta
# =====================================================================================================

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


# LIST L = L : a b c [/ sub : ...]  ended by $ or by the end of the line (Makemodel style); an optional
# leading '>' is a markdown model line, which generated lines then get too
_LIST_PAT = re.compile(r'^([ \t]*>?[ \t]*)(LIST\s+(\w+)\s*=\s*(\w+)\s*:\s*([^/$\n]*))((?:/[^$\n]*)?)(\$?)[ \t]*$',
                       re.IGNORECASE | re.MULTILINE)
# DOABLE [<options>] lhs = rhs  on one line, Makemodel style
_DOABLE_PAT = re.compile(r'^([ \t]*>?[ \t]*)DOABLE\s+(<[^>]*>)?\s*([^=\n]+?)\s*=\s*(.+?)\s*(\$?)[ \t]*$',
                         re.IGNORECASE | re.MULTILINE)


def _exclusion_display(prefix, head, lname, elems, subs, items, no_names, end):
    """The list with its NO sublist and the exclusion lists, column aligned, for reading:

        >list bodies = bodies : SUN   EARTH   MOON   /
        >              NO     : NOSUN NOEARTH NOMOON
        >list NOSUN   = NOSUN   : EARTH MOON
        ...
    """
    left = head[:len(head) - len(elems)]                 # 'list bodies = bodies : '
    colon = left.rindex(':')
    first_at = left[:colon].rstrip().rfind(' ') + 1      # where the first sublist name starts
    name_w = max(colon - first_at, len('NO '))           # sublist-name column, wide enough for 'NO '
    left = left[:first_at] + left[first_at:colon].rstrip().ljust(name_w) + left[colon:]
    colon = first_at + name_w
    widths = [max(len(e), len(n)) for e, n in zip(items, no_names)]
    row1 = left + ' '.join(e.ljust(w) for e, w in zip(items, widths)) + ' /'
    row2 = (' ' * first_at + 'NO'.ljust(colon - first_at) + left[colon:]
            + ' '.join(n.ljust(w) for n, w in zip(no_names, widths))).rstrip()
    if subs.strip():
        row2 += ' ' + subs.strip()
    keyword = head.split()[0]                            # 'list' or 'LIST', as the author wrote it
    nw = max(len(n) for n in no_names)
    rows = [row1, row2 + end] + [f'{keyword} {n.ljust(nw)} = {n.ljust(nw)} : {" ".join(o for o in items if o != e)}{end}'
                                 for n, e in zip(no_names, items)]
    out = '\n'.join(prefix + r for r in rows)
    if prefix:              # markdown model: explain in prose under the list block
        others = ' '.join(o for o in items if o != items[0])
        out += (f'\n\n*Generated from the list above:* the sublist `NO` gives each element of `{lname}` the name of '
                f'the list of all the other elements, e.g. `{no_names[0]}` = {others}. In an equation that loops '
                f'over `{lname}`, `NO{{{lname}}}` is that list and `{{NO{{{lname}}}}}` runs over its elements.\n')
    return out


def exclusion_lists(text, display=False):
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

    display=True gives the text for reading instead (mm.render): the equations stay as written, with NO{L},
    and the list line is replaced by the list with its NO sublist and the exclusion lists, column aligned,
    followed in a markdown model by a short note on what NO{L} means.
    """
    # Makemodel: either every statement ends with $ or none does - keep the style of the input. Judged on
    # the model lines only ('>' lines in a markdown model), since LaTeX documentation is full of $ signs
    lines = text.splitlines()
    model_lines = [l for l in lines if l.lstrip().startswith('>')] or lines
    end = ' $' if any(l.rstrip().endswith('$') for l in model_lines) else ''
    for m in list(_LIST_PAT.finditer(text)):
        whole = m.group(0)
        prefix, head, lname, first, elems, subs, dollar = m.groups()
        prefix = prefix.strip(' \t')          # '>' for a markdown model line, else ''
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
        taken = {mm.group(3).upper() for mm in _LIST_PAT.finditer(text)}
        if clash := [n for n in no_names if n.upper() in taken]:
            raise OdeTransformError(f'exclusion list name(s) already defined: {" ".join(clash)}')
        if display:
            text = text.replace(whole, _exclusion_display(prefix, head, lname, elems, subs, items, no_names, end), 1)
            continue
        new_lists = [f'{prefix}LIST {n} = {n} : {" ".join(o for o in items if o != e)}{end}' for n, e in zip(no_names, items)]
        new_stmt = f'{prefix}{head.rstrip()} / NO : {" ".join(no_names)} {subs.strip()}'.rstrip() + end

        def doable_pair(dm):
            pre, opts, lhs, rhs, _ = dm.groups()
            if not nested.search(lhs):
                return dm.group(0)
            p = pre.strip(' \t')
            return (f'{p}DO {lname}{end}\n{p}  DO NO{{{lname}}}{end}\n{p}    FRML {opts or "<>"} {lhs} = {rhs}{end}\n'
                    f'{p}  ENDDO{end}\n{p}ENDDO{end}')
        text = _DOABLE_PAT.sub(doable_pair, text)
        text = text.replace(whole, new_stmt + '\n' + '\n'.join(new_lists), 1)
        text = nested.sub('{{{NO}}}', text)
        text = bare.sub('{NO}', text)
    return text


def _in_notebook():
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False


def _show_report(rows, exo, dt, method, report):
    """The classification of the equations: a markdown table in a notebook, plain text otherwise
    (report='text' forces text, report='markdown' forces the table)."""
    method_name = method if isinstance(method, str) else 'own tableau'
    as_markdown = report == 'markdown' or (report is True and _in_notebook())
    if not as_markdown:
        print(f'{"variable":30} {"class":8} {"derivative / expression"}')
        for lhs, kind, rhs in rows:
            print(f'{lhs:30} {kind:8} {rhs}')
        print(f'exogenous: {" ".join(exo)}   step: {dt}   method: {method_name}')
        return
    from html import escape
    from IPython.display import Markdown, display
    footer_md = (f'**Exogenous:** {", ".join(f"`{e}`" for e in exo)} &nbsp; **Step:** `{dt}` &nbsp; '
                 f'**Method:** {method_name}')
    legend = ('<i>state</i>: integrated, the column is its derivative; <i>aux</i>: needed by a derivative, '
              'recomputed at every stage; <i>report</i>: needed by no derivative, computed once at the end of the step.')
    lines = ['| variable | class | derivative / expression |', '|:--|:--|:--|']
    lines += [f'| `{lhs}` | {kind} | `{rhs}` |' for lhs, kind, rhs in rows]
    lines += ['', footer_md, '', legend.replace('<i>', '*').replace('</i>', '*')]

    # HTML with the borders inline on every cell: JupyterLab's stylesheet draws markdown tables without grid
    # lines and inline styles win over it. Notebooks render the HTML; latex/pdf exports use the markdown.
    cell = 'border:1px solid #999;padding:3px 8px;text-align:left;vertical-align:top'
    mono = cell + ';font-family:monospace;white-space:nowrap'
    html = ['<table style="border-collapse:collapse;margin:4px 0">',
            f'<tr><th style="{cell}">variable</th><th style="{cell}">class</th>'
            f'<th style="{cell}">derivative / expression</th></tr>']
    html += [f'<tr><td style="{mono}">{escape(lhs)}</td><td style="{cell}">{kind}</td>'
             f'<td style="{mono}">{escape(rhs)}</td></tr>' for lhs, kind, rhs in rows]
    html += ['</table>',
             f'<p><b>Exogenous:</b> <code>{"</code>, <code>".join(escape(e) for e in exo)}</code> &nbsp; '
             f'<b>Step:</b> <code>{escape(dt)}</code> &nbsp; <b>Method:</b> {escape(method_name)}</p>',
             f'<p>{legend}</p>']

    class _Table(Markdown):
        def _repr_html_(self):
            return '\n'.join(html)

        def __repr__(self):
            return str(self.data)

    display(_Table('\n'.join(lines)))


def ode_transform(frml_text, method='rk4', dt='DT', funks=[], report=False):
    """FRML text of the model integrated with `method` (a key of BUTCHER or a dict a/b/c).

    report=True shows how each equation was classified (state / aux / report) and the derivatives: a markdown
    table in a notebook, plain text elsewhere; report='text' or report='markdown' chooses explicitly."""
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
        rows = [(eq.lhs, eq.kind, _retext(eq.f_tokens, lambda v, lag: v) if eq.kind == 'state' else eq.rhs)
                for eq in eqs]
        _show_report(rows, sorted(allvars - endo - {dt}), dt, method, report)

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


def ode_makemodel(text, method='rk4', dt='DT', funks=[], report=False, modelname=None, **makemodel_kwargs):
    """Overlay on Makemodel: exclusion_lists -> Makemodel (doable, lists, DO, sum) -> ode_transform -> model.

    Returns (model, makemodel). Pass Makemodel arguments through, e.g. type_input='modelflow' for a plain
    string of equations (the default reads markdown with '>' lines). The Makemodel keeps the untransformed,
    expanded equations (mm.expanded_frml, mm.render); the model has the Runge-Kutta equations.
    """
    from modelclass import model
    from modelconstruct_estimation import Makemodel, mfmod_list_to_codeblock
    mm = Makemodel(exclusion_lists(text), funks=funks, **makemodel_kwargs)
    if mm.markdown_model:
        # mm.render shows the equations as the author wrote them (NO{bodies}), with the generated sublist and
        # exclusion lists in place of the list line, so NO{bodies} can be read off the rendered model
        mm.markdown_model = mfmod_list_to_codeblock(exclusion_lists(text, display=True))
    rk_text = ode_transform(mm.expanded_frml, method=method, dt=dt, funks=funks, report=report)
    return model(rk_text, modelname=modelname or f'ODE model, {method}', funks=funks), mm

# =====================================================================================================
# N-body problem
# =====================================================================================================

# Gaussian gravitational constant squared: AU^3 / (solar mass * day^2)
G_AU_DAY = 2.959122082855911e-4
MU_EARTH_KM = 398600.4418        # km^3/s^2
R_EARTH_KM = 6371.0


def pos(body, coord):
    return f'POS__{body}__{coord}'


def vel(body, coord):
    return f'V__{body}__{coord}'


def mass(body):
    return f'M__{body}'


def nbody_template(bodies, dims=2):
    r"""The laws of motion for point masses, as a documented markdown model. Only the two list lines
    depend on the problem; `nbody_model` turns the text into a Runge-Kutta model."""
    return rf'''# {len(bodies)} bodies under gravity

Newton's laws of motion for point masses $i$ with masses $m_i$ at positions $\mathbf x_i$:

$$
\frac{{d\mathbf x_i}}{{dt}} = \mathbf v_i, \qquad
\frac{{d\mathbf v_i}}{{dt}} = G \sum_{{j \ne i}} m_j \, \frac{{\mathbf x_j - \mathbf x_i}}{{|\mathbf x_j - \mathbf x_i|^3}}
$$

The bodies and the coordinates are lists. `doable` writes one equation per body $i$ and coordinate $d$
(the loop indices are the `__{{...}}` parts of the left-hand side), `NO{{bodies}}` means *all the other
bodies* $j \ne i$, and `<ODE>` marks a derivative: the integrator (Euler, Heun, RK4) is chosen when the
model is built, not here.

>list bodies = bodies : {' '.join(bodies)}
>list dims   = dims : {' '.join('XYZ'[:dims])}
>list dd     = dd : {' '.join('XYZ'[:dims])}

`dd` is a second copy of the coordinates: the summation index $d'$ of the distance below, which must differ
from the coordinate $d$ of the equation it sits in.

**Position.** $\frac{{dx_{{i,d}}}}{{dt}} = v_{{i,d}}$

>doable <ODE> pos__{{bodies}}__{{dims}} = v__{{bodies}}__{{dims}}

**Velocity.** $\frac{{dv_{{i,d}}}}{{dt}} = G \sum_{{j \ne i}} m_j \, (x_{{j,d}} - x_{{i,d}}) \, / \, r_{{ij}}^3$
with the distance written inline, $r_{{ij}}^3 = \bigl(\sum_{{d'}} (x_{{j,d'}} - x_{{i,d'}})^2\bigr)^{{3/2}}$

>doable <ODE> v__{{bodies}}__{{dims}} = G*sum(NO{{bodies}}, m__{{NO{{bodies}}}}*(pos__{{NO{{bodies}}}}__{{dims}} - pos__{{bodies}}__{{dims}})
>>   / (sum(dd, (pos__{{NO{{bodies}}}}__{{dd}} - pos__{{bodies}}__{{dd}})**2))**1.5)

**Time**, for plotting. $\frac{{dT}}{{dt}} = 1$

><ODE> t = 1
'''


def nbody_makemodel(bodies, dims=2, method='rk4', modelname=None):
    """(model, Makemodel): the Runge-Kutta model and the documented source (mm.render shows it)."""
    return ode_makemodel(nbody_template(bodies, dims), method=method,
                         modelname=modelname or f'{len(bodies)}-body, {method}')


def nbody_model(bodies, dims=2, method='rk4', modelname=None):
    return nbody_makemodel(bodies, dims, method, modelname)[0]


def make_df(m, bodies, masses, pos0, vel0, g, dt, t_max, t0=0.0):
    """One row per step; period 0 holds the initial state. pos0/vel0: {body: (x, y[, z])}."""
    n = int(round(t_max / dt))
    df = pd.DataFrame(0.0, index=range(n + 1), columns=sorted(m.allvar))
    df['DT'], df['G'] = dt, g
    df.loc[0, 'T'] = t0
    for B in bodies:
        df[mass(B)] = masses[B]
        cs = 'XYZ'[:len(pos0[B])]
        df.loc[0, [pos(B, C) for C in cs]] = pos0[B]
        df.loc[0, [vel(B, C) for C in cs]] = vel0[B]
    return df


def zero_momentum(bodies, masses, vel0, rest_body):
    """Velocity of rest_body so that the total momentum is zero (keeps the system from drifting)."""
    p = sum(masses[B] * np.array(vel0[B]) for B in bodies if B != rest_body)
    return {**vel0, rest_body: tuple(-p / masses[rest_body])}


def energy(res, bodies, masses, g, dims=2):
    cs = 'XYZ'[:dims]
    kin = sum(0.5 * masses[B] * sum(res[vel(B, C)] ** 2 for C in cs) for B in bodies)
    pot = 0
    for i, p in enumerate(bodies):
        for q in bodies[i + 1:]:
            r = np.sqrt(sum((res[pos(q, C)] - res[pos(p, C)]) ** 2 for C in cs))
            pot = pot - g * masses[p] * masses[q] / r
    return kin + pot


def orbital_period(res, body, centre=None):
    """Mean orbital period of body around centre (or the origin) over the run:
    time span / number of turns, from the unwrapped angle."""
    x = res[pos(body, 'X')] - (res[pos(centre, 'X')] if centre else 0)
    y = res[pos(body, 'Y')] - (res[pos(centre, 'Y')] if centre else 0)
    ang = np.unwrap(np.arctan2(y, x).to_numpy())
    t = res['T'].to_numpy()
    return 2 * np.pi * (t[-1] - t[0]) / abs(ang[-1] - ang[0])
