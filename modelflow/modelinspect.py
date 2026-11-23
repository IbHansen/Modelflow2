# -*- coding: utf-8 -*-
"""
Inspection and report helpers shared by the JupyterLab Model Workbench
(``mf_workbench``) and the ModelFlow MCP server.

Everything here is a thin layer over standard ModelFlow methods
(``vlist``, ``totgraph``, ``get_att``, ``table``, ``plot``, ``reports``,
``keep_solutions``) and returns plain data - DataFrames, dicts, strings or
file paths - so a notebook panel and an AI assistant see the same thing.

Scenarios are always ``model.keep_solutions``. A scenario is created by
solving with ``keep='name'``, exactly as in a notebook.
"""

from __future__ import annotations

import base64
import contextlib
import html
import io
import math
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------
# Finding models
# --------------------------------------------------------------------------

def find_models(namespace):
    """Return {name: instance} for the ModelFlow models in a namespace
    (for instance IPython's ``user_ns``).

    The class is taken from ``sys.modules`` rather than imported: if a
    model instance exists, modelclass is already imported, and if it is
    not there is nothing to find.
    """
    import sys

    model_cls = getattr(sys.modules.get('modelclass'), 'model', None)
    if model_cls is None:
        return {}
    return {name: obj for name, obj in namespace.items()
            if not name.startswith('_') and isinstance(obj, model_cls)}


def descriptions(m):
    return getattr(m, 'var_description', {}) or {}


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

def dependencies(m, var, up=1, down=1):
    """The local dependency graph around ``var`` from ``m.totgraph``.

    Returns {'var', 'nodes': [{'name', 'kind', 'desc'}], 'edges': [(from, to)],
    'up': [...], 'down': [...]} where up/down are the direct inputs and users.
    """
    import networkx as nx

    var = var.strip().upper()
    g = m.totgraph
    if var not in g:
        raise KeyError(f'{var} is not in the dependency graph')
    keep = {var}
    keep |= set(nx.single_source_shortest_path_length(
        g.reverse(copy=False), var, cutoff=int(up)))
    keep |= set(nx.single_source_shortest_path_length(g, var, cutoff=int(down)))
    sub = g.subgraph(keep)
    desc = descriptions(m)
    endo = set(m.endogene)
    nodes = [{'name': n, 'kind': 'endo' if n in endo else 'exo',
              'desc': str(desc.get(n, ''))} for n in sorted(sub.nodes)]
    return {'var': var, 'nodes': nodes, 'edges': sorted(sub.edges),
            'up': sorted(g.predecessors(var)), 'down': sorted(g.successors(var))}


def _dot_esc(s):
    return str(s).replace('\\', '\\\\').replace('"', '\\"')


def dependencies_dot(dep):
    """DOT text for a result of :func:`dependencies`."""
    lines = ['digraph deps {', 'rankdir=LR;',
             'node [shape=box style="rounded,filled" fillcolor=white '
             'fontname=Helvetica fontsize=10];']
    for n in dep['nodes']:
        fill = ('lightsteelblue' if n['name'] == dep['var'] else
                'white' if n['kind'] == 'endo' else 'lightyellow')
        lines.append(f'"{_dot_esc(n["name"])}" [tooltip="{_dot_esc(n["desc"] or n["name"])}" '
                     f'fillcolor={fill}];')
    lines += [f'"{_dot_esc(a)}" -> "{_dot_esc(b)}";' for a, b in dep['edges']]
    lines.append('}')
    return '\n'.join(lines)


def dependencies_svg(dep):
    """SVG for a result of :func:`dependencies`, made with the Graphviz ``dot``
    executable - the same binary ``model.draw`` uses."""
    import subprocess

    res = subprocess.run(['dot', '-Tsvg'], input=dependencies_dot(dep).encode('utf-8'),
                         capture_output=True)
    if res.returncode != 0:
        raise RuntimeError('dot failed: ' + res.stderr.decode('utf-8', 'replace')[:500])
    return res.stdout.decode('utf-8')


def dependencies_text(dep):
    """Plain text for a result of :func:`dependencies` (for an assistant)."""
    desc = {n['name']: n['desc'] for n in dep['nodes']}
    kind = {n['name']: n['kind'] for n in dep['nodes']}

    def row(v):
        return f'  {v:<24} {kind.get(v, ""):<4}  {desc.get(v, "")}'
    out = [f'{dep["var"]}: {desc.get(dep["var"], "")}',
           'Direct inputs:', *([row(v) for v in dep['up']] or ['  none']),
           'Directly used by:', *([row(v) for v in dep['down']] or ['  none'])]
    if len(dep['edges']) > len(dep['up']) + len(dep['down']):
        out += ['All links in the neighbourhood:',
                *(f'  {a} -> {b}' for a, b in dep['edges'])]
    return '\n'.join(out)


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------

def var_rows(m, pattern='*', limit=4000):
    """Rows selected with ``m.vlist`` (name wildcards, ``!description``,
    ``#group``): [{'name', 'kind', 'desc', 'value'}]; value is the last
    period of ``m.lastdf``."""
    pattern = (pattern or '*').strip() or '*'
    try:
        names = m.vlist(pattern)
    except Exception as e:
        groups = ', '.join(getattr(m, 'var_groups', {}) or {})
        hint = f' Available groups: {groups}' if groups else ''
        raise ValueError(f'vlist({pattern!r}): {e}.{hint}') from e
    names = list(dict.fromkeys(names))
    desc = descriptions(m)
    df = getattr(m, 'lastdf', None)
    endo = set(m.endogene)

    def last_value(v):
        try:
            x = float(df[v].iloc[-1])
            return None if math.isnan(x) else x
        except Exception:
            return None

    return [{'name': v, 'kind': 'endo' if v in endo else 'exo',
             'desc': str(desc.get(v, '')),
             'value': last_value(v) if df is not None else None}
            for v in names[:limit]]


def equation(m, var):
    """{'var', 'desc', 'endo', 'frml', 'options', 'used_in'} for a variable."""
    v = var.strip().upper()
    if v not in m.allvar:
        raise KeyError(f'{v} is not in the model')
    used = sorted(w for w in m.endogene if w != v and
                  v in {t.var for t in m.allvar[w]['terms'] if t.var})
    endo = v in m.endogene
    return {'var': v, 'desc': str(descriptions(m).get(v, '')), 'endo': endo,
            'frml': m.allvar[v].get('frml', '') if endo else '',
            'options': m.allvar[v].get('frmlname', '') if endo else '',
            'used_in': used}


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

def scenario_names(m):
    return list(getattr(m, 'keep_solutions', {}) or {})


def default_scenarios(m):
    """'*' (all kept solutions) when there are at least two, else '' which
    makes the report classes use basedf/lastdf."""
    return '*' if len(scenario_names(m)) >= 2 else ''


@contextlib.contextmanager
def compare(m, reference='', scenario=''):
    """Temporarily make ``m.basedf``/``m.lastdf`` two kept solutions, so
    methods that compare base and last (dekomp, get_att ...) work on any pair
    of scenarios. Empty names keep the current basedf/lastdf."""
    kept = getattr(m, 'keep_solutions', {}) or {}
    for name in (reference, scenario):
        if name and name not in kept:
            raise KeyError(f'No kept solution called {name!r}. Kept: {list(kept)}')
    if not reference and not scenario:
        yield
        return
    old = m.basedf, m.lastdf
    try:
        if reference:
            m.basedf = kept[reference]
        if scenario:
            m.lastdf = kept[scenario]
        _clear_dekomp_cache(m)
        yield
    finally:
        m.basedf, m.lastdf = old
        _clear_dekomp_cache(m)


def _clear_dekomp_cache(m):
    clear = getattr(getattr(m, 'dekomp', None), 'cache_clear', None)
    if clear:
        clear()


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------

def attribution(m, var, reference='', scenario='', type='pct', lag=False,
                threshold=0.5, start='', end=''):
    """Attribution of the difference in ``var`` between two solutions, with
    ``m.get_att`` (type 'pct', 'level' or 'growth'). reference/scenario are
    names in keep_solutions; empty means basedf/lastdf."""
    var = var.strip().upper()
    with compare(m, reference, scenario):
        try:
            return m.get_att(var, type=type, lag=lag, threshold=threshold,
                             start=start, end=end)
        except Exception:
            if type != 'pct':
                raise
            res = m.dekomp(var, lprint=0)      # older models: raw decomposition
            return res[2] if isinstance(res, (list, tuple)) else res


# --------------------------------------------------------------------------
# Tables, plots and reports - model.table / model.plot / model.reports
# --------------------------------------------------------------------------

def table(m, pat='#Headline', datatype='growth', scenarios=None, **options):
    """``m.table`` (a DisplayVarTableDef). scenarios None means
    :func:`default_scenarios`."""
    scenarios = default_scenarios(m) if scenarios is None else scenarios
    return m.table(pat, datatype=datatype, scenarios=scenarios, **options)


def plot(m, pat='#Headline', datatype='growth', scenarios=None, **options):
    """``m.plot`` (a DisplayKeepFigDef)."""
    scenarios = default_scenarios(m) if scenarios is None else scenarios
    return m.plot(pat, datatype=datatype, scenarios=scenarios, **options)


def table_text(tab, periods=None, max_width=250):
    """A report table as aligned text, with title and unit line.
    periods: show only these columns (period labels)."""
    df = tab.df_str
    if periods:
        df = df.loc[:, [p for p in periods if p in df.columns]]
    with pd.option_context('display.width', max_width, 'display.max_columns', 200,
                           'display.max_rows', 2000, 'display.max_colwidth', 60):
        body = df.to_string()
    title = getattr(tab.options, 'title', '') or ''
    return f'{title}\n{body}' if title else body


def table_html(tab):
    """A report table as an HTML fragment, styled as in a notebook."""
    try:
        return tab.make_html_style(tab.df_str).to_html()
    except Exception:
        return tab.df_str.to_html()


def save_figs(figs, folder, prefix='', ext='png', dpi=110):
    """Save a dict of matplotlib figures; returns the file paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))
        p = folder / f'{prefix}{safe}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        paths.append(str(p))
    return paths


def _fig_img(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return ('<img style="max-width:100%" src="data:image/png;base64,'
            + base64.b64encode(buf.getvalue()).decode('ascii') + '">')


def report_names(m):
    return list(getattr(m, 'reports', {}) or {})


def get_report(m, name_or_spec):
    """A stored report (``m.get_report``) or one built from a JSON spec
    (``m.report_from_spec``)."""
    s = name_or_spec.strip()
    if s.startswith('{'):
        return m.report_from_spec(s)
    if s not in report_names(m):
        raise KeyError(f'No stored report {s!r}. Stored: {report_names(m) or "none"}')
    return m.get_report(s)


def report_parts(rep):
    """The displays of a report: the DisplayDefs of a container, or [rep]."""
    return list(getattr(rep, 'reports', None) or [rep])


def report_text(rep):
    """Text for an assistant: tables as text, text parts as text, charts by name."""
    out = []
    for r in report_parts(rep):
        if hasattr(r, 'figs'):
            out.append('Charts: ' + ', '.join(map(str, r.figs)))
        elif hasattr(r, 'df_str'):
            out.append(table_text(r))
        elif hasattr(r, 'text_text'):
            out.append(r.mmodel.string_substitution(r.text_text))
    return '\n\n'.join(out)


def report_html(rep, title=''):
    """A self-contained HTML page (charts inlined as PNG) for a report."""
    body = []
    for r in report_parts(rep):
        if hasattr(r, 'figs'):
            body += [_fig_img(f) for f in r.figs.values()]
        elif hasattr(r, 'df_str'):
            body.append(table_html(r))
        elif hasattr(r, 'html_text'):
            body.append(r.mmodel.string_substitution(r.html_text))
    title = html.escape(title or getattr(rep, 'name', '') or 'ModelFlow report')
    return ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{title}</title><style>body{{font-family:sans-serif;margin:2em}}'
            'table{border-collapse:collapse;font-size:13px}td,th{padding:2px 8px}'
            f'</style></head><body><h1>{title}</h1>\n' + '\n'.join(body) + '</body></html>')
