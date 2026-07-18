"""Kernel-side comm target for the ModelFlow workbench.

Importing this module inside an IPython kernel registers the
``mf_workbench`` comm target. The JupyterLab frontend opens a comm to
this target and exchanges small JSON messages:

    -> {"action": "list"}
    <- {"models": ["m", ...]}

    -> {"action": "graph", "model": "m", "var": "GDP", "up": 1, "down": 1}
    <- {"svg": "<svg ...>"}

    -> {"action": "vars", "model": "m"}
    <- {"rows": [{"name": ..., "desc": ..., "value": ..., "kind": ...}]}

    -> {"action": "attribution", "model": "m", "var": "GDP"}
    <- {"table": {"columns": [...], "index": [...], "data": [[...]]}}

All heavy lifting (graph traversal, Graphviz layout, attribution) stays
here in Python so the TypeScript side is a thin display layer.

NOTE: the three ``_mf_*`` helpers below are the integration points with
ModelFlow. Adjust them to taste -- everything else is plumbing.
"""

from __future__ import annotations

import json
import math


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------

def _find_models(ip):
    """Return {name: instance} for ModelFlow model objects in user_ns."""
    out = {}
    for name, obj in ip.user_ns.items():
        if name.startswith('_'):
            continue
        # Duck-typed on purpose: works for modelclass.model and subclasses
        # without importing modelflow into every kernel.
        if hasattr(obj, 'totgraph') and hasattr(obj, 'endogene'):
            out[name] = obj
    return out


# --------------------------------------------------------------------------
# Integration point 1: dependency subgraph -> SVG
# --------------------------------------------------------------------------

def _mf_graph_svg(m, var, up=1, down=1):
    """Render the local dependency graph around ``var`` as an SVG string."""
    import networkx as nx
    from graphviz import Digraph

    var = var.strip().upper()
    g = m.totgraph
    if var not in g:
        return f'<p>Variable <b>{var}</b> is not in the dependency graph.</p>'

    keep = {var}
    keep |= set(nx.single_source_shortest_path_length(
        g.reverse(copy=False), var, cutoff=int(up)))
    keep |= set(nx.single_source_shortest_path_length(
        g, var, cutoff=int(down)))
    sub = g.subgraph(keep)

    desc = getattr(m, 'var_description', {}) or {}
    dot = Digraph('deps')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded,filled',
             fillcolor='white', fontname='Helvetica', fontsize='10')
    for n in sub.nodes:
        dot.node(
            n,
            tooltip=str(desc.get(n, n)),
            fillcolor='lightsteelblue' if n == var else
                      ('white' if n in m.endogene else 'lightyellow'),
        )
    for a, b in sub.edges:
        dot.edge(a, b)
    return dot.pipe(format='svg').decode('utf-8')


# --------------------------------------------------------------------------
# Integration point 2: variable explorer rows
# --------------------------------------------------------------------------

def _mf_var_rows(m, limit=4000):
    """Rows for the variable explorer: name, description, last value."""
    desc = getattr(m, 'var_description', {}) or {}
    df = getattr(m, 'lastdf', None)
    endo = set(getattr(m, 'endogene', set()))
    exo = set(getattr(m, 'exogene', set()))

    def last_value(v):
        try:
            x = float(df[v].iloc[-1])
            return None if math.isnan(x) else x
        except Exception:
            return None

    rows = []
    for v in sorted(endo | exo):
        rows.append({
            'name': v,
            'kind': 'endo' if v in endo else 'exo',
            'desc': str(desc.get(v, '')),
            'value': last_value(v) if df is not None else None,
        })
        if len(rows) >= limit:
            break
    return rows


# --------------------------------------------------------------------------
# Integration point 3: attribution
# --------------------------------------------------------------------------

def _mf_attribution(m, var):
    """Attribution table for ``var`` as a pandas DataFrame.

    Swap in your preferred entry point (get_att_pct, dekomp,
    get_att_level, totexplain, ...).
    """
    var = var.strip().upper()
    try:
        return m.get_att_pct(var, lag=False, threshold=0.5)
    except Exception:
        # Fallback: raw decomposition
        res = m.dekomp(var, lprint=0)
        # dekomp returns several frames; pick the one you want to show
        return res[1] if isinstance(res, (list, tuple)) else res


# --------------------------------------------------------------------------
# Comm plumbing
# --------------------------------------------------------------------------

def _df_to_split(df):
    """DataFrame -> JSON-safe dict with columns/index/data."""
    d = json.loads(df.round(3).to_json(orient='split', date_format='iso'))
    return d


def _handle(ip, data):
    action = data.get('action', '')
    if action == 'list':
        return {'models': sorted(_find_models(ip))}

    models = _find_models(ip)
    name = data.get('model') or (sorted(models)[0] if models else None)
    if not name or name not in models:
        return {'error': f'No ModelFlow model instance found '
                         f'(looked for: {name!r}). Run a model first.'}
    m = models[name]

    if action == 'graph':
        return {'svg': _mf_graph_svg(m, data.get('var', ''),
                                     data.get('up', 1), data.get('down', 1))}
    if action == 'vars':
        return {'rows': _mf_var_rows(m)}
    if action == 'attribution':
        df = _mf_attribution(m, data.get('var', ''))
        return {'table': _df_to_split(df), 'var': data.get('var', '')}
    return {'error': f'Unknown action: {action!r}'}


def _register():
    from IPython import get_ipython
    ip = get_ipython()
    if ip is None or not hasattr(ip, 'kernel'):
        return

    def target(comm, open_msg):
        @comm.on_msg
        def _recv(msg):
            data = msg['content']['data']
            try:
                reply = _handle(ip, data)
            except Exception as e:          # noqa: BLE001 - report to panel
                reply = {'error': f'{type(e).__name__}: {e}'}
            reply['action'] = data.get('action', '')
            comm.send(reply)

    # Re-importing is harmless: register_target overwrites.
    ip.kernel.comm_manager.register_target('mf_workbench', target)


_register()
