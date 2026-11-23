"""Kernel-side comm target for the ModelFlow workbench.

Importing this module inside an IPython kernel registers the
``mf_workbench`` comm target. The JupyterLab frontend opens a comm to
this target and exchanges small JSON messages:

    -> {"action": "list"}
    <- {"models": ["m", ...]}

    -> {"action": "graph", "model": "m", "var": "GDP", "up": 1, "down": 1}
    <- {"svg": "<svg ...>"}

    -> {"action": "vars", "model": "m", "pattern": "*"}
    <- {"rows": [{"name": ..., "desc": ..., "value": ..., "kind": ...}]}
       (pattern goes through model.vlist: name wildcards, !description, #group)

    -> {"action": "attribution", "model": "m", "var": "GDP"}
    <- {"table": {"columns": [...], "index": [...], "data": [[...]]}}

    -> {"action": "focus"}
    <- {"focus": {"model": "m", "var": "GDP", "up": 1, "down": 1, "pattern": "*"}}

The ModelFlow logic is in ``modelinspect`` (part of ModelFlow), which the
ModelFlow MCP server uses too, so the panel and an AI assistant show the
same graph, variables and attribution.

``focus`` is what the panel shows now. It is kept in the kernel so an
assistant attached to the same kernel (``handle({'action': 'focus'})``) can
see what the user is looking at.
"""

from __future__ import annotations

import html
import json

import modelinspect as mi

# What the panel shows now: updated by every graph/vars/attribution request.
FOCUS: dict = {}


def _flat_labels(labels):
    """MultiIndex-safe: tuples -> 'a b' strings (to_json orient='split'
    cannot handle a MultiIndex, and dekomp frames often carry one)."""
    return [' '.join(map(str, x)) if isinstance(x, tuple) else str(x)
            for x in labels]


def _df_to_split(df):
    """DataFrame -> JSON-safe dict with columns/index/data."""
    df = df.copy()
    df.index = _flat_labels(df.index)
    df.columns = _flat_labels(df.columns)
    return json.loads(df.round(3).to_json(orient='split', date_format='iso'))


def _graph_svg(m, var, up, down):
    try:
        return mi.dependencies_svg(mi.dependencies(m, var, up, down))
    except KeyError:
        return (f'<p>Variable <b>{html.escape(var.strip().upper())}</b> is not in the '
                f'dependency graph.</p>')
    except FileNotFoundError:
        return ('<p>Graphviz <b>dot</b> executable not found on PATH - '
                'install it (conda install -c conda-forge graphviz).</p>')
    except RuntimeError as e:
        return f'<pre>{html.escape(str(e))}</pre>'


def handle(data, namespace=None):
    """Answer one request. ``namespace`` defaults to the IPython user namespace."""
    if namespace is None:
        from IPython import get_ipython
        namespace = get_ipython().user_ns
    action = data.get('action', '')
    models = mi.find_models(namespace)
    if action == 'list':
        return {'models': sorted(models)}
    if action == 'focus':
        return {'focus': dict(FOCUS)}

    name = data.get('model') or (sorted(models)[0] if models else None)
    if not name or name not in models:
        return {'error': f'No ModelFlow model instance found '
                         f'(looked for: {name!r}). Run a model first.'}
    m = models[name]
    FOCUS['model'] = name

    if action == 'graph':
        up, down = int(data.get('up', 1)), int(data.get('down', 1))
        FOCUS.update(var=data.get('var', '').strip().upper(), up=up, down=down)
        return {'svg': _graph_svg(m, data.get('var', ''), up, down)}
    if action == 'vars':
        FOCUS['pattern'] = data.get('pattern', '*')
        return {'rows': mi.var_rows(m, data.get('pattern', '*'))}
    if action == 'attribution':
        FOCUS['var'] = data.get('var', '').strip().upper()
        df = mi.attribution(m, data.get('var', ''))
        return {'table': _df_to_split(df), 'var': data.get('var', '')}
    return {'error': f'Unknown action: {action!r}'}


_handle = handle      # old name


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
                reply = handle(data, ip.user_ns)
            except Exception as e:          # noqa: BLE001 - report to panel
                reply = {'error': f'{type(e).__name__}: {e}'}
            reply['action'] = data.get('action', '')
            comm.send(reply)

    # Re-importing is harmless: register_target overwrites.
    ip.kernel.comm_manager.register_target('mf_workbench', target)


_register()
