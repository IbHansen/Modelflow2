'''An interactive causality viewer for a model, which runs in the notebook itself

The viewer shows the graph of the variables which determine - and are determined
by - a variable. A click on a variable moves the graph to it, and the boxes glide
to their new places.

It is an anywidget, so it works where the notebook runs: JupyterLab, Notebook 7,
VS Code and JupyterLite in the browser. It needs neither graphviz nor a web server
like the Dash dashboard does:

- the graph is made in Python by get_alllinks and alllinks_to_nx of modelclass
- the boxes are placed in Python by one of three engines: the graphviz dot
  program when it is there, else grandalf - the method of dot in pure Python, so
  it also works in the browser - else the columns of nx_layout
- the drawing is plain svg made by the javascript below, without any libraries,
  so nothing is fetched from the internet

The javascript and the css are strings in this module, and not separate files,
because the package ships the modules listed in pyproject py-modules only.

Use::

    mmodel.causality('FY', up=2)

After a run, the links are colored and sized by the attribution, like draw does.
'''

import json
import shutil
import subprocess
import sys
from collections import defaultdict, namedtuple

try:
    import traitlets
    from anywidget import AnyWidget
except ImportError:          # the viewer is not there, the rest of modelflow works
    AnyWidget = None

# the same fields as node in modelclass, alllinks_to_nx only needs the fields
link = namedtuple('link', 'lev,parent,child')


def causality_graph(mmodel, navn, up=1, down=0, filter=0, lag=False, des=False,
                    max_nodes=300):
    '''The networkx graph of the links around navn, made by alllinks_to_nx'''
    alllinks = mmodel.get_alllinks(navn, down=down, up=up, lag=lag, filter=filter)
    nodes = {n for l in alllinks for n in (l.parent, l.child)}
    if len(nodes) > max_nodes:
        raise ValueError(f'The graph has {len(nodes)} variables, more than max_nodes={max_nodes}. '
                         'Use fewer levels or a filter')
    if not alllinks:
        # a variable alone, a link to itself gives the node with all its properties
        G = mmodel.alllinks_to_nx([link(0, navn, navn)], navn=navn, des=des)
        G.remove_edge(navn, navn)
        return G
    return mmodel.alllinks_to_nx(alllinks, navn=navn, des=des)


#: the layout engines in the order engine='auto' tries them
layout_engines = ('dot', 'grandalf', 'nx')


def _border(center, box, v, toward):
    '''The point where the line from the center of v towards the point toward
    leaves the box of v'''
    cx, cy = center[v]
    w, h = box[v]
    dx, dy = toward[0] - cx, toward[1] - cy
    if not dx and not dy:
        return cx, cy
    scale = min(w / 2. / abs(dx) if dx else 1e9, h / 2. / abs(dy) if dy else 1e9)
    return cx + dx * scale, cy + dy * scale


def _smooth(points):
    '''Path commands for a smooth line through the points of a polyline

    The line turns at the middle of each piece, with the inner points as control
    points, so it follows the points closely without corners.'''
    if len(points) == 2:
        return [('M', points[0]), ('L', points[1])]
    cmds = [('M', points[0])]
    for p, q in zip(points[1:-1], points[2:]):
        cmds.append(('Q', p, ((p[0] + q[0]) / 2., (p[1] + q[1]) / 2.)))
    cmds.append(('L', points[-1]))
    return cmds


def _loop(center, box, v):
    '''Path commands for a link from v to itself, a loop at the upper right corner'''
    cx, cy = center[v]
    w, h = box[v]
    x0, y0 = cx + w / 2. - 12., cy - h / 2.
    return [('M', (x0, y0)), ('C', (x0, y0 - 28.), (cx + w / 2. + 30., cy), (cx + w / 2., cy))]


def _layout_dot(mmodel, G, box, gap_x, gap_y):
    '''The layout of the graphviz dot program, the same as draw makes

    Only the positions are taken from dot, it gets boxes of the right size
    without labels, and returns the positions and the splines of the links.'''
    if sys.platform == 'emscripten' or getattr(mmodel, 'no_graphviz', False):
        raise RuntimeError('graphviz is not used here')
    program = shutil.which('dot')
    if not program:
        raise RuntimeError('the graphviz dot program is not found')

    def quote(name):
        return '"' + str(name).replace('\\', '\\\\').replace('"', '\\"') + '"'

    # dot measures in points, 72 to an inch, and the boxes in inches
    lines = ['digraph G {',
             f'graph [rankdir=LR, ranksep={gap_x / 72.:.3f}, nodesep={gap_y / 72.:.3f}];',
             'node [shape=box, fixedsize=true, label=""];']
    lines += [f'{quote(v)} [width={box[v][0] / 72.:.3f}, height={box[v][1] / 72.:.3f}];'
              for v in G.nodes]
    lines += [f'{quote(a)} -> {quote(b)};' for a, b in G.edges]
    lines.append('}')
    result = subprocess.run([program, '-Tjson'], input='\n'.join(lines), capture_output=True,
                            encoding='utf-8', timeout=60,
                            # no console window flashing on Windows
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'dot failed: {result.stderr.strip()}')
    out = json.loads(result.stdout)

    def point(text):
        # dot has y upwards, the svg downwards
        x, y = text.split(',')[:2]
        return float(x), -float(y)

    objects = out.get('objects', [])
    names = {o['_gvid']: o['name'] for o in objects}
    center = {o['name']: point(o['pos']) for o in objects if 'pos' in o and o['name'] in box}
    if set(center) != set(G.nodes):
        raise RuntimeError('dot did not place all the variables')

    routes = {}
    for e in out.get('edges', []):
        if 'pos' not in e:
            continue
        points, end = [], None
        for token in e['pos'].replace('\\\n', '').split():
            if token.startswith('e,'):
                end = point(token[2:])        # the tip of the arrow
            elif not token.startswith('s,'):
                points.append(point(token))
        # a b-spline: a start point and then three points for each piece
        cmds = [('M', points[0])] + [('C', points[i], points[i + 1], points[i + 2])
                                     for i in range(1, len(points) - 2, 3)]
        if end:
            cmds.append(('L', end))
        routes[names[e['tail']], names[e['head']]] = cmds
    return center, routes


def _layout_grandalf(mmodel, G, box, gap_x, gap_y):
    '''The Sugiyama layout of grandalf, the method of dot in pure Python

    Long links get bend points in the columns they pass, so they go around the
    boxes instead of through them.'''
    from grandalf.graphs import Vertex, Edge, Graph
    from grandalf.layouts import SugiyamaLayout

    class View:
        def __init__(self, w, h):
            self.w, self.h, self.xy = w, h, (0., 0.)

    # grandalf places the layers from the top down. So the layout is made with
    # the width and the height of the boxes swapped, and x and y are swapped back
    # afterwards, then the layers go from the left to the right like rankdir=LR
    vertex = {}
    for v in G.nodes:
        vertex[v] = Vertex(v)
        vertex[v].view = View(box[v][1], box[v][0])
    ends = {Edge(vertex[a], vertex[b]): (a, b) for a, b in G.edges if a != b}
    g = Graph(list(vertex.values()), list(ends))

    center, routes, offset = {}, {}, 0.
    for core in g.C:     # each part of the graph which is not connected to the rest
        sug = SugiyamaLayout(core)
        sug.xspace, sug.yspace = gap_y, gap_x
        roots = [v for v in core.sV if len(v.e_in()) == 0] or [next(iter(core.sV))]
        sug.init_all(roots=roots)
        sug.draw()

        part = {v.data: (v.view.xy[1], v.view.xy[0]) for v in core.sV}
        top = min(y - box[v][1] / 2. for v, (x, y) in part.items())
        bottom = max(y + box[v][1] / 2. for v, (x, y) in part.items())
        for v, (x, y) in part.items():
            center[v] = (x, y - top + offset)

        for e in core.sE:
            a, b = ends[e]
            bends = []
            if e in sug.ctrls:
                # the same walk over the ranks as grandalf's draw_edges
                r0, r1 = sug.grx[e.v[0]].rank, sug.grx[e.v[1]].rank
                ranks = range(r0 + 1, r1) if r0 < r1 else range(r0 - 1, r1, -1)
                bends = [(xy[1], xy[0] - top + offset)
                         for xy in (sug.ctrls[e][r].view.xy for r in ranks)]
                if e.v[0] is not vertex[a]:
                    bends.reverse()
            first = bends[0] if bends else center[b]
            last = bends[-1] if bends else center[a]
            routes[a, b] = _smooth([_border(center, box, a, first), *bends,
                                    _border(center, box, b, last)])
        offset += bottom - top + 2 * gap_y
    return center, routes


def _layout_nx(mmodel, G, box, gap_x, gap_y):
    '''Columns by the layer from alllinks_to_nx, ordered by nx_layout

    The same as display_nx_svg draws. The links are curved where they point back
    or run within a column.'''
    pos = mmodel.nx_layout(G, sort=True)
    layers = defaultdict(list)
    for v in sorted(G.nodes, key=lambda v: pos[v][1]):
        layers[G.nodes[v]['layer']].append(v)

    center, colwidths, x = {}, {}, 0.
    for lev in sorted(layers):
        colwidth = max(box[v][0] for v in layers[lev])
        colwidths[lev] = colwidth
        colheight = sum(box[v][1] + gap_y for v in layers[lev]) - gap_y
        y = -colheight / 2.
        for v in layers[lev]:
            center[v] = (x + colwidth / 2., y + box[v][1] / 2.)
            y += box[v][1] + gap_y
        x += colwidth + gap_x

    def bow(child, parent):
        '''How far the link bows out from the straight line, see display_nx_svg'''
        lchild, lparent = G.nodes[child]['layer'], G.nodes[parent]['layer']
        if lchild == lparent:
            return colwidths[lchild] + gap_x / 2.
        if lparent < lchild or G.has_edge(parent, child):
            return gap_y * (1 + abs(lparent - lchild))
        return 0.

    routes = {}
    for child, parent in G.edges:
        if child == parent:
            continue
        (cx1, cy1), (cx2, cy2) = center[child], center[parent]
        dx, dy = cx2 - cx1, cy2 - cy1
        span = max((dx * dx + dy * dy) ** 0.5, 0.1)
        side = bow(child, parent)
        ctrl = ((cx1 + cx2) / 2. - dy / span * side, (cy1 + cy2) / 2. + dx / span * side)
        routes[child, parent] = [('M', _border(center, box, child, ctrl)),
                                 ('Q', ctrl, _border(center, box, parent, ctrl))]
    return center, routes


_engines = dict(dot=_layout_dot, grandalf=_layout_grandalf, nx=_layout_nx)


def causality_layout(mmodel, G, navn, fontsize=12, engine='auto'):
    '''Positions of the boxes and the paths of the links, as a dict for json

    :engine: dot uses the graphviz dot program, grandalf the Sugiyama layout of
             the grandalf package - the method of dot in pure Python - and nx the
             columns of nx_layout. auto takes the first of them which works, so
             dot on a desktop with graphviz and grandalf in a browser.
    '''
    if engine != 'auto' and engine not in _engines:
        raise ValueError(f'{engine} is not a layout engine, use auto, {", ".join(_engines)}')
    pad, gap_y = 6, 22
    char_width = 0.62 * fontsize   # the labels are drawn in a monospace font
    box = {v: (len(G.nodes[v]['label']) * char_width + 2 * pad, fontsize + 2 * pad)
           for v in G.nodes}
    gap_x = max(90., 0.25 * max(w for w, h in box.values()))

    tries = layout_engines if engine == 'auto' else (engine,)
    for used in tries:
        try:
            center, routes = _engines[used](mmodel, G, box, gap_x, gap_y)
            break
        except Exception:
            if used == tries[-1]:
                raise

    for child, parent in G.edges:
        if (child, parent) not in routes:
            routes[child, parent] = (_loop(center, box, child) if child == parent else
                                     [('M', _border(center, box, child, center[parent])),
                                      ('L', _border(center, box, parent, center[child]))])

    # move the drawing, so it starts a margin from the upper left corner
    xs = [cx + s * box[v][0] / 2. for v, (cx, cy) in center.items() for s in (-1, 1)]
    ys = [cy + s * box[v][1] / 2. for v, (cx, cy) in center.items() for s in (-1, 1)]
    for cmds in routes.values():
        for cmd in cmds:
            xs += [p[0] for p in cmd[1:]]
            ys += [p[1] for p in cmd[1:]]
    left, top = min(xs) - gap_y, min(ys) - gap_y

    def path(cmds):
        return ' '.join(cmd[0] + ''.join(f' {x - left:.1f} {y - top:.1f}' for x, y in cmd[1:])
                        for cmd in cmds)

    nodes = []
    for v, (cx, cy) in center.items():
        w, h = box[v]
        nodes.append(dict(id=v, label=G.nodes[v]['label'], x=round(cx - left, 1),
                          y=round(cy - top, 1), w=round(w, 1), h=round(h, 1),
                          color=G.nodes[v]['color'], tooltip=G.nodes[v]['tooltip']))

    edges = []
    for child, parent in G.edges:
        edge = G.edges[child, parent]
        if not edge.get('visible', True):
            continue
        # the sign of the largest attribution colors the link
        att_min, att_max = edge.get('att_min'), edge.get('att_max')
        if att_min is None or att_max is None:
            sign = 0
        else:
            biggest = att_max if abs(att_max) >= abs(att_min) else att_min
            sign = 1 if biggest > 0 else (-1 if biggest < 0 else 0)
        # a link which does not go to the right points back, like a lag feedback
        back = child == parent or center[parent][0] <= center[child][0]
        edges.append(dict(source=child, target=parent, d=path(routes[child, parent]),
                          width=round(max(1., edge['width']), 1), sign=sign,
                          back=bool(back), tooltip=edge['tooltip']))

    try:
        description = mmodel.var_des(navn)
    except Exception:
        description = ''
    return dict(center=navn, title=f'{navn}: {description}' if description else navn,
                engine=used, fontsize=fontsize,
                width=round(max(xs) - left + gap_y, 1), height=round(max(ys) - top + gap_y, 1),
                nodes=nodes, edges=edges)


_ESM = r'''
const NS = "http://www.w3.org/2000/svg";
const SIGN = {pos: "#3b6fc4", neg: "#c9463d", none: "#999999"};
let uid = 0;

function h(tag, attrs, parent, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function s(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function render({ model, el }) {
  const id = `mfc${++uid}${Math.random().toString(36).slice(2, 7)}`;
  const handlers = [];
  const on = (name, f) => { model.on(name, f); handlers.push([name, f]); };
  const root = h("div", {class: "mfc"}, el);

  // ---- the toolbar -------------------------------------------------------
  const bar = h("div", {class: "mfc-bar"}, root);
  const backBtn = h("button", {title: "Back to the previous variable"}, bar, "◀ Back");
  const varLab = h("label", {}, bar, "Variable ");
  const varIn = h("input", {class: "mfc-var", list: `${id}-vars`, spellcheck: "false",
                            title: "Type or pick a variable, Enter moves the graph to it"}, varLab);
  const datalist = h("datalist", {id: `${id}-vars`}, bar);
  const num = (label, title) => {
    const l = h("label", {title}, bar, label + " ");
    return h("input", {class: "mfc-num", type: "number", min: "0", max: "20", step: "1"}, l);
  };
  const upIn = num("Up", "Levels of the variables which determine the variable");
  const downIn = num("Down", "Levels of the variables which the variable determines");
  const filterLab = h("label", {title: "Prune the branches where the attribution is below this in all periods"}, bar, "Filter ");
  const filterIn = h("select", {}, filterLab);
  for (let t = 0; t < 100; t += 10) h("option", {value: String(t)}, filterIn, `${t}%`);
  const check = (label, title) => {
    const l = h("label", {title}, bar);
    const c = h("input", {type: "checkbox"}, l);
    l.appendChild(document.createTextNode(" " + label));
    return c;
  };
  const lagIn = check("Lags", "Show the lagged variables as separate boxes");
  const desIn = check("Descriptions", "Show the descriptions of the variables in the boxes");
  const engineLab = h("label", {title: "How the boxes are placed: dot is graphviz, grandalf the same method in Python, nx simple columns. auto takes the first which works"}, bar, "Layout ");
  const engineIn = h("select", {}, engineLab);
  for (const k of ["auto", "dot", "grandalf", "nx"]) h("option", {value: k}, engineIn, k);
  const status = h("span", {class: "mfc-status"}, bar);

  const title = h("div", {class: "mfc-title"}, root);
  const msg = h("div", {class: "mfc-msg"}, root);

  // ---- the graph ---------------------------------------------------------
  const gbox = h("div", {class: "mfc-graph"}, root);
  const zoom = h("div", {class: "mfc-zoom"}, gbox);
  const zin = h("button", {title: "Zoom in, or Ctrl + mouse wheel"}, zoom, "+");
  const zout = h("button", {title: "Zoom out, or Ctrl + mouse wheel"}, zoom, "−");
  const zfit = h("button", {title: "Fit the graph, or double click"}, zoom, "Fit");
  const gsvg = s("svg", {class: "mfc-gsvg", preserveAspectRatio: "xMidYMid meet"}, gbox);
  const defs = s("defs", {}, gsvg);
  for (const [k, color] of Object.entries(SIGN)) {
    const m = s("marker", {id: `${id}-${k}`, viewBox: "0 0 10 10", refX: "10", refY: "5",
                           markerWidth: "9", markerHeight: "9", markerUnits: "userSpaceOnUse",
                           orient: "auto-start-reverse"}, defs);
    s("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: color}, m);
  }
  const gEdges = s("g", {}, gsvg);
  const gNodes = s("g", {}, gsvg);

  // ---- state -------------------------------------------------------------
  let vars = new Set();
  const history = [];
  let view = {x: 0, y: 0, w: 100, h: 100}, full = {x: 0, y: 0, w: 100, h: 100};
  let userView = false, edgeTimer = null;

  function go(v) {
    v = (v || "").trim().toUpperCase().replace(/\(.*$/, "");
    if (!v || v === model.get("var")) return;
    history.push(model.get("var"));
    model.set("var", v);
    model.save_changes();
    backBtn.disabled = false;
  }

  backBtn.onclick = () => {
    let v;
    do { v = history.pop(); } while (v !== undefined && v === model.get("var"));
    if (v !== undefined) { model.set("var", v); model.save_changes(); }
    backBtn.disabled = !history.length;
  };
  varIn.addEventListener("change", () => go(varIn.value));
  varIn.addEventListener("input", e => {
    // a pick from the list, not a typed letter
    if ((e.inputType === undefined || e.inputType === "insertReplacementText") &&
        vars.has(varIn.value.trim().toUpperCase())) go(varIn.value);
  });
  const setter = (input, name, value) => input.addEventListener("change", () => {
    model.set(name, value()); model.save_changes();
  });
  setter(upIn, "up", () => Math.max(0, parseInt(upIn.value, 10) || 0));
  setter(downIn, "down", () => Math.max(0, parseInt(downIn.value, 10) || 0));
  setter(filterIn, "filter", () => parseFloat(filterIn.value) || 0);
  setter(lagIn, "lag", () => lagIn.checked);
  setter(desIn, "des", () => desIn.checked);
  setter(engineIn, "engine", () => engineIn.value);

  function syncControls() {
    varIn.value = model.get("var");
    upIn.value = model.get("up");
    downIn.value = model.get("down");
    const f = String(model.get("filter"));
    if (![...filterIn.options].some(o => o.value === f)) h("option", {value: f}, filterIn, `${f}%`);
    filterIn.value = f;
    lagIn.checked = model.get("lag");
    desIn.checked = model.get("des");
    engineIn.value = model.get("engine");
    backBtn.disabled = !history.length;
  }

  function syncVariables() {
    const names = model.get("variables") || [];
    vars = new Set(names);
    datalist.replaceChildren(...names.map(v => { const o = document.createElement("option"); o.value = v; return o; }));
  }

  function syncBusy() {
    const busy = model.get("busy");
    root.classList.toggle("mfc-busy", busy);
    const used = (model.get("graph") || {}).engine;
    status.textContent = busy ? "calculating…" : used ? `placed by ${used}` : "";
  }

  const syncMsg = () => { msg.textContent = model.get("message") || ""; };
  const syncHeight = () => { gbox.style.height = model.get("height"); };

  // ---- zoom and pan, the viewBox of the svg is the view -------------------
  function setView(v) {
    view = v;
    gsvg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`);
  }

  function fitView() {
    const r = gbox.getBoundingClientRect();
    if (!r.width || !r.height) { setView({...full}); return; }
    // not blown up more than 1.3 times, a small graph would get huge letters
    const k = Math.min(r.width / full.w, r.height / full.h, 1.3);
    const w = r.width / k, hh = r.height / k;
    setView({x: full.x + (full.w - w) / 2, y: full.y + (full.h - hh) / 2, w, h: hh});
  }

  function zoomAt(f, cx, cy) {
    userView = true;
    setView({x: cx - (cx - view.x) * f, y: cy - (cy - view.y) * f, w: view.w * f, h: view.h * f});
  }

  function toSvg(e) {
    const p = gsvg.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    const m = gsvg.getScreenCTM();
    return m ? p.matrixTransform(m.inverse()) : {x: view.x + view.w / 2, y: view.y + view.h / 2};
  }

  zin.onclick = () => zoomAt(1 / 1.25, view.x + view.w / 2, view.y + view.h / 2);
  zout.onclick = () => zoomAt(1.25, view.x + view.w / 2, view.y + view.h / 2);
  zfit.onclick = () => { userView = false; fitView(); };
  gsvg.addEventListener("wheel", e => {
    // a plain wheel scrolls the notebook
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    const p = toSvg(e);
    zoomAt(Math.exp(e.deltaY * 0.002), p.x, p.y);
  }, {passive: false});

  let drag = null;
  gsvg.addEventListener("pointerdown", e => {
    if (e.button !== 0 || e.target.closest(".mfc-node")) return;
    drag = {sx: e.clientX, sy: e.clientY, v: {...view}};
    gsvg.setPointerCapture(e.pointerId);
    gbox.classList.add("mfc-dragging");
  });
  gsvg.addEventListener("pointermove", e => {
    if (!drag) return;
    const r = gsvg.getBoundingClientRect();
    const k = Math.max(drag.v.w / r.width, drag.v.h / r.height);
    userView = true;
    setView({...drag.v, x: drag.v.x - (e.clientX - drag.sx) * k, y: drag.v.y - (e.clientY - drag.sy) * k});
  });
  const endDrag = () => { drag = null; gbox.classList.remove("mfc-dragging"); };
  gsvg.addEventListener("pointerup", endDrag);
  gsvg.addEventListener("pointercancel", endDrag);
  gsvg.addEventListener("dblclick", e => {
    if (!e.target.closest(".mfc-node")) { userView = false; fitView(); }
  });

  // ---- the graph ---------------------------------------------------------
  function focusNode(nid) {
    const near = new Set([nid]);
    for (const p of gEdges.children) {
      const hit = p.dataset.s === nid || p.dataset.t === nid;
      p.classList.toggle("mfc-hl", hit);
      if (hit) { near.add(p.dataset.s); near.add(p.dataset.t); }
    }
    for (const n of gNodes.children) n.classList.toggle("mfc-hl", near.has(n.dataset.id));
    gbox.classList.add("mfc-focus");
  }

  function newNode(nid) {
    const e = s("g", {class: "mfc-node mfc-enter"}, gNodes);
    e.dataset.id = nid;
    s("title", {}, e);
    s("rect", {rx: "5"}, e);
    s("text", {"text-anchor": "middle", "dominant-baseline": "central"}, e);
    e.addEventListener("mouseenter", () => focusNode(nid));
    e.addEventListener("mouseleave", () => gbox.classList.remove("mfc-focus"));
    e.addEventListener("click", ev => { ev.stopPropagation(); go(nid); });
    // two frames, so the box is drawn transparent before it fades in
    requestAnimationFrame(() => requestAnimationFrame(() => e.classList.remove("mfc-enter")));
    return e;
  }

  function drawGraph() {
    const g = model.get("graph");
    if (!g || !g.nodes) return;
    title.textContent = g.title || "";
    gbox.classList.remove("mfc-focus");

    // the boxes are kept by name, so a box which stays glides to its new place
    const firstDraw = gNodes.children.length === 0;
    const old = new Map([...gNodes.children].map(e => [e.dataset.id, e]));
    for (const n of g.nodes) {
      let e = old.get(n.id);
      if (e) { old.delete(n.id); e.classList.remove("mfc-leave"); }
      else e = newNode(n.id);
      const [tip, rect, text] = e.children;
      tip.textContent = n.tooltip;
      rect.setAttribute("x", -n.w / 2);
      rect.setAttribute("y", -n.h / 2);
      rect.setAttribute("width", n.w);
      rect.setAttribute("height", n.h);
      rect.setAttribute("fill", n.color);
      text.setAttribute("font-size", g.fontsize);
      text.textContent = n.label;
      e.classList.toggle("mfc-center", n.id === g.center);
      e.style.transform = `translate(${n.x}px, ${n.y}px)`;
    }
    for (const e of old.values()) {
      e.classList.add("mfc-leave");
      setTimeout(() => { if (e.classList.contains("mfc-leave")) e.remove(); }, 500);
    }

    // the links are drawn anew, and shown when the boxes have arrived
    gEdges.replaceChildren();
    const paths = g.edges.map(ed => {
      const k = ed.sign > 0 ? "pos" : ed.sign < 0 ? "neg" : "none";
      const p = s("path", {d: ed.d, class: `mfc-edge mfc-hidden mfc-${k}${ed.back ? " mfc-back" : ""}`,
                           "stroke-width": ed.width, "marker-end": `url(#${id}-${k})`}, gEdges);
      p.dataset.s = ed.source;
      p.dataset.t = ed.target;
      s("title", {}, p, ed.tooltip);
      return p;
    });
    clearTimeout(edgeTimer);
    edgeTimer = setTimeout(() => paths.forEach(p => p.classList.remove("mfc-hidden")), firstDraw ? 0 : 450);

    full = {x: 0, y: 0, w: g.width, h: g.height};
    userView = false;
    fitView();
  }

  // ---- wiring ------------------------------------------------------------
  on("change:graph", drawGraph);
  on("change:graph", syncBusy);
  for (const name of ["var", "up", "down", "filter", "lag", "des", "engine"]) on(`change:${name}`, syncControls);
  on("change:variables", syncVariables);
  on("change:busy", syncBusy);
  on("change:message", syncMsg);
  on("change:height", syncHeight);

  // the graph fits the box again when the box changes size, unless zoomed by hand
  const ro = new ResizeObserver(() => {
    if (!userView && (model.get("graph") || {}).nodes) fitView();
  });
  ro.observe(gbox);

  syncVariables();
  syncControls();
  syncBusy();
  syncMsg();
  syncHeight();
  drawGraph();

  return () => {
    ro.disconnect();
    clearTimeout(edgeTimer);
    for (const [name, f] of handlers) model.off(name, f);
  };
}

export default { render };
'''

_CSS = r'''
.mfc {
  font-family: var(--jp-ui-font-family, system-ui, sans-serif);
  font-size: 13px;
  color: var(--jp-ui-font-color1, #222);
  background: var(--jp-layout-color0, #fff);
  border: 1px solid var(--jp-border-color2, #ddd);
  border-radius: 6px;
  padding: 8px;
}
.mfc-bar { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin-bottom: 6px; }
.mfc-bar label { display: inline-flex; gap: 4px; align-items: center; }
.mfc input, .mfc select, .mfc button {
  font: inherit; color: inherit;
  background: var(--jp-layout-color1, #fff);
  border: 1px solid var(--jp-border-color1, #bbb);
  border-radius: 4px; padding: 2px 6px;
}
.mfc button { cursor: pointer; }
.mfc button:disabled { opacity: 0.4; cursor: default; }
.mfc-var { width: 14em; font-family: var(--jp-code-font-family, monospace); }
.mfc-num { width: 3.5em; }
.mfc-status { margin-left: auto; opacity: 0.7; font-size: 12px; }
.mfc-busy .mfc-graph { opacity: 0.6; }
.mfc-title { font-weight: 600; margin: 2px 0 4px; }
.mfc-msg { color: #c0392b; margin: 2px 0 4px; }
.mfc-msg:empty { display: none; }

.mfc-graph {
  position: relative; overflow: hidden; cursor: grab; touch-action: none;
  border: 1px solid var(--jp-border-color2, #ddd); border-radius: 4px;
  background: var(--jp-layout-color0, #fff);
}
.mfc-graph.mfc-dragging { cursor: grabbing; }
.mfc-gsvg { display: block; width: 100%; height: 100%; user-select: none; }
.mfc-zoom { position: absolute; right: 6px; top: 6px; display: flex; gap: 4px; z-index: 1; }
.mfc-zoom button { padding: 0 8px; opacity: 0.85; }

.mfc-node { cursor: pointer; transition: transform 0.45s ease, opacity 0.35s ease; }
.mfc-node rect { stroke: #333; stroke-width: 0.7; }
.mfc-node text { font-family: monospace; fill: #1040c0; pointer-events: none; }
.mfc-node.mfc-center rect { stroke-width: 2.4; }
.mfc-node:hover rect { stroke-width: 2; }
.mfc-enter, .mfc-leave { opacity: 0; }
.mfc-leave { pointer-events: none; }

.mfc-edge { fill: none; opacity: 0.75; transition: opacity 0.3s ease; }
.mfc-pos { stroke: #3b6fc4; }
.mfc-neg { stroke: #c9463d; }
.mfc-none { stroke: #999999; }
.mfc-back { stroke-dasharray: 5 3; }
.mfc-focus .mfc-node:not(.mfc-hl) { opacity: 0.25; }
.mfc-focus .mfc-edge:not(.mfc-hl) { opacity: 0.08; }
.mfc-focus .mfc-edge.mfc-hl { opacity: 1; }
.mfc .mfc-edge.mfc-hidden { opacity: 0; }
'''


if AnyWidget is not None:

    class CausalityViewer(AnyWidget):
        '''The causality viewer, made by model.causality

        The traits var, up, down, filter, lag, des and engine can also be set from Python,
        and the graph follows. graph is the data which is drawn.'''
        _esm = _ESM
        _css = _CSS

        var = traitlets.Unicode('').tag(sync=True)
        up = traitlets.Int(1).tag(sync=True)
        down = traitlets.Int(0).tag(sync=True)
        filter = traitlets.Float(0.).tag(sync=True)
        lag = traitlets.Bool(False).tag(sync=True)
        des = traitlets.Bool(False).tag(sync=True)
        engine = traitlets.Unicode('auto').tag(sync=True)
        height = traitlets.Unicode('480px').tag(sync=True)
        variables = traitlets.List([]).tag(sync=True)
        graph = traitlets.Dict({}).tag(sync=True)
        busy = traitlets.Bool(False).tag(sync=True)
        message = traitlets.Unicode('').tag(sync=True)

        _settings = ['var', 'up', 'down', 'filter', 'lag', 'des', 'engine']

        def __init__(self, mmodel, var, fontsize=12, max_nodes=300, **kwargs):
            super().__init__(var=var.upper(), variables=sorted(mmodel.allvar.keys()), **kwargs)
            self.mmodel = mmodel
            self.fontsize = fontsize
            self.max_nodes = max_nodes
            self._shown = ''
            self.observe(self._refresh, names=self._settings)
            self._refresh()

        def _refresh(self, change=None):
            navn = self.var.upper()
            if navn not in self.mmodel.allvar:
                self.message = f'{navn} is not a variable in the model'
                if self._shown:
                    # back to the variable which is shown, without drawing it again
                    self.unobserve(self._refresh, names=self._settings)
                    self.var = self._shown
                    self.observe(self._refresh, names=self._settings)
                return
            self.busy = True
            try:
                G = causality_graph(self.mmodel, navn, up=self.up, down=self.down,
                                    filter=self.filter, lag=self.lag, des=self.des,
                                    max_nodes=self.max_nodes)
                graph = causality_layout(self.mmodel, G, navn, fontsize=self.fontsize,
                                         engine=self.engine)
                with self.hold_sync():
                    self.graph = graph
                    self.message = ''
                self._shown = navn
            except Exception as e:
                self.message = f'{type(e).__name__}: {e}'
            finally:
                self.busy = False


class Causality_Mixin():
    '''Adds the causality viewer to the model class'''

    def causality(self, var, up=1, down=0, filter=0, lag=False, des=False, engine='auto',
                  **kwargs):
        '''An interactive causality viewer for var, shown in the notebook

        The graph shows the variables which determine var, and the ones var
        determines. Hovering over a variable highlights its links.

        A click on a variable moves the graph to it, Back goes back. Ctrl + mouse
        wheel zooms, dragging moves the graph.

        Parameters:
            var: the variable in the center
            up: levels of variables which determine var
            down: levels of variables which var determines
            filter: prune the branches where the attribution is below filter pct
            lag: show the lagged variables as separate boxes
            des: show the descriptions in the boxes
            engine: how the boxes are placed, dot uses the graphviz dot program,
                    grandalf the same method in pure Python, nx simple columns.
                    auto, the default, takes the first of them which works
            height: height of the graph, default '480px'
            fontsize: size of the letters in the boxes, default 12
            max_nodes: the largest graph which is drawn, default 300
        '''
        if AnyWidget is None:
            raise ImportError('The causality viewer needs anywidget, install it with:\n'
                              '    conda install -c conda-forge anywidget\n'
                              'or  pip install anywidget')
        return CausalityViewer(self, var, up=up, down=down, filter=filter, lag=lag, des=des,
                               engine=engine, **kwargs)
