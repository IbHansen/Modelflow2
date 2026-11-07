'''A grid for forecasting with a model, which runs in the notebook itself

The grid shows variables in row groups and periods in columns. The lines of a
variable follow its role in the model (see :class:`modelwidget_core.GridCore`):
an exogenous variable has its value; a fixable variable has its value, the fix
value ``_X``, the fix flag ``_D`` and the add factor ``_A``. A known value typed in
the value line of a fixable variable fixes it to that value.

The main use is forecasting: some values are known and some assumptions have to
be adjusted. Edit, press Solve, look at the new forecast, adjust, Solve again.
A name in "keep as" keeps the solution, so forecast rounds can be compared with
the usual keep tools.

It is an anywidget, so it works where the notebook runs: JupyterLab, Notebook 7,
VS Code and JupyterLite. The javascript and the css are strings in this module,
because the package ships the modules listed in pyproject py-modules only. The
javascript only edits cells and sends requests; all the rules are in Python, in
:class:`modelwidget_core.GridCore`.

Use::

    mpak.gridwidget('#Headline PAKNECONPRVTKN PAKGGREVCO2CER', start=2025, end=2030)

The grid also works as the input of ``updatewidget(mpak, grid)``.
'''

import math

from modelwidget_core import GridCore, GRID_TOOL_OPS

try:
    import traitlets
    from anywidget import AnyWidget
except ImportError:          # the grid is not there, the rest of modelflow works
    AnyWidget = None


_ESM = r'''
function h(tag, attrs, parent, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

// a number as typed or pasted: 1234.5  1,234.5  1.234,5  1234,5
function parseNumber(text) {
  let t = String(text).trim().replace(/\s/g, "");
  if (t === "") return null;
  const hasC = t.includes(","), hasD = t.includes(".");
  if (hasC && hasD) {
    t = t.lastIndexOf(",") > t.lastIndexOf(".") ? t.replace(/\./g, "").replace(",", ".")
                                                : t.replace(/,/g, "");
  } else if (hasC) {
    t = (t.match(/,/g) || []).length > 1 ? t.replace(/,/g, "") : t.replace(",", ".");
  }
  const v = Number(t);
  if (!Number.isFinite(v)) throw new Error(`"${text}" is not a number`);
  return v;
}

// +x adds to the shown value, *x multiplies, x% changes by x pct, else a number
function parseInput(text, cur) {
  const t = String(text).trim();
  if (t === "") return null;
  const rel = /^\+./.test(t) || /^\*./.test(t) || /%$/.test(t);
  if (rel && (cur === null || cur === undefined)) throw new Error("No value to change");
  if (/^\+./.test(t)) return cur + parseNumber(t.slice(1));
  if (/^\*./.test(t)) return cur * parseNumber(t.slice(1));
  if (/%$/.test(t)) return cur * (1 + parseNumber(t.slice(0, -1)) / 100);
  return parseNumber(t);
}

function parseFlag(text) {
  const t = String(text).trim().toLowerCase();
  if (t === "") return null;
  if (["1", "x", "y", "yes", "true", "✓"].includes(t)) return 1;
  if (["0", "n", "no", "false"].includes(t)) return 0;
  throw new Error(`"${text}" is not a fix flag, use 1 or 0`);
}

function render({ model, el }) {
  const handlers = [];
  const on = (name, f) => { model.on(name, f); handlers.push([name, f]); };
  const root = h("div", {class: "mfg", "data-lm-suppress-shortcuts": "true"}, el);
  for (const t of ["keydown", "keyup", "keypress"]) root.addEventListener(t, e => e.stopPropagation());

  // ---- toolbar -------------------------------------------------------------
  const title = h("div", {class: "mfg-title"}, root);
  const bar = h("div", {class: "mfg-bar"}, root);
  const solveBtn = h("button", {class: "mfg-solve", title: "Solve the model with the edits"}, bar, "Solve");
  const nameIn = h("input", {class: "mfg-name", placeholder: "keep as (optional)", spellcheck: "false",
                             title: "When a name is given the solution is kept, to compare forecast rounds"}, bar);
  const resetBtn = h("button", {title: "Remove all edits"}, bar, "Reset");
  const updBtn = h("button", {title: "Show the edits as upd text"}, bar, "Show upd");
  const sep = h("span", {class: "mfg-sep"}, bar, "Selection:");
  const opSel = h("select", {title: "Operator for the selected cells"}, bar);
  const valIn = h("input", {class: "mfg-val", placeholder: "value", title: "Value for the operator"}, bar);
  const applyBtn = h("button", {title: "Apply the operator to the selected editable cells"}, bar, "Apply");
  const fixBtn = h("button", {title: "Fix the selected fixable variables at the values shown"}, bar, "Fix at shown");
  const status = h("span", {class: "mfg-status"}, bar);
  const msg = h("div", {class: "mfg-msg"}, root);
  const updPre = h("pre", {class: "mfg-upd"}, root);
  const wrap = h("div", {class: "mfg-wrap"}, root);
  const table = h("table", {class: "mfg-table"}, wrap);
  const sink = h("textarea", {class: "mfg-sink", "aria-label": "grid keyboard input", spellcheck: "false"}, root);
  h("div", {class: "mfg-help"}, root,
    "Type a number, +x to add, *x to multiply, x% to change by pct. A value typed in the value line of a " +
    "fixable variable fixes it. Delete clears, Ctrl+C/Ctrl+V copy and paste (also from Excel), " +
    "Ctrl+R fills right, F2 or double click edits, click a variable name to fold its lines.");

  for (const op of model.get("ops")) h("option", {value: op}, opSel, op);

  // ---- state ---------------------------------------------------------------
  let rows = [], vis = [], cols = [], data = {}, edits = {}, cells = [];
  const collapsed = new Set();
  let anchor = [0, 0], cursor = [0, 0], dragging = false, editing = null, selCells = [];
  const fmts = {};
  // decimals per line: dec, more when all values of the line are below 1 (add factors)
  function formatter(row) {
    if (fmts[row.key]) return fmts[row.key];
    const dec = model.get("dec");
    const vals = (data[row.key] || []).filter(v => v !== null && v !== 0).map(Math.abs);
    const top = vals.length ? Math.max(...vals) : 1;
    const d = top >= 1 ? dec : Math.min(8, dec + Math.ceil(-Math.log10(top)) + 1);
    const f = new Intl.NumberFormat("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
    return fmts[row.key] = v => { const t = f.format(v); return /^-0(\.0+)?$/.test(t) ? t.slice(1) : t; };
  }

  const editable = (r, c) => vis[r] && vis[r].editable && !cols[c].hist;
  const value = (r, c) => { const d = data[vis[r].key]; return d ? d[c] : null; };
  const isEdited = (r, c) => {
    if (cols[c].hist) return false;
    const e = edits[vis[r].target]; return !!e && e[cols[c].p] !== undefined;
  };
  const show = (row, v) => row.kind === "flag" ? (v === 1 ? "✓" : "") : (v === null || v === undefined ? "" : formatter(row)(v));
  const send = m => model.send(m);
  const setMsg = t => { msg.textContent = t; };

  function selRect() {
    return [Math.min(anchor[0], cursor[0]), Math.min(anchor[1], cursor[1]),
            Math.max(anchor[0], cursor[0]), Math.max(anchor[1], cursor[1])];
  }
  function selected() {          // [r, c] of the selection
    const [r0, c0, r1, c1] = selRect(), out = [];
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) out.push([r, c]);
    return out;
  }

  function draw() {
    rows = model.get("rows");
    data = model.get("data");
    edits = model.get("edits");
    for (const k of Object.keys(fmts)) delete fmts[k];
    cols = [...model.get("hist_periods").map(p => ({p, hist: true})),
            ...model.get("periods").map(p => ({p, hist: false}))];
    vis = rows.filter(r => r.first || !collapsed.has(r.group));
    title.textContent = model.get("heading");

    table.textContent = "";
    const head = h("tr", {}, h("thead", {}, table));
    h("th", {class: "mfg-var"}, head, "Variable");
    h("th", {class: "mfg-line"}, head, "Line");
    for (const col of cols) h("th", {class: col.hist ? "mfg-hist" : ""}, head, col.p);
    const body = h("tbody", {}, table);
    cells = [];
    vis.forEach((row, r) => {
      const tr = h("tr", {class: row.first ? "mfg-first" : ""}, body);
      const tv = h("td", {class: "mfg-var", title: `${row.var}\n${row.des}`}, tr);
      if (row.first) {
        const many = rows.some(x => x.group === row.group && !x.first);
        tv.textContent = (many ? (collapsed.has(row.group) ? "▸ " : "▾ ") : "  ") + row.var;
        tv.dataset.group = row.group;
        tv.classList.add("mfg-fold");
      }
      h("td", {class: "mfg-line", title: row.key}, tr, row.label);
      const line = [];
      cols.forEach((col, c) => {
        const v = value(r, c);
        const cls = ["mfg-cell", `mfg-k-${row.kind}`];
        if (!editable(r, c)) cls.push("mfg-ro");
        if (col.hist) cls.push("mfg-hist");
        if (isEdited(r, c)) cls.push("mfg-edited");
        const td = h("td", {class: cls.join(" ")}, tr, show(row, v));
        td.dataset.r = r; td.dataset.c = c;
        line.push(td);
      });
      cells.push(line);
    });
    anchor = clamp(anchor); cursor = clamp(cursor);
    paintSel();
  }

  function clamp([r, c]) {
    return [Math.max(0, Math.min(r, vis.length - 1)), Math.max(0, Math.min(c, cols.length - 1))];
  }

  function paintSel() {
    for (const td of selCells) td.classList.remove("mfg-sel", "mfg-cur");
    selCells = [];
    if (!vis.length) return;
    for (const [r, c] of selected()) { const td = cells[r][c]; td.classList.add("mfg-sel"); selCells.push(td); }
    const cur = cells[cursor[0]][cursor[1]];
    cur.classList.add("mfg-cur"); selCells.push(cur);
  }

  function move(dr, dc, extend) {
    cursor = clamp([cursor[0] + dr, cursor[1] + dc]);
    if (!extend) anchor = cursor.slice();
    paintSel();
    cells[cursor[0]][cursor[1]].scrollIntoView({block: "nearest", inline: "nearest"});
  }

  function focusSink() { sink.focus({preventScroll: true}); }

  // ---- editing -------------------------------------------------------------
  function startEdit(r, c, initial) {
    if (!editable(r, c)) { setMsg(`${vis[r].label} of ${vis[r].var} in ${cols[c].p} can not be edited`); return; }
    const td = cells[r][c], v = value(r, c);
    const input = h("input", {class: "mfg-input", spellcheck: "false"});
    input.value = initial !== null ? initial
                : (vis[r].kind === "flag" ? (v === null ? "" : String(v)) : (v === null ? "" : String(v)));
    td.textContent = "";
    td.appendChild(input);
    editing = {r, c, input};
    input.focus();
    if (initial === null) input.select();
    input.addEventListener("keydown", e => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); commit(); move(1, 0, false); }
      else if (e.key === "Tab") { e.preventDefault(); commit(); move(0, e.shiftKey ? -1 : 1, false); }
      else if (e.key === "Escape") { e.preventDefault(); cancel(); }
      else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault(); commit(); move(e.key === "ArrowUp" ? -1 : 1, 0, false);
      }
    });
    input.addEventListener("blur", () => { if (editing && editing.input === input) commit(); });
  }

  function commit() {
    if (!editing) return;
    const {r, c, input} = editing;
    editing = null;
    const row = vis[r];
    let v;
    try {
      v = row.kind === "flag" ? parseFlag(input.value) : parseInput(input.value, value(r, c));
    } catch (err) {
      setMsg(err.message); cells[r][c].textContent = show(row, value(r, c)); focusSink(); return;
    }
    cells[r][c].textContent = show(row, v === null ? value(r, c) : v);
    setMsg("");
    send({cmd: "set", cells: [[row.key, cols[c].p, v]]});
    focusSink();
  }

  function cancel() {
    if (!editing) return;
    const {r, c} = editing;
    editing = null;
    cells[r][c].textContent = show(vis[r], value(r, c));
    focusSink();
  }

  function editableSelected() { return selected().filter(([r, c]) => editable(r, c)); }

  function clearSelected() {
    const todo = editableSelected().filter(([r, c]) => isEdited(r, c));
    if (todo.length) send({cmd: "set", cells: todo.map(([r, c]) => [vis[r].key, cols[c].p, null])});
  }

  function fillRight() {
    const [r0, c0, r1, c1] = selRect(), out = [];
    for (let r = r0; r <= r1; r++) {
      const v = value(r, c0);
      if (v === null) continue;
      for (let c = c0 + 1; c <= c1; c++) if (editable(r, c)) out.push([vis[r].key, cols[c].p, v]);
    }
    if (out.length) send({cmd: "set", cells: out});
  }

  function copyText() {
    const [r0, c0, r1, c1] = selRect(), lines = [];
    for (let r = r0; r <= r1; r++) {
      const line = [];
      for (let c = c0; c <= c1; c++) { const v = value(r, c); line.push(v === null ? "" : String(v)); }
      lines.push(line.join("\t"));
    }
    return lines.join("\n");
  }

  function paste(text) {
    const lines = text.replace(/\r/g, "").split("\n");
    while (lines.length && lines[lines.length - 1] === "") lines.pop();
    if (!lines.length) return;
    const block = lines.map(l => l.split("\t"));
    const out = [];
    let skipped = 0, bad = 0;
    const put = (r, c, t) => {
      if (r >= vis.length || c >= cols.length) return;
      if (!editable(r, c)) { if (t.trim() !== "") skipped++; return; }
      try {
        const v = vis[r].kind === "flag" ? parseFlag(t) : parseNumber(t);
        if (v !== null) out.push([vis[r].key, cols[c].p, v]);
      } catch (err) { bad++; }
    };
    const [r0, c0, r1, c1] = selRect();
    if (block.length === 1 && block[0].length === 1) {          // one value fills the selection
      for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) put(r, c, block[0][0]);
    } else {
      block.forEach((line, i) => line.forEach((t, j) => put(r0 + i, c0 + j, t)));
    }
    if (out.length) send({cmd: "set", cells: out});
    setMsg([skipped ? `${skipped} values pasted on cells which can not be edited were left out` : "",
            bad ? `${bad} values were not numbers` : ""].filter(Boolean).join(", "));
  }

  // ---- events --------------------------------------------------------------
  table.addEventListener("mousedown", e => {
    const fold = e.target.closest("td.mfg-fold");
    if (fold) {
      const g = Number(fold.dataset.group);
      collapsed.has(g) ? collapsed.delete(g) : collapsed.add(g);
      draw(); e.preventDefault(); focusSink(); return;
    }
    const td = e.target.closest("td.mfg-cell");
    if (!td || (editing && e.target === editing.input)) return;
    if (editing) commit();
    const rc = [Number(td.dataset.r), Number(td.dataset.c)];
    cursor = rc;
    if (!e.shiftKey) anchor = rc.slice();
    dragging = true;
    paintSel();
    e.preventDefault();
    focusSink();
  });
  table.addEventListener("mouseover", e => {
    if (!dragging) return;
    const td = e.target.closest("td.mfg-cell");
    if (!td) return;
    cursor = [Number(td.dataset.r), Number(td.dataset.c)];
    paintSel();
  });
  const up = () => { dragging = false; };
  document.addEventListener("mouseup", up);
  table.addEventListener("dblclick", e => {
    const td = e.target.closest("td.mfg-cell");
    if (td) startEdit(Number(td.dataset.r), Number(td.dataset.c), null);
  });

  sink.addEventListener("keydown", e => {
    e.stopPropagation();
    if (!vis.length) return;
    const ctrl = e.ctrlKey || e.metaKey, k = e.key;
    const arrows = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]};
    if (arrows[k]) { e.preventDefault(); move(...arrows[k], e.shiftKey); }
    else if (k === "Tab") { e.preventDefault(); move(0, e.shiftKey ? -1 : 1, false); }
    else if (k === "Enter") { e.preventDefault(); move(e.shiftKey ? -1 : 1, 0, false); }
    else if (k === "F2") { e.preventDefault(); startEdit(cursor[0], cursor[1], null); }
    else if (k === "Delete" || k === "Backspace") { e.preventDefault(); clearSelected(); }
    else if (ctrl && (k === "c" || k === "C")) {
      sink.value = copyText(); sink.select();             // the browser copies the selected text
      setTimeout(() => { sink.value = ""; }, 0);
    }
    else if (ctrl && (k === "r" || k === "R")) { e.preventDefault(); fillRight(); }
    else if (ctrl) { /* Ctrl+V and the browser's own keys */ }
    else if (k === " " && vis[cursor[0]].kind === "flag" && editable(cursor[0], cursor[1])) {
      e.preventDefault();
      send({cmd: "set", cells: [[vis[cursor[0]].key, cols[cursor[1]].p, value(cursor[0], cursor[1]) === 1 ? 0 : 1]]});
    }
    else if (k.length === 1 && !e.altKey) { e.preventDefault(); startEdit(cursor[0], cursor[1], k); }
  });
  sink.addEventListener("paste", e => {
    e.preventDefault();
    paste((e.clipboardData || window.clipboardData).getData("text"));
  });

  const selPeriods = () => { const [, c0, , c1] = selRect(); const out = [];
    for (let c = c0; c <= c1; c++) if (!cols[c].hist) out.push(cols[c].p); return out; };
  const selRows = () => { const [r0, , r1] = selRect(); return vis.slice(r0, r1 + 1); };

  solveBtn.addEventListener("click", () => {
    send(model.get("solve_button") ? {cmd: "solve", name: nameIn.value.trim()} : {cmd: "refresh"});
  });
  resetBtn.addEventListener("click", () => send({cmd: "reset"}));
  updBtn.addEventListener("click", () => {
    if (updPre.textContent) { model.set("upd_text", ""); model.save_changes(); }
    else send({cmd: "upd"});
  });
  applyBtn.addEventListener("click", () => {
    const keys = [...new Set(selRows().filter(r => r.editable).map(r => r.key))];
    if (!keys.length || !selPeriods().length) { setMsg("Select editable cells first"); return; }
    send({cmd: "tool", keys, periods: selPeriods(), op: opSel.value, value: valIn.value});
  });
  fixBtn.addEventListener("click", () => {
    const vars = [...new Set(selRows().filter(r => r.role === "fixable").map(r => r.var))];
    if (!vars.length || !selPeriods().length) { setMsg("Select cells of fixable variables first"); return; }
    send({cmd: "fix", vars, periods: selPeriods()});
  });

  function syncState() {
    msg.textContent = model.get("message");
    updPre.textContent = model.get("upd_text");
    status.textContent = model.get("status");
    solveBtn.textContent = model.get("solve_button") ? "Solve" : "Show last solution";
    nameIn.style.display = model.get("solve_button") ? "" : "none";
    solveBtn.classList.toggle("mfg-pending", model.get("pending"));
    root.classList.toggle("mfg-busy", model.get("busy"));
    for (const b of [solveBtn, resetBtn, applyBtn, fixBtn]) b.disabled = model.get("busy");
    wrap.style.maxHeight = model.get("height");
  }

  on("change:rows", draw);
  on("change:data", draw);
  on("change:edits", draw);
  for (const t of ["message", "upd_text", "status", "pending", "busy", "height", "solve_button"]) on(`change:${t}`, syncState);
  draw();
  syncState();

  return () => {
    document.removeEventListener("mouseup", up);
    for (const [name, f] of handlers) model.off(name, f);
  };
}

export default { render };
'''

_CSS = r'''
.mfg {
  font-family: var(--jp-ui-font-family, system-ui, sans-serif);
  font-size: 13px;
  color: var(--jp-ui-font-color1, #222);
  background: var(--jp-layout-color0, #fff);
  border: 1px solid var(--jp-border-color2, #ddd);
  border-radius: 6px;
  padding: 8px;
  position: relative;
}
.mfg-title { font-weight: 600; margin: 0 0 6px; }
.mfg-bar { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; margin-bottom: 6px; }
.mfg input, .mfg select, .mfg button {
  font: inherit; color: inherit;
  background: var(--jp-layout-color1, #fff);
  border: 1px solid var(--jp-border-color1, #bbb);
  border-radius: 4px; padding: 2px 6px;
}
.mfg button { cursor: pointer; }
.mfg button:disabled { opacity: 0.4; cursor: default; }
.mfg-solve { font-weight: 600; }
.mfg-solve.mfg-pending { background: #f2c94c; border-color: #c9a227; color: #222; }
.mfg-name { width: 11em; }
.mfg-val { width: 6em; }
.mfg-sep { margin-left: 10px; opacity: 0.7; }
.mfg-status { margin-left: auto; opacity: 0.7; font-size: 12px; }
.mfg-msg { color: #c0392b; margin: 2px 0 4px; white-space: pre-wrap; }
.mfg-msg:empty, .mfg-upd:empty { display: none; }
.mfg-upd {
  font-family: var(--jp-code-font-family, monospace); font-size: 12px;
  background: var(--jp-layout-color2, #f3f3f3); padding: 6px; margin: 2px 0 6px;
  max-height: 160px; overflow: auto; user-select: text;
}
.mfg-help { opacity: 0.65; font-size: 11px; margin-top: 4px; }
.mfg-busy .mfg-wrap { opacity: 0.6; }

.mfg-wrap {
  overflow: auto; max-width: 100%;
  border: 1px solid var(--jp-border-color2, #ddd); border-radius: 4px;
}
.mfg-table { border-collapse: separate; border-spacing: 0; user-select: none; }
.mfg-table th, .mfg-table td {
  padding: 2px 8px; white-space: nowrap;
  border-right: 1px solid var(--jp-border-color3, #eee);
  border-bottom: 1px solid var(--jp-border-color3, #eee);
}
.mfg-table thead th {
  position: sticky; top: 0; z-index: 2;
  background: var(--jp-layout-color2, #f3f3f3); font-weight: 600; text-align: right;
}
.mfg-table .mfg-var, .mfg-table .mfg-line { position: sticky; z-index: 1; text-align: left;
  background: var(--jp-layout-color1, #fafafa); }
.mfg-table .mfg-var { left: 0; width: 13em; min-width: 13em; max-width: 13em; overflow: hidden;
  text-overflow: ellipsis; font-family: var(--jp-code-font-family, monospace); font-size: 12px; }
.mfg-table .mfg-line { left: calc(13em + 17px); min-width: 9em; color: var(--jp-ui-font-color2, #666); }
.mfg-table thead .mfg-var, .mfg-table thead .mfg-line { z-index: 3; background: var(--jp-layout-color2, #f3f3f3); }
.mfg-fold { cursor: pointer; }
.mfg-first td { border-top: 1px solid var(--jp-border-color1, #bbb); }

.mfg-cell { text-align: right; min-width: 7em; font-variant-numeric: tabular-nums; cursor: cell; }
.mfg-cell.mfg-ro { color: var(--jp-ui-font-color2, #777); background: var(--jp-layout-color1, #fafafa); }
.mfg-cell.mfg-hist { color: var(--jp-ui-font-color2, #777); background: var(--jp-layout-color2, #f0f0f0); }
.mfg-cell.mfg-k-flag { text-align: center; }
.mfg-cell.mfg-k-value { font-weight: 600; }
.mfg-cell.mfg-edited { background: rgba(242, 201, 76, 0.35); color: var(--jp-ui-font-color0, #000); }
.mfg-cell.mfg-sel { background-image: linear-gradient(rgba(33, 150, 243, 0.18), rgba(33, 150, 243, 0.18)); }
.mfg-cell.mfg-cur { outline: 2px solid #2196f3; outline-offset: -2px; }
.mfg-input {
  width: 100%; box-sizing: border-box; text-align: right; padding: 0 2px !important;
  border: none !important; outline: none; background: transparent !important;
}
.mfg-sink { position: absolute; left: -10000px; top: 0; width: 1px; height: 1px; opacity: 0; }
'''


def _num(x):
    '''A value as it can be sent to javascript: a finite float or None'''
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


if AnyWidget is not None:

    class ScenarioGrid(AnyWidget):
        '''The forecast grid, made by model.gridwidget

        ``self.core`` is the :class:`modelwidget_core.GridCore` with the edits,
        ``self.solution`` the latest solution (None before the first Solve).
        The grid has ``update_df``, ``reset`` and ``datawidget``, so it can also be
        the input of ``updatewidget``; then use ``solve_button=False``.'''
        _esm = _ESM
        _css = _CSS

        heading = traitlets.Unicode('').tag(sync=True)
        periods = traitlets.List([]).tag(sync=True)
        hist_periods = traitlets.List([]).tag(sync=True)
        rows = traitlets.List([]).tag(sync=True)
        data = traitlets.Dict({}).tag(sync=True)
        edits = traitlets.Dict({}).tag(sync=True)
        dec = traitlets.Int(2).tag(sync=True)
        ops = traitlets.List([op for op in ('+', '=', '%', '%growth', '+impulse', '=impulse')
                              if op in GRID_TOOL_OPS]).tag(sync=True)
        height = traitlets.Unicode('420px').tag(sync=True)
        solve_button = traitlets.Bool(True).tag(sync=True)
        pending = traitlets.Bool(False).tag(sync=True)
        busy = traitlets.Bool(False).tag(sync=True)
        message = traitlets.Unicode('').tag(sync=True)
        status = traitlets.Unicode('').tag(sync=True)
        upd_text = traitlets.Unicode('').tag(sync=True)

        def __init__(self, mmodel, vars, heading='Forecast grid', solve_options=None, height='420px',
                     solve_button=True, **content):
            super().__init__(heading=heading, height=height, solve_button=solve_button)
            self.mmodel = mmodel
            self.core = GridCore.from_model(mmodel, vars, heading=heading, **content)
            self.solve_options = {'silent': 1, 'progressbar': 0, **(solve_options or {})}
            self.solution = None
            self.periods = [str(p) for p in self.core.periods]
            self.hist_periods = [str(p) for p in self.core.hist_periods]
            self.dec = self.core.dec
            self.rows = self._rows()
            self.status = 'Showing the data'
            self.on_msg(self._on_msg)
            self._push()

        def __repr__(self):
            # the text/plain output, shown where the widget is not run (a static book)
            return f'Interactive forecast grid with {len(self.core.groups)} variables, run the notebook to use it'

        # --- what javascript sees ---------------------------------------------

        def _rows(self):
            out = []
            for gi, g in enumerate(self.core.groups):
                for li, line in enumerate(g.lines):
                    out.append({'key': line.line_var, 'target': line.target or line.line_var,
                                'var': g.var, 'des': g.description, 'role': g.role,
                                'kind': line.kind, 'label': line.label, 'editable': line.editable,
                                'group': gi, 'first': li == 0})
            return out

        @property
        def shown_df(self):
            '''The data the grid shows the values from: the latest solution, else the data'''
            return self.core.base if self.solution is None else self.solution

        def _push(self):
            shown = self.core.shown(self.shown_df)
            edits = {}
            for (line_var, p), v in self.core.edits.items():
                edits.setdefault(line_var, {})[str(p)] = v
            with self.hold_sync():
                self.data = {k: [_num(v) for v in shown.loc[k]] for k in shown.index}
                self.edits = edits
                if self.upd_text:
                    self.upd_text = self.core.as_upd() or 'No edits'

        def _on_msg(self, widget, content, buffers):
            cmd = content.get('cmd', '')
            message = ''
            try:
                if cmd == 'set':
                    refused = []
                    for key, per, value in content['cells']:
                        try:
                            if value is None:
                                self.core.clear_cell(key, per)
                            else:
                                self.core.set_cell(key, per, value)
                        except ValueError as e:
                            refused.append(str(e))
                    if refused:
                        message = (refused[0] if len(refused) == 1
                                   else f'{len(refused)} cells were not changed, the first: {refused[0]}')
                    self.pending = True
                elif cmd == 'tool':
                    try:
                        value = float(str(content['value']).replace(',', '.'))
                    except ValueError:
                        raise ValueError(f"The operator needs a number, not {content['value']!r}") from None
                    for key in content['keys']:
                        self.core.apply_tool(key, content['periods'], content['op'], value, df=self.shown_df)
                    self.pending = True
                elif cmd == 'fix':
                    for var in content['vars']:
                        self.core.fix_to_baseline(var, content['periods'], df=self.shown_df)
                    self.pending = True
                elif cmd == 'solve':
                    self.solve(content.get('name', ''))
                    return
                elif cmd == 'refresh':
                    self.solution = self.mmodel.lastdf
                    self.status = 'Showing the last solution of the model'
                    self.pending = False
                elif cmd == 'reset':
                    self.reset()
                    return
                elif cmd == 'upd':
                    self.upd_text = self.core.as_upd() or 'No edits'
                    return
                else:
                    message = f'Unknown request {cmd!r}'
            except Exception as e:
                message = f'{type(e).__name__}: {e}'
            self.message = message
            self._push()

        # --- Python side use --------------------------------------------------

        def solve(self, name=''):
            '''Solve the model with the edits over the grid periods. With a name the
            solution is kept (``keep=name``) so forecast rounds can be compared.'''
            options = {'start': self.core.periods[0], 'end': self.core.periods[-1], **self.solve_options}
            if name:
                options['keep'] = name
            self.busy = True
            self.message = ''
            try:
                self.solution = self.mmodel(self.core.current(), **options)
                self.pending = False
                self.status = f'Solved{" and kept as " + repr(name) if name else ""}'
            except Exception as e:
                self.message = f'The model did not solve: {type(e).__name__}: {e}'
            finally:
                self.busy = False
            self._push()
            return self.solution

        def as_upd(self):
            '''The edits as upd text'''
            return self.core.as_upd()

        # --- so updatewidget can use the grid as its input -----------------------

        @property
        def datawidget(self):
            return self

        def update_df(self, df, current_per=None):
            self.core.update_df(df, current_per)

        def reset(self, g=None):
            '''Remove all edits and show the data again'''
            self.core.reset()
            self.solution = None
            self.pending = False
            self.status = 'Showing the data'
            self.message = ''
            self._push()


class Grid_Mixin():
    '''Adds the forecast grid to the model class'''

    def gridwidget(self, vars, start='', end='', **kwargs):
        '''An interactive grid to enter known values and adjust assumptions, then solve

        Each variable gets a group of lines from its role in the model: an exogenous
        variable its value; a fixable variable its value, fix value (_X), fix flag (_D)
        and add factor (_A). A value typed in the value line of a fixable variable
        fixes it to that value.

        Parameters:
            vars: variables, a pattern or a list. #Group names, wildcards and -removals mix
            start, end: the periods in the grid, default the current period
            history: periods before start which are shown read only, default 2
            show_growth: also show the growth in pct of the values
            show_fitted: also show V_FITTED and diff = V - V_FITTED, where it exists
            fitted_diff: 'level' (default) or 'pct'
            value_fixes: a value typed for a fixable variable fixes it, default True
            auto_dummy: typing a fix value sets the fix flag, default True
            lines: {var: [kinds]} to show fewer lines, kinds are value growth fix flag add fitted diff
            df: the data, default the model's basedf
            solve_options: options for solving the model, like {'solver': 'newton'}
            solve_button: False when the grid is the input of updatewidget
            dec: decimals shown, default 2
            height: height of the grid, default '420px'
        '''
        if AnyWidget is None:
            raise ImportError('The forecast grid needs anywidget, install it with:\n'
                              '    conda install -c conda-forge anywidget\n'
                              'or  pip install anywidget')
        with self.set_smpl(start, end):
            return ScenarioGrid(self, vars, **kwargs)
