"use strict";
(self["webpackChunkmf_workbench"] = self["webpackChunkmf_workbench"] || []).push([["lib_index_js"],{

/***/ "./lib/index.js"
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @lumino/widgets */ "webpack/sharing/consume/default/@lumino/widgets");
/* harmony import */ var _lumino_widgets__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_lumino_widgets__WEBPACK_IMPORTED_MODULE_2__);



const COMM_TARGET = 'mf_workbench';
/**
 * The workbench panel. All content is plain DOM; the kernel does the
 * heavy lifting and sends back SVG / JSON over a comm.
 */
class WorkbenchWidget extends _lumino_widgets__WEBPACK_IMPORTED_MODULE_2__.Widget {
    constructor(tracker) {
        super();
        this._comm = null;
        this._notebookLabel = '';
        this._varsPattern = '*';
        this._tracker = tracker;
        this.addClass('mf-workbench');
        this.node.innerHTML = `
      <div class="mf-toolbar">
        <label>Model</label><select class="mf-model"></select>
        <label>Variable</label>
        <input class="mf-var" type="text" size="12" placeholder="e.g. GDP" />
        <label>Up</label><input class="mf-up" type="number" value="1" min="0" />
        <label>Down</label><input class="mf-down" type="number" value="1" min="0" />
        <button class="mf-go">Show</button>
        <button class="mf-connect" title="Bind to the current notebook's kernel">Reconnect</button>
      </div>
      <div class="mf-status"></div>
      <div class="mf-tabs">
        <button data-pane="graph" class="mf-active">Dependencies</button>
        <button data-pane="vars">Variables</button>
        <button data-pane="att">Attribution</button>
      </div>
      <div class="mf-pane" data-pane="graph"><p>Connect to a notebook, then pick a variable.</p></div>
      <div class="mf-pane" data-pane="vars" hidden></div>
      <div class="mf-pane" data-pane="att" hidden></div>
    `;
        this._status = this._el('.mf-status');
        this._modelSel = this._el('.mf-model');
        this._varInput = this._el('.mf-var');
        this._upInput = this._el('.mf-up');
        this._downInput = this._el('.mf-down');
        this._el('.mf-go').onclick = () => this._refreshVar();
        this._el('.mf-connect').onclick = () => {
            void this._connect();
        };
        this._varInput.onkeydown = ev => {
            if (ev.key === 'Enter') {
                this._refreshVar();
            }
        };
        for (const btn of this.node.querySelectorAll('.mf-tabs button')) {
            btn.onclick = () => { var _a; return this._showPane((_a = btn.dataset.pane) !== null && _a !== void 0 ? _a : 'graph'); };
        }
        void this._connect();
    }
    dispose() {
        var _a;
        (_a = this._comm) === null || _a === void 0 ? void 0 : _a.close();
        super.dispose();
    }
    // ---------------------------------------------------------- comm layer
    async _connect() {
        var _a, _b;
        (_a = this._comm) === null || _a === void 0 ? void 0 : _a.close();
        this._comm = null;
        const panel = this._tracker.currentWidget;
        const kernel = (_b = panel === null || panel === void 0 ? void 0 : panel.sessionContext.session) === null || _b === void 0 ? void 0 : _b.kernel;
        if (!panel || !kernel) {
            this._setStatus('No notebook with a running kernel is active.');
            return;
        }
        this._setStatus(`Connecting to ${panel.title.label} ...`);
        // Make sure the kernel side is registered (idempotent).
        await kernel.requestExecute({
            code: 'import mf_workbench.kernel',
            silent: true,
            store_history: false,
            stop_on_error: false
        }).done;
        const comm = kernel.createComm(COMM_TARGET);
        comm.onMsg = (msg) => {
            this._onReply(msg.content.data);
        };
        comm.onClose = () => {
            if (this._comm === comm) {
                this._comm = null;
                this._setStatus('Comm closed. Is mf_workbench installed in the kernel environment?');
            }
        };
        comm.open({});
        this._comm = comm;
        this._notebookLabel = panel.title.label;
        this._send({ action: 'list' });
    }
    _send(data) {
        if (!this._comm || this._comm.isDisposed) {
            this._setStatus('Not connected - press Reconnect.');
            return;
        }
        void this._comm.send(data);
    }
    // ------------------------------------------------------------ requests
    _refreshVar() {
        const v = this._varInput.value.trim();
        if (!v) {
            this._setStatus('Enter a variable name.');
            return;
        }
        const model = this._modelSel.value;
        this._send({
            action: 'graph',
            model,
            var: v,
            up: Number(this._upInput.value),
            down: Number(this._downInput.value)
        });
        this._send({ action: 'attribution', model, var: v });
        this._setStatus(`Requested ${v} from ${model || '(first model)'} ...`);
    }
    // ------------------------------------------------------------- replies
    _onReply(d) {
        var _a;
        if (d.error) {
            this._setStatus(d.error);
            return;
        }
        if (d.models) {
            this._modelSel.innerHTML = '';
            for (const name of d.models) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                this._modelSel.appendChild(opt);
            }
            this._setStatus(d.models.length
                ? `Connected to ${this._notebookLabel}. ` +
                    `Models: ${d.models.join(', ')}`
                : `Connected to ${this._notebookLabel}, ` +
                    'but no model instance found yet.');
            if (d.models.length) {
                this._send({
                    action: 'vars',
                    model: this._modelSel.value,
                    pattern: this._varsPattern
                });
            }
            return;
        }
        if (d.svg !== undefined) {
            this._renderGraph(d.svg);
            this._showPane('graph');
            this._setStatus('');
            return;
        }
        if (d.rows) {
            this._renderVars(d.rows);
            return;
        }
        if (d.table) {
            this._renderAttribution(d.table, (_a = d.var) !== null && _a !== void 0 ? _a : '');
            return;
        }
    }
    // ------------------------------------------------------------ renderers
    _renderGraph(svg) {
        const pane = this._pane('graph');
        pane.innerHTML = svg;
        // Click a node to make it the new focus variable.
        for (const node of pane.querySelectorAll('svg g.node')) {
            node.addEventListener('click', () => {
                var _a, _b;
                const name = (_b = (_a = node.querySelector('title')) === null || _a === void 0 ? void 0 : _a.textContent) === null || _b === void 0 ? void 0 : _b.trim();
                if (name) {
                    this._varInput.value = name;
                    this._refreshVar();
                }
            });
        }
    }
    _renderVars(rows) {
        const pane = this._pane('vars');
        pane.innerHTML =
            '<input class="mf-filter" type="text" ' +
                'title="model.vlist pattern - space-separated name wildcards (* ?), ' +
                '!text searches descriptions, #GROUP or #ENDO selects a group. ' +
                'Press Enter to apply." ' +
                'placeholder="vlist pattern, e.g. *GDP*  !carbon  #ENDO" />' +
                `<div class="mf-count">${rows.length} variables</div>` +
                '<table class="mf-table"><thead><tr>' +
                '<th>Name</th><th>Kind</th><th>Last value</th><th>Description</th>' +
                '</tr></thead><tbody></tbody></table>';
        // The pattern is evaluated kernel-side by model.vlist - send on Enter.
        const patInput = pane.querySelector('.mf-filter');
        patInput.value = this._varsPattern;
        patInput.onkeydown = ev => {
            if (ev.key === 'Enter') {
                this._varsPattern = patInput.value.trim() || '*';
                this._send({
                    action: 'vars',
                    model: this._modelSel.value,
                    pattern: this._varsPattern
                });
                this._setStatus(`Requested variables matching ${this._varsPattern} ...`);
            }
        };
        const tbody = pane.querySelector('tbody');
        for (const r of rows) {
            const tr = document.createElement('tr');
            const val = r.value === null || r.value === undefined
                ? ''
                : r.value.toLocaleString(undefined, {
                    maximumFractionDigits: 3
                });
            tr.innerHTML =
                `<td>${r.name}</td><td>${r.kind}</td>` +
                    `<td class="mf-num">${val}</td><td>${escapeHtml(r.desc)}</td>`;
            tr.onclick = () => {
                this._varInput.value = r.name;
                this._refreshVar();
            };
            tbody.appendChild(tr);
        }
        this._setStatus('');
    }
    _renderAttribution(t, varName) {
        const pane = this._pane('att');
        let html = `<h4>Attribution for ${escapeHtml(varName)}</h4>`;
        html += '<table class="mf-table"><thead><tr><th></th>';
        for (const c of t.columns) {
            html += `<th>${escapeHtml(String(c))}</th>`;
        }
        html += '</tr></thead><tbody>';
        t.index.forEach((ix, i) => {
            html += `<tr><td>${escapeHtml(String(ix))}</td>`;
            for (const cell of t.data[i]) {
                html += `<td class="mf-num">${cell !== null && cell !== void 0 ? cell : ''}</td>`;
            }
            html += '</tr>';
        });
        html += '</tbody></table>';
        pane.innerHTML = html;
        // No _showPane here: the attribution reply arrives after the graph
        // reply, and switching would steal the pane the user just got.
    }
    // -------------------------------------------------------------- helpers
    _showPane(name) {
        for (const pane of this.node.querySelectorAll('.mf-pane')) {
            pane.hidden = pane.dataset.pane !== name;
        }
        for (const btn of this.node.querySelectorAll('.mf-tabs button')) {
            btn.classList.toggle('mf-active', btn.dataset.pane === name);
        }
    }
    _pane(name) {
        return this.node.querySelector(`.mf-pane[data-pane="${name}"]`);
    }
    _el(sel) {
        return this.node.querySelector(sel);
    }
    _setStatus(text) {
        this._status.textContent = text;
    }
}
function escapeHtml(s) {
    return s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
/**
 * Plugin registration: adds "ModelFlow: Open Model Workbench" to the
 * command palette and opens the panel split-right next to the notebook.
 */
const plugin = {
    id: 'mf-workbench:plugin',
    description: 'ModelFlow model workbench panel',
    autoStart: true,
    requires: [_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.ICommandPalette, _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_1__.INotebookTracker],
    activate: (app, palette, tracker) => {
        const command = 'mf-workbench:open';
        app.commands.addCommand(command, {
            label: 'ModelFlow: Open Model Workbench',
            execute: () => {
                const content = new WorkbenchWidget(tracker);
                const widget = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.MainAreaWidget({ content });
                widget.id = `mf-workbench-${Date.now()}`;
                widget.title.label = 'Model Workbench';
                widget.title.closable = true;
                app.shell.add(widget, 'main', { mode: 'split-right' });
                app.shell.activateById(widget.id);
            }
        });
        palette.addItem({ command, category: 'ModelFlow' });
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }

}]);
//# sourceMappingURL=lib_index_js.293d7fdf438716c3e590.js.map