import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { INotebookTracker } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { Widget } from '@lumino/widgets';

const COMM_TARGET = 'mf_workbench';

interface IVarRow {
  name: string;
  kind: string;
  desc: string;
  value: number | null;
}

interface ISplitTable {
  columns: string[];
  index: (string | number)[];
  data: (number | string | null)[][];
}

interface IReply {
  action?: string;
  error?: string;
  models?: string[];
  svg?: string;
  rows?: IVarRow[];
  table?: ISplitTable;
  var?: string;
}

/**
 * The workbench panel. All content is plain DOM; the kernel does the
 * heavy lifting and sends back SVG / JSON over a comm.
 */
class WorkbenchWidget extends Widget {
  constructor(tracker: INotebookTracker) {
    super();
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
    this._modelSel = this._el<HTMLSelectElement>('.mf-model');
    this._varInput = this._el<HTMLInputElement>('.mf-var');
    this._upInput = this._el<HTMLInputElement>('.mf-up');
    this._downInput = this._el<HTMLInputElement>('.mf-down');

    this._el<HTMLButtonElement>('.mf-go').onclick = () => this._refreshVar();
    this._el<HTMLButtonElement>('.mf-connect').onclick = () => {
      void this._connect();
    };
    this._varInput.onkeydown = ev => {
      if (ev.key === 'Enter') {
        this._refreshVar();
      }
    };

    for (const btn of this.node.querySelectorAll<HTMLButtonElement>(
      '.mf-tabs button'
    )) {
      btn.onclick = () => this._showPane(btn.dataset.pane ?? 'graph');
    }

    void this._connect();
  }

  dispose(): void {
    this._comm?.close();
    super.dispose();
  }

  // ---------------------------------------------------------- comm layer

  private async _connect(): Promise<void> {
    this._comm?.close();
    this._comm = null;

    const panel = this._tracker.currentWidget;
    const kernel = panel?.sessionContext.session?.kernel;
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
    comm.onMsg = (msg: KernelMessage.ICommMsgMsg) => {
      this._onReply(msg.content.data as unknown as IReply);
    };
    comm.onClose = () => {
      if (this._comm === comm) {
        this._comm = null;
        this._setStatus(
          'Comm closed. Is mf_workbench installed in the kernel environment?'
        );
      }
    };
    comm.open({});
    this._comm = comm;
    this._notebookLabel = panel.title.label;
    this._send({ action: 'list' });
  }

  private _send(data: Record<string, unknown>): void {
    if (!this._comm || this._comm.isDisposed) {
      this._setStatus('Not connected - press Reconnect.');
      return;
    }
    void this._comm.send(data as never);
  }

  // ------------------------------------------------------------ requests

  private _refreshVar(): void {
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

  private _onReply(d: IReply): void {
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
      this._setStatus(
        d.models.length
          ? `Connected to ${this._notebookLabel}. ` +
            `Models: ${d.models.join(', ')}`
          : `Connected to ${this._notebookLabel}, ` +
            'but no model instance found yet.'
      );
      if (d.models.length) {
        this._send({ action: 'vars', model: this._modelSel.value });
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
      this._renderAttribution(d.table, d.var ?? '');
      return;
    }
  }

  // ------------------------------------------------------------ renderers

  private _renderGraph(svg: string): void {
    const pane = this._pane('graph');
    pane.innerHTML = svg;
    // Click a node to make it the new focus variable.
    for (const node of pane.querySelectorAll<SVGGElement>('svg g.node')) {
      node.addEventListener('click', () => {
        const name = node.querySelector('title')?.textContent?.trim();
        if (name) {
          this._varInput.value = name;
          this._refreshVar();
        }
      });
    }
  }

  private _renderVars(rows: IVarRow[]): void {
    const pane = this._pane('vars');
    pane.innerHTML =
      '<input class="mf-filter" type="text" ' +
      'placeholder="Filter by name or description..." />' +
      '<table class="mf-table"><thead><tr>' +
      '<th>Name</th><th>Kind</th><th>Last value</th><th>Description</th>' +
      '</tr></thead><tbody></tbody></table>';

    const tbody = pane.querySelector('tbody') as HTMLElement;
    const render = (filter: string) => {
      const f = filter.toLowerCase();
      tbody.innerHTML = '';
      for (const r of rows) {
        if (
          f &&
          !r.name.toLowerCase().includes(f) &&
          !r.desc.toLowerCase().includes(f)
        ) {
          continue;
        }
        const tr = document.createElement('tr');
        const val =
          r.value === null || r.value === undefined
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
    };
    const filterInput = pane.querySelector('.mf-filter') as HTMLInputElement;
    filterInput.oninput = () => render(filterInput.value);
    render('');
  }

  private _renderAttribution(t: ISplitTable, varName: string): void {
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
        html += `<td class="mf-num">${cell ?? ''}</td>`;
      }
      html += '</tr>';
    });
    html += '</tbody></table>';
    pane.innerHTML = html;
    this._showPane('att');
  }

  // -------------------------------------------------------------- helpers

  private _showPane(name: string): void {
    for (const pane of this.node.querySelectorAll<HTMLElement>('.mf-pane')) {
      pane.hidden = pane.dataset.pane !== name;
    }
    for (const btn of this.node.querySelectorAll<HTMLButtonElement>(
      '.mf-tabs button'
    )) {
      btn.classList.toggle('mf-active', btn.dataset.pane === name);
    }
  }

  private _pane(name: string): HTMLElement {
    return this.node.querySelector(
      `.mf-pane[data-pane="${name}"]`
    ) as HTMLElement;
  }

  private _el<T extends HTMLElement = HTMLElement>(sel: string): T {
    return this.node.querySelector(sel) as T;
  }

  private _setStatus(text: string): void {
    this._status.textContent = text;
  }

  private _tracker: INotebookTracker;
  private _comm: Kernel.IComm | null = null;
  private _notebookLabel = '';
  private _status: HTMLElement;
  private _modelSel: HTMLSelectElement;
  private _varInput: HTMLInputElement;
  private _upInput: HTMLInputElement;
  private _downInput: HTMLInputElement;
}

function escapeHtml(s: string): string {
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
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'mf-workbench:plugin',
  description: 'ModelFlow model workbench panel',
  autoStart: true,
  requires: [ICommandPalette, INotebookTracker],
  activate: (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    tracker: INotebookTracker
  ) => {
    const command = 'mf-workbench:open';
    app.commands.addCommand(command, {
      label: 'ModelFlow: Open Model Workbench',
      execute: () => {
        const content = new WorkbenchWidget(tracker);
        const widget = new MainAreaWidget({ content });
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

export default plugin;
