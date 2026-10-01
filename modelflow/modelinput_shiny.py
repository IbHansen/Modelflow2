# -*- coding: utf-8 -*-
"""
modelinput_shiny
================

Draw ModelFlow input widgets with Shiny for Python (Shiny Core) and make a
standalone scenario web app.

It uses the same widget definitions as :func:`modelwidget_input.make_widget`
and the logic in :mod:`modelwidget_core`, so a definition which works in a
notebook works in the app. All widget types are supported: ``base``, ``tab``,
``slide``, ``sumslide``, ``radio``, ``check`` and ``sheet``.

Shiny only: a ``slide`` or ``radio`` definition can have ``'columns': n`` to
show its sliders (radio groups) in at most n columns (``'columns': 1`` is a
vertical list like the notebook). Without it they fill as many columns as fit.
The notebook widgets ignore the key.

The result viewer has the options of :class:`modelwidget_input.keep_plot_widget`
with the same names (see :class:`ViewerOptions`); saving charts to a folder
becomes a download.

:func:`make_app` takes the arguments of :class:`modelwidget_input.updatewidget`
(and a widget made by ``make_widget`` as well as a definition), and
:func:`show_app` shows the app below a notebook cell, so in a notebook::

    w = make_widget(tabdef)
    updatewidget(mpak, w, legend=True)               # ipywidgets
    show_app(make_app(w, mpak, legend=True))         # Shiny, same arguments

Requires ``pip install shiny`` (or ``conda install -c conda-forge shiny``).

Example ``app.py``::

    from modelclass import model
    from modelinput_shiny import make_app

    co2_common = {'value': 0.0, 'min': -50.0, 'max': 50.0, 'step': 1.0, 'op': '+'}
    widgetdef = ['tab', {'content': [
        ('CO2 tax', ['slide', {'heading': 'CO2 tax rate changes', 'content': {
            'Coal': {'var': 'PAKGGREVCO2CER', **co2_common},
            'Gas':  {'var': 'PAKGGREVCO2GER', **co2_common},
            'Oil':  {'var': 'PAKGGREVCO2OER', **co2_common}}}]),
    ]}]

    def load_model():
        mpak, _ = model.modelload('pak.pcim', start=2023, end=2040, run=True)
        return mpak

    app = make_app(widgetdef, model_factory=load_model, selected='PAKNYGDPMKTPKN')

Run it with ``shiny run app.py``, or export it for the browser with
``shinylive export <folder> <site>``.
"""

from __future__ import annotations

import io
import sys
import zipfile
from copy import copy, deepcopy
from typing import Any, Callable

import pandas as pd
from shiny import App, reactive, render, ui

from modelwidget_core import (
    CheckCore,
    ContainerCore,
    BaseCore,
    RadioCore,
    ScenarioRunner,
    SheetCore,
    SlideCore,
    SumSlideCore,
    TabCore,
    ViewerOptions,
    assign_ids,
    viewer_options,
    figs_addname,
    fig_to_image,
    keep_figs,
    make_core,
)


def _get(input, id: str, default: Any = None) -> Any:
    """The value of an input, or ``default`` if it is not shown (yet)."""
    try:
        return input[id]()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Input widgets: core -> Shiny ui
# ---------------------------------------------------------------------------

def _heading(core):
    return ui.h5(core.heading) if core.heading and core.heading != 'Heading' else None


def _columns(items, columns=None, min_width='280px'):
    """Items in a grid.

    Without ``columns`` there are as many columns as fit (each at least
    ``min_width``). With ``columns=n`` there are at most n columns, fewer
    when the window is too narrow for them.
    """
    if not columns:
        return ui.layout_column_wrap(*items, width=min_width, fill=False)
    n = max(1, int(columns))
    col = f'max({min_width}, calc((100% - {n - 1}rem) / {n}))'
    return ui.div(*items, style=f'display: grid; gap: 1rem; '
                                f'grid-template-columns: repeat(auto-fill, minmax({col}, 1fr));')


def _slack_id(core, des):
    return f'{core.control_id(des)}__slack'


def input_ui(core) -> Any:
    """The Shiny ui for a tree of core widgets (see :func:`modelwidget_core.make_core`)."""
    match core:
        case TabCore():
            titles = core.titles
            selected = titles[core.selected_index] if titles else None
            if core.tab:
                return ui.navset_tab(
                    *[ui.nav_panel(title, ui.div(input_ui(child), class_='pt-3'))
                      for title, child in zip(titles, core.children)],
                    id=core.id, selected=selected)
            return ui.accordion(
                *[ui.accordion_panel(title, input_ui(child)) for title, child in zip(titles, core.children)],
                id=core.id, open=selected)

        case BaseCore():
            return ui.div(*[input_ui(child) for child in core.children])

        case SumSlideCore():
            rows = [
                ui.div(
                    ui.div(ui.input_slider(core.control_id(des), des, min=line["min"], max=line["max"],
                                           value=core.values[des], step=line.get("step", 0.01), width='100%'),
                           class_='flex-grow-1'),
                    ui.input_checkbox(_slack_id(core, des), 'Slack', value=slack),
                    class_='d-flex align-items-center gap-3')
                for (des, line), slack in zip(core.lines.items(), core.slack)]
            return ui.div(
                _heading(core),
                ui.p(f'The sliders add up to {core.maxsum:g}; the ones marked slack absorb a change.',
                     class_='text-muted small'),
                *rows)

        case SlideCore():
            sliders = [ui.input_slider(core.control_id(des), des,
                                       min=line["min"], max=line["max"], value=core.values[des],
                                       step=line.get("step", 0.01), width='100%')
                       for des, line in core.lines.items()]
            return ui.div(_heading(core), _columns(sliders, core.widgetdef.get('columns')))

        case RadioCore():
            groups = [ui.input_radio_buttons(core.control_id(des), des,
                                             choices={str(i): label for i, (label, _) in enumerate(cont)},
                                             selected=str(core.selected[des]))
                      for des, cont in core.content.items()]
            return ui.div(_heading(core), _columns(groups, core.widgetdef.get('columns'), '240px'))

        case CheckCore():
            return ui.div(
                _heading(core),
                *[ui.input_checkbox(core.control_id(des), des, value=core.values[des])
                  for des in core.content])

        case SheetCore():
            return ui.div(_heading(core), ui.output_data_frame(core.id))

        case _:
            raise NotImplementedError(f"No Shiny ui for {type(core).__name__}")


# ---------------------------------------------------------------------------
# Input widgets: server side
# ---------------------------------------------------------------------------

def _leaves(core):
    if isinstance(core, ContainerCore):
        for child in core.children:
            yield from _leaves(child)
    else:
        yield core


def _sheet_frame(core: SheetCore) -> pd.DataFrame:
    """The sheet data as shown: the row labels as a first column, periods as text."""
    frame = core.data.astype(float)
    frame.columns = [str(c) for c in frame.columns]
    frame.insert(0, ' ', [str(i) for i in frame.index])
    return frame.reset_index(drop=True)


def _sheet_values(core: SheetCore, shown: pd.DataFrame) -> pd.DataFrame:
    """Back from the shown table (edited cells come as text) to the core's layout."""
    values = shown.drop(columns=shown.columns[0]).apply(pd.to_numeric, errors='coerce')
    values.index = core.org_df_var.index
    values.columns = core.org_df_var.columns
    return values.fillna(0.0)


class InputServer:
    """The server side of the input widgets for one browser session.

    Keeps ``sumslide`` sliders summing to ``maxsum`` while they are moved and
    draws the ``sheet`` tables. :meth:`read` copies all input values into the
    core, :meth:`reset` puts the inputs back to their start values.
    """

    def __init__(self, core, input, output):
        self.core = core
        self.input = input
        self.sheets = {}                         # sheet id -> data frame renderer
        self.sheet_version = reactive.value(0)   # bumped to redraw the sheets on reset
        for leaf in _leaves(core):
            match leaf:
                case SumSlideCore():
                    self._sumslide(leaf)
                case SheetCore():
                    self._sheet(leaf, output)

    def _sumslide(self, core: SumSlideCore) -> None:
        input = self.input
        ids = {des: core.control_id(des) for des in core.lines}
        expected = {}       # values we sent to the browser and wait to get back
        before = {}         # the values before that update

        def near(a, b, des):
            return abs(a - b) <= core.lines[des].get("step", 0.01) / 2 + 1e-12

        @reactive.effect
        def _keep_sum():
            values = {des: input[i]() for des, i in ids.items()}
            slack = [bool(input[_slack_id(core, des)]()) for des in ids]

            if expected:
                if all(near(values[des], expected[des], des) for des in values):
                    # our own update came back (maybe rounded to the steps)
                    expected.clear()
                    core.values.update(values)
                    return
                if all(near(values[des], expected[des], des) or near(values[des], before[des], des)
                       for des in values):
                    return      # only part of our update has come back yet
                expected.clear()    # the user moved a slider meanwhile

            if not any(slack):
                slack[-1] = True
                ui.update_checkbox(_slack_id(core, list(ids)[-1]), value=True)
            core.slack = slack

            changed = [des for des in values if abs(values[des] - core.values[des]) > 1e-12]
            if not changed:
                return
            for des in changed:
                core.set_value(des, values[des])
            if core.rebalance(changed[0]):
                before.clear()
                before.update(values)
                expected.update(core.values)
                for des, i in ids.items():
                    ui.update_slider(i, value=core.values[des])

    def _sheet(self, core: SheetCore, output) -> None:
        version = self.sheet_version

        # drawn even when its tab is not open, so Run can always read it
        @output(id=core.id, suspend_when_hidden=False)
        @render.data_frame
        def _grid():
            version()
            return render.DataGrid(_sheet_frame(core), editable=True, width='100%')

        @_grid.set_patch_fn
        def _number(*, patch):
            # an edited cell comes as text; the table holds numbers
            try:
                return float(str(patch["value"]).replace(',', ''))
            except ValueError:
                return 0.0

        self.sheets[core.id] = _grid

    def read(self, core=None) -> None:
        """Copy the current input values into the core widgets."""
        core = self.core if core is None else core
        input = self.input
        match core:
            case ContainerCore():
                for child in core.children:
                    self.read(child)
            case SumSlideCore():
                pass        # kept up to date while the sliders move
            case SlideCore():
                for des in core.lines:
                    core.set_value(des, input[core.control_id(des)]())
            case RadioCore():
                for des in core.content:
                    core.select(des, int(input[core.control_id(des)]()))
            case CheckCore():
                for des in core.content:
                    core.set_value(des, bool(input[core.control_id(des)]()))
            case SheetCore():
                core.set_data(_sheet_values(core, self.sheets[core.id].data_patched()))

    def reset(self) -> None:
        """Back to the start values, in the core and on the page."""
        self.core.reset()
        for leaf in _leaves(self.core):
            match leaf:
                case SumSlideCore():
                    for des, slack in zip(leaf.lines, leaf.slack):
                        ui.update_slider(leaf.control_id(des), value=leaf.values[des])
                        ui.update_checkbox(_slack_id(leaf, des), value=slack)
                case SlideCore():
                    for des in leaf.lines:
                        ui.update_slider(leaf.control_id(des), value=leaf.values[des])
                case RadioCore():
                    for des in leaf.content:
                        ui.update_radio_buttons(leaf.control_id(des), selected=str(leaf.selected[des]))
                case CheckCore():
                    for des in leaf.content:
                        ui.update_checkbox(leaf.control_id(des), value=leaf.values[des])
        self.sheet_version.set(self.sheet_version() + 1)


# ---------------------------------------------------------------------------
# Result viewer
# ---------------------------------------------------------------------------

_DIFF = {'no': False, 'yes': True, 'pct': 'pct'}
_DIFF_LABELS = {'no': 'No', 'yes': 'Yes', 'pct': 'In percent'}


class ResultViewer:
    """Charts of the kept solutions for one browser session.

    ``runs`` is a reactive value which is bumped when the kept solutions change.
    The ui is drawn in the output ``mf_viewer`` (controls) and ``mf_figs`` (charts).
    """

    def __init__(self, model, runs, opt: ViewerOptions, input, output):
        self.model = model
        self.opt = opt
        self.input = input
        m = model

        old_per = copy(m.current_per)
        with m.set_smpl(*opt.smpl):
            with m.set_smpl_relative(opt.relativ_start, 0):
                show_per = list(m.current_per)
        m.current_per = old_per
        periods = {str(p): p for p in m.lastdf.index}

        allkeep = [set(df.columns) for df in m.keep_solutions.values()]
        keepvar = sorted(allkeep[0].intersection(*allkeep[1:])) if allkeep else []
        keepset = set(keepvar)

        select_scenario = opt.select_scenario and not opt.switch and not opt.short
        if opt.use_var_groups:
            prefix_dict = opt.var_groups or getattr(m, 'var_groups', {}) or {}
        else:
            prefix_dict = {}

        def title(v):
            return f'{(v + " ") if opt.add_var_name else ""}{m.var_description[v] if opt.use_descriptions else v}'

        def names_from(pattern):
            if not pattern:
                return keepvar
            try:
                return [s.upper() for s in m.vlist(pattern) if s in keepset]
            except Exception:
                return []

        def in_groups(names, pats):
            if not pats:
                return names
            gpat = ' '.join(m.string_substitution(p) for p in pats)
            match = set(m.list_names([m.string_substitution(v) for v in names], gpat))
            return [v for v in names if v in match]

        def scenario_keys():
            with m.keepswitch(switch=opt.switch, scenarios='*'):
                return list(m.keep_solutions.keys())

        # start values of the controls
        first_groups = [next(iter(prefix_dict.values()))] if prefix_dict else []
        names0 = in_groups(names_from(opt.selectfrom), first_groups)
        if opt.selected:
            wanted = set(names_from(opt.selected))
            sel0 = [v for v in names0 if v in wanted]
        elif prefix_dict:
            sel0 = names0
        else:
            sel0 = names0[:1]
        diff0 = 'pct' if isinstance(opt.init_dif, str) else ('yes' if opt.init_dif else 'no')

        # ---------------------------------------------------------------- controls
        @output(id='mf_viewer')
        @render.ui
        def _viewer():
            keys = scenario_keys()
            rows = []
            if select_scenario:
                rows.append(ui.layout_columns(
                    ui.input_select('mf_base', 'First scenario', keys, selected=keys[0] if keys else None),
                    ui.input_selectize('mf_next', 'Next scenarios', keys[1:], selected=keys[1:], multiple=True)))
            if opt.showselectfrom:
                rows.append(ui.input_text('mf_selectfrom', 'Display variables (pattern)', value=opt.selectfrom,
                                          width='100%'))
            varsel = ui.input_selectize('mf_vars', 'Select one or more variables',
                                        {v: title(v) for v in names0}, selected=sel0, multiple=True, width='100%')
            if prefix_dict:
                rows.append(ui.layout_columns(
                    varsel,
                    ui.input_selectize('mf_groups', 'Groups', {c: label for label, c in prefix_dict.items()},
                                       selected=first_groups, multiple=True, width='100%'),
                    col_widths=(9, 3)))
            else:
                rows.append(varsel)

            options = [ui.input_radio_buttons('mf_diff', f'Difference to: "{keys[0] if keys else ""}"',
                                              _DIFF_LABELS, selected=diff0, inline=True)]
            if opt.short < 2:
                options.append(ui.input_radio_buttons('mf_legend', 'Legends', {'1': 'Yes', '0': 'No'},
                                                      selected='1' if opt.legend else '0', inline=True))
            if not opt.short:
                options += [ui.input_radio_buttons('mf_showtype', 'Data type', {'level': 'Level', 'growth': 'Growth'},
                                                   inline=True),
                            ui.input_radio_buttons('mf_scale', 'Y-scale', {'linear': 'Linear', 'log': 'Log'},
                                                   inline=True)]
            rows.append(ui.div(*options, class_='d-flex gap-4 flex-wrap'))

            if opt.use_smpl:
                labels = list(periods)
                rows.append(ui.layout_columns(
                    ui.input_select('mf_from', 'From', labels, selected=str(show_per[0])),
                    ui.input_select('mf_to', 'To', labels, selected=str(show_per[-1]))))

            if opt.allow_download and not opt.short:
                rows.append(ui.div(
                    ui.input_checkbox_group('mf_formats', 'Chart files', {'svg': 'svg', 'png': 'png', 'pdf': 'pdf'},
                                            selected=['svg'], inline=True),
                    ui.download_button('mf_download', 'Download charts'),
                    class_='d-flex gap-3 align-items-end flex-wrap'))
            return ui.TagList(*rows)

        # ---------------------------------------------------------------- scenarios
        @reactive.calc
        def keys():
            runs()
            return scenario_keys()

        @reactive.effect
        @reactive.event(keys, ignore_init=True)
        def _new_scenarios():
            if not select_scenario:
                return
            k = keys()
            base = _get(input, 'mf_base')
            base = base if base in k else (k[0] if k else None)
            others = [s for s in k if s != base]
            ui.update_select('mf_base', choices=k, selected=base)
            ui.update_selectize('mf_next', choices=others, selected=others)

        @reactive.effect
        @reactive.event(lambda: _get(input, 'mf_base'), ignore_init=True)
        def _new_base():
            base = _get(input, 'mf_base')
            if base is None:
                return
            others = [s for s in keys() if s != base]
            ui.update_selectize('mf_next', choices=others, selected=others)
            ui.update_radio_buttons('mf_diff', label=f'Difference to: "{base}"')

        @reactive.calc
        def scenarios():
            k = keys()
            if not select_scenario or not k:
                return '|'.join(k)
            base = _get(input, 'mf_base', k[0])
            base = base if base in k else k[0]
            nxt = _get(input, 'mf_next', k[1:]) or []
            return '|'.join([base] + [s for s in nxt if s in k and s != base])

        # ---------------------------------------------------------------- variables
        @reactive.calc
        def choices():
            pattern = _get(input, 'mf_selectfrom', opt.selectfrom) if opt.showselectfrom else opt.selectfrom
            groups = list(_get(input, 'mf_groups', first_groups) or []) if prefix_dict else []
            return in_groups(names_from(pattern), groups), groups

        last_groups = {'groups': first_groups}

        @reactive.effect
        @reactive.event(choices, ignore_init=True)
        def _new_choices():
            names, groups = choices()
            current = list(_get(input, 'mf_vars', []) or [])
            keep = [v for v in current if v in names]
            old = last_groups['groups']
            last_groups['groups'] = groups
            if not keep and old and groups and current:
                # the groups changed: show the same variables for the new groups
                suffixes = {v[len(old[0]):] for v in current if v.startswith(old[0])}
                keep = [f'{n}{s}' for s in suffixes for n in groups if f'{n}{s}' in names]
            if not keep:
                keep = names[:1]
            ui.update_selectize('mf_vars', choices={v: title(v) for v in names}, selected=keep)

        # ---------------------------------------------------------------- charts
        @reactive.calc
        def settings():
            diff = _get(input, 'mf_diff', diff0)
            showtype = _get(input, 'mf_showtype', 'level')
            scale = _get(input, 'mf_scale', 'linear')
            return _DIFF[diff], showtype, scale

        @reactive.calc
        def smpl():
            if not opt.use_smpl:
                return (show_per[0], show_per[-1])
            start = periods.get(_get(input, 'mf_from', ''), show_per[0])
            end = periods.get(_get(input, 'mf_to', ''), show_per[-1])
            return (start, end)

        @reactive.calc
        def figs():
            runs()
            variables = list(_get(input, 'mf_vars', sel0) or [])
            if not variables:
                return {}
            diff, showtype, scale = settings()
            legend = int(_get(input, 'mf_legend', '1' if opt.legend else '0'))
            return keep_figs(m, variables, smpl(), diff=diff, showtype=showtype, scale=scale,
                             legend=legend, dec=opt.dec, vline=opt.vline,
                             switch=opt.switch, scenarios=scenarios())

        @output(id='mf_figs')
        @render.ui
        def _figs():
            f = figs()
            if not f:
                return ui.p('Select one or more variables')
            panels = [(title(v), ui.HTML(fig_to_image(fig))) for v, fig in f.items()]
            match opt.displaytype:
                case 'tab':
                    return ui.navset_tab(*[ui.nav_panel(t, ui.div(h, class_='pt-2')) for t, h in panels])
                case 'accordion':
                    return ui.accordion(*[ui.accordion_panel(t, h) for t, h in panels], open=panels[0][0])
                case _:
                    return ui.div(*[ui.div(ui.h6(t), h, class_='mb-3') for t, h in panels])

        def _zipname():
            diff, showtype, scale = settings()
            return f'charts{figs_addname(showtype, diff, scale)}.zip'

        @output(id='mf_download')
        @render.download_button(filename=_zipname)
        def _download():
            formats = list(_get(input, 'mf_formats', ['svg']) or ['svg'])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                for v, fig in figs().items():
                    for fmt in formats:
                        b = io.BytesIO()
                        fig.savefig(b, format=fmt, bbox_inches='tight')
                        z.writestr(f'{v}.{fmt}', b.getvalue())
            yield buf.getvalue()


# ---------------------------------------------------------------------------
# The scenario app
# ---------------------------------------------------------------------------

# updatewidget / keep_plot_widget arguments which only make sense in a notebook;
# make_app accepts and ignores them, so a call can be moved from the notebook.
NOTEBOOK_ONLY = {'lwupdate', 'outputwidget', 'display_first', 'render_mode', 'save_location',
                 'select_width', 'select_height', 'colab_selector_width'}


def _is_model(obj) -> bool:
    return hasattr(obj, 'basedf') and hasattr(obj, 'keep_solutions')


def _core_template(widget) -> Any:
    """A core tree with ids from a widget definition or a notebook widget
    (made by :func:`modelwidget_input.make_widget`)."""
    core = getattr(widget, 'core', None)
    if core is not None:          # a notebook widget: start from its definition values
        template = deepcopy(core)
        template.reset()
        assign_ids(template)
        return template
    return make_core(widget)


def make_app(widgetdef, mmodel: Any = None, *, model_factory: Callable[[], Any] = None,
             title: str = 'ModelFlow scenarios', basename: str = 'Business as usual', keeppat: str = '*',
             layout: str = 'stacked', sidebar_width: int = 420,
             lwrun: bool = True, lwreset: bool = True, lwsetbas: bool = True,
             varpat: str = None, showvarpat: bool = None, **viewer) -> App:
    """A Shiny app: input widgets, run scenarios, and charts of the kept solutions.

    The arguments are those of :class:`modelwidget_input.updatewidget`, so a
    notebook call ``updatewidget(mmodel, w, ...)`` becomes ``make_app(w, mmodel, ...)``
    (the model may also come first). Notebook-only arguments (see
    ``NOTEBOOK_ONLY``) are accepted and ignored.

    Parameters
    ----------
    widgetdef:
        A widget definition, as for :func:`modelwidget_input.make_widget`, or a
        widget made by it.
    mmodel:
        A model with its ``basedf`` set. All browser sessions share it, so use
        this for one user (or Shinylive, where every user has their own Python).
    model_factory:
        A function returning a new model. Called for every browser session, so
        users on a shared server get their own model. Use this or ``mmodel``.
    basename, keeppat:
        Name of the baseline scenario and pattern of the variables to keep, as
        in :class:`modelwidget_input.updatewidget`.
    layout:
        ``'stacked'``: inputs above the results; ``'sidebar'``: inputs in a sidebar.
    lwrun, lwreset, lwsetbas:
        Show the buttons Run scenario, Reset to start and Use as baseline.
    varpat, showvarpat:
        Old names of ``selectfrom`` and ``showselectfrom``.
    viewer:
        Options for the result viewer, see :class:`modelwidget_core.ViewerOptions`
        (the names are those of :class:`modelwidget_input.keep_plot_widget`).
    """
    if _is_model(widgetdef) and not _is_model(mmodel):     # updatewidget order: model first
        widgetdef, mmodel = mmodel, widgetdef
    if (mmodel is None) == (model_factory is None):
        raise ValueError('Give either mmodel or model_factory')
    for name in NOTEBOOK_ONLY:
        viewer.pop(name, None)
    opt = viewer_options(varpat=varpat, showvarpat=showvarpat, **viewer)

    if 'ipykernel' not in sys.modules:   # no screen in a web server
        import matplotlib
        matplotlib.use('Agg')

    # built once for the page; every session gets its own copy with the same ids
    template = _core_template(widgetdef)

    buttons = []
    if lwrun:
        buttons.append(ui.input_action_button('mf_run', 'Run scenario', class_='btn-success'))
    if lwreset:
        buttons.append(ui.input_action_button('mf_reset', 'Reset to start'))
    if lwsetbas:
        buttons.append(ui.input_action_button('mf_setbasis', 'Use as baseline'))
    run_row = ui.div(
        ui.input_text('mf_name', 'Scenario name', value='Experiment 1', width='240px'),
        *buttons,
        class_='d-flex gap-2 flex-wrap align-items-end')
    results = [ui.output_text('mf_kept'), ui.output_ui('mf_viewer'), ui.output_ui('mf_figs')]

    if layout == 'sidebar':
        app_ui = ui.page_sidebar(
            ui.sidebar(input_ui(template), run_row, width=sidebar_width),
            ui.card(*results),
            title=title)
    else:
        app_ui = ui.page_fluid(
            ui.h3(title, class_='my-3'),
            ui.card(ui.card_header('Scenario inputs'), input_ui(template), run_row),
            ui.card(ui.card_header('Results'), *results),
            title=title)

    def server(input, output, session):
        model = model_factory() if model_factory is not None else mmodel
        core = deepcopy(template)
        runner = ScenarioRunner(model, basename=basename, keeppat=keeppat)
        runs = reactive.value(0)          # bumped when the kept solutions change

        inputs = InputServer(core, input, output)
        ResultViewer(model, runs, opt, input, output)
        ui.update_text('mf_name', value=runner.next_name)

        @reactive.effect
        @reactive.event(input.mf_run)
        def _run():
            try:
                inputs.read()
                with ui.Progress() as progress:
                    progress.set(message='Running the model')
                    runner.run(core, input.mf_name())
            except Exception as e:      # show it; an unhandled error would end the session
                ui.notification_show(f'The scenario was not run: {e}', type='error', duration=None)
                return
            ui.update_text('mf_name', value=runner.next_name)
            runs.set(runs() + 1)

        @reactive.effect
        @reactive.event(input.mf_reset)
        def _reset():
            try:
                inputs.reset()
            except Exception as e:
                ui.notification_show(f'Reset failed: {e}', type='error', duration=None)

        @reactive.effect
        @reactive.event(input.mf_setbasis)
        def _setbasis():
            try:
                runner.setbasis()
            except Exception as e:
                ui.notification_show(f'Could not use as baseline: {e}', type='error', duration=None)
                return
            runs.set(runs() + 1)

        @render.text
        def mf_kept():
            runs()
            return 'Scenarios kept: ' + ', '.join(model.keep_solutions)

    return App(app_ui, server)


# ---------------------------------------------------------------------------
# Showing an app in a notebook
# ---------------------------------------------------------------------------

_running = {}       # port -> uvicorn server started by show_app


def show_app(app: App, port: int = None, height: int = 900, host: str = '127.0.0.1'):
    """Run a Shiny app in the background and show it below the notebook cell.

    Works in Jupyter and VS Code on a desktop. Not in JupyterLite (no server
    can run in the browser; use the notebook widgets there) and not in Colab.

    Example::

        w = make_widget(tabdef)
        updatewidget(mpak, w, legend=True)              # notebook widgets
        show_app(make_app(w, mpak, legend=True))        # the same as a Shiny app

    Returns the server; :func:`stop_app` (or ``stop_app()`` for all) stops it.
    Calling ``show_app`` again with the same port replaces that app.
    """
    import socket
    import threading
    import time

    import uvicorn
    from IPython.display import IFrame, display

    if sys.platform == 'emscripten':
        raise RuntimeError('show_app needs a server, which can not run in JupyterLite; '
                           'use the notebook widgets (updatewidget) there')

    if port is None:
        with socket.socket() as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
    stop_app(port)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):            # up to 10 seconds
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f'The app did not start on port {port}')
    _running[port] = (server, thread)

    display(IFrame(f'http://{host}:{port}/', width='100%', height=height))
    return server


def stop_app(port: int = None) -> None:
    """Stop an app started by :func:`show_app`; without ``port`` all of them."""
    ports = list(_running) if port is None else [port]
    for p in ports:
        running = _running.pop(p, None)
        if running is not None:
            server, thread = running
            server.should_exit = True
            thread.join(timeout=5)
