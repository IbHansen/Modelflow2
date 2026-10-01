# -*- coding: utf-8 -*-
"""
modelinput_shiny
================

Draw ModelFlow input widgets with Shiny for Python (Shiny Core) and make a
standalone scenario web app.

It uses the same widget definitions as :func:`modelwidget_input.make_widget`
and the logic in :mod:`modelwidget_core`, so a definition which works in a
notebook works in the app.

Supported widget types: ``base``, ``tab``, ``slide``, ``radio`` and ``check``.
``sumslide`` and ``sheet`` are not supported yet.

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

    app = make_app(widgetdef, model_factory=load_model, varpat='PAKNYGDPMKTPKN PAKGGREVCO2CER')

Run it with ``shiny run app.py``, or export it for the browser with
``shinylive export <folder> <site>``.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

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
    fig_to_image,
    keep_figs,
    make_core,
)


# ---------------------------------------------------------------------------
# Input widgets: core -> Shiny ui, and the values back
# ---------------------------------------------------------------------------

def input_ui(core) -> Any:
    """The Shiny ui for a tree of core widgets (see :func:`modelwidget_core.make_core`)."""
    match core:
        case TabCore():
            titles = core.titles
            selected = titles[core.selected_index] if titles else None
            if core.tab:
                return ui.navset_tab(
                    *[ui.nav_panel(title, input_ui(child)) for title, child in zip(titles, core.children)],
                    id=core.id, selected=selected)
            return ui.accordion(
                *[ui.accordion_panel(title, input_ui(child)) for title, child in zip(titles, core.children)],
                id=core.id, open=selected)

        case BaseCore():
            return ui.div(*[input_ui(child) for child in core.children])

        case SumSlideCore() | SheetCore():
            raise NotImplementedError(f"{type(core).__name__} ({core.heading!r}) can not be shown in Shiny yet")

        case SlideCore():
            return ui.div(
                ui.h5(core.heading),
                *[ui.input_slider(core.control_id(des), des,
                                  min=line["min"], max=line["max"], value=core.values[des],
                                  step=line.get("step", 0.01))
                  for des, line in core.lines.items()])

        case RadioCore():
            return ui.div(
                ui.h5(core.heading),
                *[ui.input_radio_buttons(core.control_id(des), des,
                                         choices={str(i): label for i, (label, _) in enumerate(cont)},
                                         selected=str(core.selected[des]))
                  for des, cont in core.content.items()])

        case CheckCore():
            return ui.div(
                ui.h5(core.heading),
                *[ui.input_checkbox(core.control_id(des), des, value=core.values[des])
                  for des in core.content])

        case _:
            raise NotImplementedError(f"No Shiny ui for {type(core).__name__}")


def read_inputs(core, input) -> None:
    """Copy the current Shiny input values into the core widgets."""
    match core:
        case ContainerCore():
            for child in core.children:
                read_inputs(child, input)
        case SlideCore():
            for des in core.lines:
                core.set_value(des, input[core.control_id(des)]())
        case RadioCore():
            for des in core.content:
                core.select(des, int(input[core.control_id(des)]()))
        case CheckCore():
            for des in core.content:
                core.set_value(des, bool(input[core.control_id(des)]()))


def show_values(core) -> None:
    """Move the Shiny inputs to the values in the core widgets (after a reset)."""
    match core:
        case ContainerCore():
            for child in core.children:
                show_values(child)
        case SlideCore():
            for des in core.lines:
                ui.update_slider(core.control_id(des), value=core.values[des])
        case RadioCore():
            for des in core.content:
                ui.update_radio_buttons(core.control_id(des), selected=str(core.selected[des]))
        case CheckCore():
            for des in core.content:
                ui.update_checkbox(core.control_id(des), value=core.values[des])


# ---------------------------------------------------------------------------
# The scenario app
# ---------------------------------------------------------------------------

_DIFF = {'no': False, 'yes': True, 'pct': 'pct'}


def make_app(widgetdef, mmodel: Any = None, *, model_factory: Callable[[], Any] = None,
             title: str = 'ModelFlow scenarios', varpat: str = '*', selected: str = '',
             basename: str = 'Business as usual',
             keeppat: str = '*', relativ_start: int = 0, vline: Any = None,
             sidebar_width: int = 420) -> App:
    """A Shiny app: input widgets, run scenarios, and charts of the kept solutions.

    Parameters
    ----------
    widgetdef:
        A widget definition, as for :func:`modelwidget_input.make_widget`.
    mmodel:
        A model with its ``basedf`` set. All browser sessions share it, so use
        this for one user (or Shinylive, where every user has their own Python).
    model_factory:
        A function returning a new model. Called for every browser session, so
        users on a shared server get their own model. Use this or ``mmodel``.
    varpat:
        Pattern of the variables which can be shown in the charts.
    selected:
        Pattern of the variables charted when the app opens (default the first one).
    basename, keeppat:
        Name of the baseline scenario and pattern of the variables to keep, as
        in :class:`modelwidget_input.updatewidget`.
    relativ_start:
        Start the charts this many periods relative to the model's current period.
    """
    if (mmodel is None) == (model_factory is None):
        raise ValueError('Give either mmodel or model_factory')

    if 'ipykernel' not in sys.modules:   # no screen in a web server
        import matplotlib
        matplotlib.use('Agg')

    # built once for the page; every session builds its own core with the same ids
    template = make_core(widgetdef)

    app_ui = ui.page_sidebar(
        ui.sidebar(
            input_ui(template),
            ui.input_text('mf_name', 'Scenario name', value='Experiment 1'),
            ui.div(
                ui.input_action_button('mf_run', 'Run scenario', class_='btn-success'),
                ui.input_action_button('mf_reset', 'Reset to start'),
                ui.input_action_button('mf_setbasis', 'Use as baseline'),
                class_='d-flex gap-2 flex-wrap'),
            width=sidebar_width,
        ),
        ui.card(
            ui.input_selectize('mf_vars', 'Variables', choices=[], multiple=True, width='100%'),
            ui.div(
                ui.input_radio_buttons('mf_diff', 'Difference to first scenario',
                                       {'no': 'No', 'yes': 'Yes', 'pct': 'In percent'}, inline=True),
                ui.input_radio_buttons('mf_showtype', 'Data type',
                                       {'level': 'Level', 'growth': 'Growth'}, inline=True),
                ui.input_radio_buttons('mf_scale', 'Y-scale',
                                       {'linear': 'Linear', 'log': 'Log'}, inline=True),
                class_='d-flex gap-4 flex-wrap'),
            ui.output_text('mf_kept'),
            ui.output_ui('mf_figs'),
        ),
        title=title,
    )

    def server(input, output, session):
        model = model_factory() if model_factory is not None else mmodel
        core = make_core(widgetdef)
        runner = ScenarioRunner(model, basename=basename, keeppat=keeppat)
        runs = reactive.value(0)          # bumped when the kept solutions change

        with model.set_smpl_relative(relativ_start, 0):
            show_per = list(model.current_per)
        smpl = (show_per[0], show_per[-1])

        allkeep = [set(df.columns) for df in model.keep_solutions.values()]
        keepvar = set.intersection(*allkeep) if allkeep else set()
        showvars = [v.upper() for v in model.vlist(varpat) if v in keepvar]

        def label(v):
            des = model.var_description[v]
            return v if des == v else f'{des} ({v})'

        showset = set(showvars)
        first = [v.upper() for v in model.vlist(selected) if v.upper() in showset] if selected else []
        ui.update_selectize('mf_vars', choices={v: label(v) for v in showvars},
                            selected=first or showvars[:1])
        ui.update_text('mf_name', value=runner.next_name)

        @reactive.effect
        @reactive.event(input.mf_run)
        def _run():
            read_inputs(core, input)
            with ui.Progress() as progress:
                progress.set(message='Running the model')
                runner.run(core, input.mf_name())
            ui.update_text('mf_name', value=runner.next_name)
            runs.set(runs() + 1)

        @reactive.effect
        @reactive.event(input.mf_reset)
        def _reset():
            core.reset()
            show_values(core)

        @reactive.effect
        @reactive.event(input.mf_setbasis)
        def _setbasis():
            runner.setbasis()
            runs.set(runs() + 1)

        @render.text
        def mf_kept():
            runs()
            return 'Scenarios: ' + ', '.join(model.keep_solutions)

        @render.ui
        def mf_figs():
            runs()
            variables = input.mf_vars()
            if not variables:
                return ui.p('Select one or more variables')
            figs = keep_figs(model, variables, smpl, diff=_DIFF[input.mf_diff()],
                             showtype=input.mf_showtype(), scale=input.mf_scale(), vline=vline)
            return ui.navset_tab(*[ui.nav_panel(label(v), ui.HTML(fig_to_image(fig)))
                                   for v, fig in figs.items()])

    return App(app_ui, server)
