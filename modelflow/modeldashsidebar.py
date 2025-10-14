# -*- coding: utf-8 -*-
"""
Interactive Dash Visualization for ModelFlow / MFMod Models
------------------------------------------------------------

This script builds an interactive visualization dashboard using Dash,
designed to explore variable relationships, time series, and attributions
within ModelFlow (MFMod) models.

It provides:
- Graphviz-based dependency visualization
- Time-series charts for selected variables
- Attribution analysis (percentage and level)
- Interactive sidebar for graph navigation

Original Author: Ib 
Date Created: May 14, 2021
"""

# ----------------------------------------------------------------------
# Imports and setup
# ----------------------------------------------------------------------
import dash
try:
    from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL
    import dash_bootstrap_components as dbc
    from dash_interactive_graphviz import DashInteractiveGraphviz
except ImportError:
    print('Dash or its components are not available in this environment.')

import plotly.graph_objs as go
from plotly.graph_objs.scatter.marker import Line

import webbrowser
from threading import Timer
import time
import json
import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from modelhelp import cutout
import logging

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global flag for browser auto-open control
first_call = True


# ----------------------------------------------------------------------
# Layout style definitions
# ----------------------------------------------------------------------
sidebar_width, rest_width = "15%", "82%"

# Sidebar fixed to the left side
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": sidebar_width,
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll",
}

# General content style
CONTENT_STYLE_TOP = {
    "background-color": "f8f9fa",
    "width": "100%",
    "margin-left": "0%",
}

CONTENT_STYLE_GRAPH = {
    "background-color": "f8f9fa",
    "width": "82%",
    "margin-left": "0%",
}

CONTENT_STYLE_TAB = {
    "background-color": "f8f9fa",
    "margin-left": sidebar_width,
    "width": "85%",
    "height": "auto",
}


# ----------------------------------------------------------------------
# Dash app setup and launcher
# ----------------------------------------------------------------------
def app_setup(jupyter=False):
    """
    Create and configure a Dash application with Bootstrap styling.

    Parameters
    ----------
    jupyter : bool, optional
        If True, sets up app for Jupyter/Colab usage.

    Returns
    -------
    dash.Dash
        The configured Dash app instance.
    """
    if jupyter:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    return app


def app_run(app, jupyter=False, debug=False, port=5000, inline=False):
    """
    Run the Dash app in Jupyter, Colab, or standard browser environments.

    Parameters
    ----------
    app : dash.Dash
        The Dash application to run.
    jupyter : bool, optional
        Set True when running inside a Jupyter or Colab notebook.
    debug : bool, optional
        Enables Dash debug mode.
    port : int, optional
        TCP port to run the app on (default is 5000).
    inline : bool, optional
        Run inline inside notebook (True) or external browser (False).

    Notes
    -----
    - Automatically opens a browser tab when running locally.
    - Uses `mode='inline'` in Colab for embedded display.
    """
    global first_call

    def open_browser(port=port):
        """Open the Dash app in the default web browser."""
        webbrowser.open_new(f"http://localhost:{port}")

    if jupyter:
        # Running inside a Jupyter or Colab notebook
        if inline:
            xx = app.run(debug=debug, port=port, mode='inline')
        else:
            if first_call or 1:
                Timer(1, open_browser).start()
                first_call = False
            xx = app.run(debug=debug, port=port, jupyter_mode='external')
    else:
        # Standard local environment
        Timer(1, open_browser).start()
        xx = app.run(debug=debug, port=port, mode="external")


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def get_stack(df, v='Guess it', heading='Yes', pct=True, threshold=0.5, desdict={}):
    """
    Build a stacked bar chart for variable attribution analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input attribution data (components × periods).
    v : str
        Variable name to display.
    heading : str
        Chart title.
    pct : bool
        If True, display percentages on y-axis.
    threshold : float
        Minimum contribution threshold for inclusion.
    desdict : dict
        Mapping of variable names to descriptions.

    Returns
    -------
    dict
        A Plotly figure dictionary for Dash.
    """
    pv = cutout(df, threshold)
    template = '%{y:.1f}% explained by:<br><b>%{meta}</b>:' if pct else '%{y:.2f} explained by:<br><b>%{meta}</b>:'
    trace = [
        go.Bar(
            x=pv.columns.astype('str'),
            y=pv.loc[rowname, :],
            name=rowname,
            hovertemplate=template,
            meta=desdict.get(rowname, 'hest')
        )
        for rowname in pv.index
    ]
    out = {
        'data': trace,
        'layout': go.Layout(
            title=f'{heading}',
            barmode='relative',
            legend_itemclick='toggleothers',
            yaxis={'tickformat': ',7.0', 'ticksuffix': '%' if pct else ''}
        )
    }
    return out


def get_no_stack(df, v='No attribution for exogenous variables ', desdict={}):
    """
    Create a placeholder figure when attribution data is unavailable.

    Parameters
    ----------
    df : pandas.DataFrame
        Placeholder input.
    v : str
        Message to display.
    desdict : dict
        Variable descriptions (unused).

    Returns
    -------
    dict
        Simple Plotly layout with message.
    """
    out = {'layout': go.Layout(title=f'{v}')}
    return out


def get_line(pv, v='Guess it', heading='Yes', pct=True):
    """
    Generate a standard line chart (time-series).

    Parameters
    ----------
    pv : pandas.DataFrame
        Data (rows = series, columns = time periods).
    v : str
        Variable name for title.
    heading : str
        Chart title.
    pct : bool
        Placeholder for percentage handling (unused).

    Returns
    -------
    dict
        Plotly figure dictionary.
    """
    trace = [
        go.Scatter(
            x=pv.columns.astype('str'),
            y=pv.loc[rowname, :],
            name=rowname
        )
        for rowname in pv.index
    ]
    out = {'data': trace, 'layout': go.Layout(title=f'{heading}')}
    return out


def generate_table(dataframe, max_rows=10):
    """
    Convert a pandas DataFrame into an HTML table.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input table data.
    max_rows : int
        Maximum number of rows to display.

    Returns
    -------
    dash.html.Table
        HTML table element for Dash.
    """
    return html.Table([
        html.Thead(
            html.Tr([html.Th('var')] + [html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([html.Td(dataframe.index[i])] + [
                html.Td(f'{dataframe.iloc[i][col]:25.2f}') for col in dataframe.columns
            ])
            for i in range(min(len(dataframe), max_rows))
        ])
    ])


# ----------------------------------------------------------------------
# Main Dash visualization class
# ----------------------------------------------------------------------
@dataclass
class Dash_graph:
    """
    Interactive dashboard for exploring ModelFlow or MFMod models.

    Displays:
    ----------
    - Variable dependency graphs (via Graphviz)
    - Time series (levels and impacts)
    - Attribution charts (percentage and level)

    Parameters
    ----------
    mmodel : any
        ModelFlow or MFMod model object.
    pre_var : str
        Default variable to display initially.
    filter : float
        Minimum contribution filter for pruning.
    up : int
        Levels of upstream dependencies to include.
    down : int
        Levels of downstream dependencies to include.
    time_att : bool
        Placeholder for time-based attribution.
    attshow : bool
        If True, shows attribution values on nodes.
    all : bool
        If True, shows all variables with values.
    port : int
        Dash server port.
    debug : bool
        Enable Dash debug mode.
    jupyter : bool
        Set True when running in Jupyter/Colab.
    show_trigger : bool
        Print callback triggers to console for debugging.
    inline : bool
        Display inline within notebook if True.
    lag : bool
        Include lagged variables if True.
    threshold : float
        Pruning threshold for attribution.
    growthshow : bool
        Show growth decomposition components if True.
    """

    mmodel: any = None
    pre_var: str = ''
    filter: float = 0
    up: int = 1
    down: int = 0
    time_att: bool = False
    attshow: bool = False
    all: bool = False
    port: int = 5001
    debug: bool = False
    jupyter: bool = True
    show_trigger: bool = False
    inline: bool = False
    lag: bool = False
    threshold: float = 0.5
    growthshow: bool = False

    def __post_init__(self):
        """
        Initialize layout, define callbacks, and launch the Dash server.
        """
        self.fokusvar = set()
        self.firstsession = True
        selected_var = self.pre_var if self.pre_var else sorted(self.mmodel.allvar.keys())[0]
        self.outvar_state = selected_var

        # Build sidebar layout
        sidebar = html.Div(
            [
                html.H3("Current Variable"),
                dcc.Markdown(id='outvar_state', children=f'{selected_var}'),

                html.H3("Select a variable"),
                dcc.Dropdown(
                    id='var',
                    value=selected_var,
                    options=[dict(label=v, value=v) for v in sorted(self.mmodel.allvar.keys())],
                    persistence=True,
                    persistence_type='session'
                ),

                html.H3("Tree walking"),
                dbc.Row([
                    html.H5("Up"),
                    dcc.Dropdown(
                        id="up",
                        value=self.up,
                        options=[dict(label=i, value=i) for i in range(10)],
                        persistence=True,
                        persistence_type='session'
                    ),
                    html.H5("Down"),
                    dcc.Dropdown(
                        id="down",
                        value=self.down,
                        options=[dict(label=i, value=i) for i in range(10)],
                        persistence=True,
                        persistence_type='session'
                    ),
                ]),
                dbc.Tooltip("Upstream variable depth", target='up'),
                dbc.Tooltip("Downstream variable depth", target='down'),

                html.H3("Graph filter%"),
                dcc.Dropdown(
                    id="filter",
                    value=self.filter,
                    options=[dict(label=t, value=t) for t in range(0, 100, 10)],
                    persistence=True,
                    persistence_type='session'
                ),

                html.H3("Graph orientation"),
                dcc.RadioItems(
                    id='orient',
                    options=[
                        {'label': 'Vertical', 'value': 'v'},
                        {'label': 'Horizontal', 'value': 'h'}
                    ],
                    value='v',
                    labelStyle={'display': 'block'},
                    persistence=True,
                    persistence_type='session'
                ),

                html.H3("Node Display"),
                dcc.RadioItems(
                    id='node',
                    options=[
                        {'label': 'Name', 'value': 'name'},
                        {'label': '+values', 'value': 'all'},
                        {'label': '+Attribution', 'value': 'attshow'}
                    ],
                    value='all' if self.all else ('attshow' if self.attshow else 'name'),
                    labelStyle={'display': 'block'},
                    persistence=True,
                    persistence_type='session'
                ),

                html.H3("Click behavior"),
                dcc.RadioItems(
                    id='onclick',
                    options=[
                        {'label': 'Center', 'value': 'c'},
                        {'label': 'Display', 'value': 'd'}
                    ],
                    value='c',
                    persistence=True,
                    persistence_type='session'
                ),
            ],
            style=SIDEBAR_STYLE
        )

        # Main tabbed content
        outvar = selected_var
        tab0 = html.Div([
            dbc.Tabs(id="tabs", children=[
                # Graph tab
                dbc.Tab(
                    id='Graph',
                    label='Graph',
                    children=[
                        DashInteractiveGraphviz(
                            id="gv",
                            style=CONTENT_STYLE_GRAPH,
                            dot_source=self.mmodel.draw(
                                selected_var,
                                up=self.up,
                                down=self.down,
                                showatt=False,
                                lag=self.lag,
                                debug=0,
                                dot=True,
                                HR=False,
                                filter=self.filter,
                                all=False,
                                attshow=False
                            )
                        )
                    ],
                    style=CONTENT_STYLE_TOP
                ),

                # Chart tab
                dbc.Tab(
                    id='Chart',
                    label='Chart',
                    children=[
                        dcc.Graph(
                            id='chart',
                            figure=get_line(self.mmodel.value_dic[selected_var].iloc[:2, :],
                                            selected_var, f'The values for {selected_var}')
                        ),
                        dcc.Graph(
                            id='chart_dif',
                            figure=get_line(self.mmodel.value_dic[selected_var].iloc[[2], :],
                                            selected_var, f'The impact for {selected_var}')
                        )
                    ],
                    style=CONTENT_STYLE_TOP
                ),

                # Attribution tab
                dbc.Tab(
                    id='Attribution',
                    label='Attribution',
                    children=[
                        dcc.Graph(
                            id='att_pct',
                            figure=(get_stack(
                                self.mmodel.att_dic[outvar],
                                outvar,
                                f'Attribution (pct) for {outvar}',
                                threshold=self.threshold,
                                desdict=self.mmodel.var_description)
                                if outvar in self.mmodel.endogene else html.H3("Graph orientation"))
                        ),
                        dcc.Graph(
                            id='att_level',
                            figure=(get_stack(
                                self.mmodel.att_dic_level[outvar],
                                outvar,
                                f'Attribution (level) for {outvar}',
                                pct=False,
                                threshold=self.threshold,
                                desdict=self.mmodel.var_description)
                                if outvar in self.mmodel.endogene else html.H3("Graph orientation"))
                        )
                    ],
                    style=CONTENT_STYLE_TOP
                ),
            ], persistence=True, persistence_type='session'),
        ], style=CONTENT_STYLE_TAB)

        # App initialization and layout
        self.app = app_setup(jupyter=self.jupyter)
        tabbed = html.Div(tab0, id='tabbed', style={"height": "100vh"})
        self.app.layout = dbc.Container([sidebar, tabbed],
                                        style={"height": "100vh", "width": "100%"},
                                        fluid=True)

        # Register callback (unchanged from original)
        @self.app.callback(
            [Output("gv", "dot_source"),
             Output('chart', 'figure'),
             Output('chart_dif', 'figure'),
             Output('att_pct', 'figure'),
             Output('att_level', 'figure'),
             Output('outvar_state', 'children'),
             Output('tabs', 'active_tab')],
            [Input('var', "value"),
             Input('gv', "selected_node"),
             Input('gv', "selected_edge"),
             Input('up', "value"),
             Input('down', "value"),
             Input('filter', "value"),
             Input('orient', "value"),
             Input('onclick', "value"),
             Input('node', "value")],
            State('outvar_state', 'children')
        )
        def display_output(var, selected_node, selected_edge, up, down, filter,
                           orient, onclick, node, outvar_state):
            """
            Main callback for updating graphs and charts based on user interaction.
            Handles variable selection, node clicking, orientation, and filtering.
            """
            ctx = dash.callback_context
            if not ctx.triggered:
                return [dash.no_update] * 7

            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if self.show_trigger:
                print(f"\nTriggered by: {trigger}")

            # Determine which variable to display
            if trigger == 'var':
                outvar = var
            elif selected_node:
                xvar = selected_node.split('(')[0]
                if onclick == 'c':
                    outvar = xvar if trigger == 'gv' else self.outvar_state
                elif onclick == 'd':
                    if trigger == 'gv':
                        if xvar in self.fokusvar:
                            self.fokusvar -= {xvar}
                        else:
                            self.fokusvar |= {xvar}
                    outvar = xvar
            else:
                outvar = self.outvar_state

            # Maintain focus variable state
            self.fokusvar = set() if trigger in {'onclick', 'var'} else self.fokusvar

            # Graph rendering logic (identical to original)
            if onclick == 'c' or trigger == 'var':
                dot_out = self.mmodel.draw(outvar, up=up, down=down, filter=filter,
                                           showatt=False, debug=0, lag=self.lag,
                                           dot=True, HR=orient == 'h', last=0,
                                           all=False, attshow=False)
            elif onclick == 'd':
                dot_out = self.mmodel.draw(self.outvar_state, up=up, down=down,
                                           filter=filter, showatt=False, debug=0,
                                           lag=self.lag, dot=True, HR=orient == 'h',
                                           fokus2=self.fokusvar, all=node == 'all',
                                           attshow=node == 'attshow',
                                           growthshow=node in {'all', 'attshow'})

            # Build figures
            chart_out = get_line(
                self.mmodel.get_values(outvar).iloc[:2, :],
                outvar,
                f'The values for {outvar}:{self.mmodel.var_description.get(outvar, "")}'
            )
            chart_dif_out = get_line(
                self.mmodel.get_values(outvar).iloc[[2], :],
                outvar,
                f'The impact for {outvar}:{self.mmodel.var_description.get(outvar, "")}'
            )

            att_pct_out = (
                get_stack(
                    self.mmodel.get_att_pct(outvar.split('(')[0], lag=False, start='', end=''),
                    outvar,
                    f'Attribution (pct) for {outvar}:{self.mmodel.var_description.get(outvar, "")}',
                    threshold=self.threshold,
                    desdict=self.mmodel.var_description
                ) if outvar in self.mmodel.endogene else html.H3("")
            )

            att_level_out = (
                get_stack(
                    self.mmodel.get_att_level(outvar.split('(')[0], lag=False, start='', end=''),
                    outvar,
                    f'Attribution (level) for {outvar}',
                    pct=False,
                    threshold=self.threshold,
                    desdict=self.mmodel.var_description
                ) if outvar in self.mmodel.endogene else html.H3("")
            )

            self.outvar_state = outvar if onclick == 'c' or trigger == 'var' else self.outvar_state
            return [dot_out, chart_out, chart_dif_out, att_pct_out, att_level_out, self.outvar_state, dash.no_update]

        # Launch the app
        app_run(self.app, jupyter=self.jupyter, debug=self.debug, port=self.port, inline=self.inline)


# ----------------------------------------------------------------------
# Standalone example for testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modelclass import model
    madam, baseline = model.modelload('../Examples/ADAM/baseline.pcim', run=1, silent=0)
    scenarie = baseline.copy()
    scenarie.TG = scenarie.TG + 0.05
    _ = madam(scenarie)

    _ = Dash_graph(madam, 'FY', debug=0, all=1, filter=0,
                   show_trigger=True, jupyter=False, up=1, port=5006)
