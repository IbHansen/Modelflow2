# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:46:21 2021

@author: bruger
"""

"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_interactive_graphviz
"""

initial_dot_source = """
digraph  {
node[style="filled"]
a ->b->d
a->c->d
}
"""

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {"background-color": "blue"
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
       dcc.Dropdown(
        id='demo-ib',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC'
    ),
    ],
    style=SIDEBAR_STYLE,
)

graph = dash_interactive_graphviz.DashInteractiveGraphviz(id="gv" ,dot_source = initial_dot_source,style=CONTENT_STYLE)



body = dbc.Container([
    dbc.Row(
            [
                dbc.Col(
                    html.P("This is column 1"),
                    width={'size':12,'offset':1,'order':'last'},
                    style={"height": "100%", "background-color": "red"})

            ],
            className="h-50",justify='start'
        ),
    dbc.Row(dbc.Col(graph,width={'size':12,'offset':1,'order':'last'}),className="h-50",justify='start')]
    
    ,style={"height": "100vh"})


app.layout = html.Div([sidebar,body])




if __name__ == "__main__":
    app.run_server(port=5002)
