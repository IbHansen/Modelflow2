# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:46:21 2021

@author: bruger
"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from  dash_interactive_graphviz import DashInteractiveGraphviz

import webbrowser
from threading import Timer

import json
import networkx as nx


from pathlib import Path
import pandas as pd

from modelclass import model 
from modelhelp import cutout





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


class Dash_Mixin():
    def modeldas(self,pre_var=''
                         debug=True,jupyter=False,show_trigger=False,port=5001): 
        
        def get_stack(df,v='Guess it',heading='Yes'):
            pv = cutout(df,5. )
            trace = [go.Bar(x=pv.columns, y=pv.loc[rowname,:], name=rowname,hovertemplate = '%{y:10.0f}%',) for rowname in pv.index]
            out = { 'data': trace,
            'layout':
            go.Layout(title=f'{heading}', barmode='relative',legend_itemclick='toggleothers',
                      yaxis = {'tickformat': ',.0','ticksuffix':'%'})
            }
            return out 
        
        def get_line(pv,v='Guess it',heading='Yes'):
            trace = [go.Line(x=pv.columns, y=pv.loc[rowname,:], name=rowname) for rowname in pv.index]
            out = { 'data': trace,
            'layout':
            go.Layout(title=f'{heading}')
            }
            return out 

    
        def generate_table(dataframe, max_rows=10):
            return html.Table([
                html.Thead(
                    html.Tr([html.Th('var')]+[html.Th(col) for col in dataframe.columns])
                ),
                html.Tbody([
                    html.Tr([html.Td(dataframe.index[i])]+[html.Td(f'{dataframe.iloc[i][col]:25.2f}') for col in dataframe.columns]) 
                    for i in range(min(len(dataframe), max_rows))
                ])
            ])
              
          
        if jupyter: 
            app = JupyterDash(__name__)
        else:
            app = dash.Dash(__name__)
        selected_var = pre_var if pre_var else sorted(self.allvar.keys())[0] 
    
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
        
        graph = dbc.Col( DashInteractiveGraphviz(id="gv" ,dot_source = initial_dot_source,style=CONTENT_STYLE)
                         ,width={'size':12,'offset':1,'order':'last'})
        top =   dbc.Col(html.P("This is column 1"),
                            width={'size':12,'offset':1,'order':'last'},
                            style={"height": "100%", "background-color": "red"})
        twopanel = [
            dbc.Row(top ,className="h-50",justify='start' ),
            dbc.Row(graph,className="h-50",justify='start')]
        
        onepanel = [
            dbc.Row(graph,className="h-100",justify='start')]
        
        body2 = dbc.Container(twopanel,id='body',style={"height": "100vh"})
        body1 = dbc.Container(onepanel,id='body',style={"height": "100vh"})
        
        
        app.layout = html.Div([sidebar,body1])

        def open_browser(port=port):
        	webbrowser.open_new(f"http://localhost:{port}")
        
        
        
        
        
        
        
        
        
        
        if jupyter:
            Timer(1, open_browser).start()            
            app.run_server(debug=debug,port=port,mode='external')
        else:     
            Timer(1, open_browser).start()
            app.run_server(debug=debug,port=port)



if __name__ == "__main__":
    from modelclass import model 

    class xmodel(model,Dash_Mixin):
        
        ... 
        
    if not  'baseline' in locals():    
        mmodel,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0 )
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = mmodel(scenarie)
    setattr(model, "modeldashexplain", Dash_Mixin.modeldashexplain)    
    mmodel.modeldashexplain('FY',jupyter=False,show_trigger=True,debug=False) 
