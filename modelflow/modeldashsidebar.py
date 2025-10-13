# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:46:21 2021

@author: bruger
"""

import dash
# from jupyter_dash import JupyterDash
#from jupyter_dash.comms import _send_jupyter_config_comm_request
try:
    from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL
    import dash_bootstrap_components as dbc
    from  dash_interactive_graphviz import DashInteractiveGraphviz
    # from dash.exceptions import PreventUpdate

    # try:
    #     from dash import dcc 
    # except:
    #     import dash_core_components as dcc
    
    # import dash_bootstrap_components as dbc
    # try:
    #     from dash import html
    # except:     
    #     import dash_html_components as html
        
    # from dash.dependencies import Input, Output, State
    
    # from  dash_interactive_graphviz import DashInteractiveGraphviz
except:
    print('No Dash')    

import plotly.graph_objs as go
from plotly.graph_objs.scatter.marker import Line

import webbrowser
from threading import Timer

import time

import json
import networkx as nx

from dataclasses import dataclass,field


from pathlib import Path
import pandas as pd

from modelhelp import cutout

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


first_call = True



initial_dot_source = """
digraph  {
node[style="filled"]
a ->b->d
a->c->d
}
"""

sidebar_width, rest_width = "15%", "82%"
# the style arguments for the sidebar. We use position:fixed and a fixed width
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

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE_TOP = {
    "background-color": "f8f9fa",
    "width": "100%",
    
    "margin-left": "0%",
}
CONTENT_STYLE_GRAPH = {
    "background-color": "f8f9fa",
    "width":"82%",
    
    "margin-left": '0%',
}
CONTENT_STYLE_TAB = {
    "background-color": "f8f9fa",
    "margin-left": sidebar_width,
    "width": "85%",
    "height": "auto",
}


def app_setup(jupyter=False):  
    # print('apprun')
    if jupyter: 
        # Timer(1,_send_jupyter_config_comm_request).start()
        # JupyterDash.infer_jupyter_proxy_config()
        # app = JupyterDash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
        app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    return app                 

def app_run(app,jupyter=False,debug=False,port=5000,inline=False):
    global first_call
    def open_browser(port=port):
    	webbrowser.open_new(f"http://localhost:{port}")
    
    # print(f'{jupyter=} {inline=}')   
    if jupyter:
        if inline:
            xx = app.run(debug=debug,port=port,mode='inline')
        else:     
            if first_call or  1:
                Timer(1, open_browser).start()   
                first_call = False
            # print('ko')   

            xx =app.run(debug=debug,port=port,jupyter_mode='external')
            # print('gris')   

            # print(f'{xx=}')

    else:    
        Timer(1, open_browser).start()            

        xx = app.run(debug=debug,port=port,mode="external")
        
    


def get_stack(df,v='Guess it',heading='Yes',pct=True,threshold=0.5,desdict = {}):
    pv = cutout(df,threshold)
    template =  '%{y:.1f}% explained by:<br><b>%{meta}</b>:' if pct else '%{y:.2f} explained by:<br><b>%{meta}</b>:' 
    trace = [go.Bar(x=pv.columns.astype('str'), y=pv.loc[rowname,:], name=rowname,hovertemplate = template,
                    meta = desdict.get(rowname,'hest')) for rowname in pv.index]
    out = { 'data': trace,
    'layout':
    go.Layout(title=f'{heading}', barmode='relative',legend_itemclick='toggleothers',
              yaxis = {'tickformat': ',7.0','ticksuffix':'%' if pct else ''})
    }
    return out 

def get_no_stack(df,v='No attribution for exogenous variables ',desdict = {}):
    out = {'layout':
    go.Layout(title=f'{v}')
    }
    return out 

def get_line_old(pv,v='Guess it',heading='Yes'):
    trace = [go.Line(x=pv.columns.astype('str'), y=pv.loc[rowname,:], name=rowname) for rowname in pv.index]
    out = { 'data': trace,
    'layout':
    go.Layout(title=f'{heading}')
    }
    return out 
def get_line(pv,v='Guess it',heading='Yes',pct=True):
    trace = [go.Scatter(x=pv.columns.astype('str'),
                  y=pv.loc[rowname,:], 
                  name=rowname
                  ) for rowname in pv.index]
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


@dataclass 
class Dash_graph():
    
     mmodel : any = None  
     pre_var : str =''
     filter: float  = 0
     up : int = 1
     down : int = 0
     time_att  : bool = False 
     attshow :bool = False
     all : bool = False 
     port : int = 5001
     debug : bool = False
     jupyter : bool = True 
     show_trigger : bool = False 
     inline : bool =False
     lag : bool = False 
     threshold : float  =0.5
     growthshow : bool = False
     
     def __post_init__(self):
        self.fokusvar = set()
        # print('Still worlking on the layout of this')
        self.firstsession = True
        selected_var = self.pre_var if self.pre_var else sorted(self.mmodel.allvar.keys())[0] 
        self.outvar_state =  selected_var
        sidebar = html.Div(
            [ 
              html.H3("Current Variable"),
              dcc.Markdown(
                  id='outvar_state',
                  children=f'{selected_var}'),
              
              html.H3("Select a variable"),             
              dcc.Dropdown(id='var',value=selected_var,options=[
                  dict(label=v, value=v)
                  for v in sorted(self.mmodel.allvar.keys())],persistence = True,persistence_type='session'),
              html.H3("Tree walking"),
              dbc.Row([
                  html.H5("Up"),
                  dcc.Dropdown(id="up",value=self.up,options=[
                      dict(label=engine, value=engine)
                      for engine in list(range(10))],persistence = True,persistence_type='session'),
    
                  html.H5("Down"),
                  dcc.Dropdown(id="down",value=self.down,options=[
                      dict(label=engine, value=engine)
                      for engine in list(range(10))],persistence = True,persistence_type='session'),
              ]),
              dbc.Tooltip('Looking at preceding variabels up to this level',
                          target = 'up'),
              dbc.Tooltip('Looking at dependent variabels down to this level',
                          target = 'down'),

              html.H3("Graph filter%"),
              dcc.Dropdown(id="filter",value=self.filter,options=[
                  dict(label=t, value=t)
                  for t in list(range(0,100,10))],persistence = True,persistence_type='session'),
              dbc.Tooltip('All branches contribution less are pruned from the graph',
                          target = 'filter'),
              
              html.H3("Graph orientation"),             
              dcc.RadioItems(id='orient',
              options=[
                  {'label': 'Vertical', 'value':'v'},
                  
                  {'label': 'Horisontal', 'value': 'h'},
                  ],
              value='v',labelStyle={'display': 'block'},persistence = True,persistence_type='session' ),
              
              html.H3("Node Display"),             
              dcc.RadioItems(id='node',
              options=[
                  {'label': 'Name', 'value':'name'},
                  {'label': '+values', 'value':'all'},
                  
                  {'label': '+Attribution', 'value': 'attshow'},
                  ],persistence = True, persistence_type='session',
            value='all' if self.all else ('attshow' if self.attshow else 'name') ,labelStyle={'display': 'block'}),
              
              html.H3("Behavior when clicking on node"),             
              dcc.RadioItems(id='onclick',
              options=[
                  {'label': 'Center', 'value':'c'},                  
                  {'label': 'Display', 'value': 'd'},
                  ],
            value='c',persistence = True,persistence_type='session'),
                   
             
            ],
            style=SIDEBAR_STYLE
        )
        # breakpoint()
        outvar= selected_var    
        tab0 = html.Div([
            dbc.Tabs(id="tabs", children=[
                dbc.Tab(id='Graph',label='Graph',children= [DashInteractiveGraphviz(id="gv" , style=CONTENT_STYLE_GRAPH, 
                                                                                                        
                        dot_source =   self.mmodel.draw(selected_var,up=self.up,down=self.down,showatt=False,lag=self.lag,
                                                 debug=0,dot=True,HR=False,filter = self.filter,
                                                 all=False,attshow=False))],
                        style=CONTENT_STYLE_TOP , ),

                dbc.Tab(id='Chart',label='Chart', 
                        children = [dcc.Graph(id='chart',
                        figure=get_line(self.mmodel.value_dic[selected_var].iloc[:2,:],selected_var,f'The values for {selected_var}'))
                        , dcc.Graph(id='chart_dif',
                        figure=get_line(self.mmodel.value_dic[selected_var].iloc[[2],:],selected_var,f'The impact for {selected_var}'))
                         ],                            
                        style=CONTENT_STYLE_TOP),

                dbc.Tab(id='Attribution',label='Attribution', 
                        children = [dcc.Graph(id='att_pct',
                        figure = (get_stack(self.mmodel.att_dic[outvar],outvar,f'Attribution of the impact - pct. for {outvar}',
                                            threshold=self.threshold,desdict=self.mmodel.var_description)
                                   if outvar in self.mmodel.endogene else html.H3("Graph orientation")))
                        , dcc.Graph(id='att_level',
                        figure =  (get_stack(self.mmodel.att_dic_level[outvar],outvar,f'Attribution of the impact - level for {outvar}',
                                           pct=False,threshold=self.threshold,desdict=self.mmodel.var_description) if outvar in self.mmodel.endogene else   html.H3("Graph orientation")))
               
                         ],                            
                        style=CONTENT_STYLE_TOP)


                    ],persistence=True, persistence_type = 'session'),
                  ],style = CONTENT_STYLE_TAB)
        # tabbed = tab0
        # tabbed = dbc.Container(tab0,id='tabbed',style={"height": "100vh"},fluid=True)
        tabbed =  html.Div(tab0,id='tabbed',style={"height": "100vh"})
        self.app  = app_setup(jupyter=self.jupyter)
    
        
        
        # app.layout = html.Div([sidebar,body2])
        self.app.layout = dbc.Container([sidebar,tabbed],style={"height": "100vh","width":"100%"},fluid=True)

        @self.app.callback(
            [Output("gv", "dot_source"),
             Output('chart','figure'), Output('chart_dif','figure'), 
             Output('att_pct','figure'), Output('att_level','figure'), 
             Output('outvar_state','children'),
             Output('tabs','active_tab')],
            
            [Input('var', "value"),
                Input('gv', "selected_node"), Input('gv', "selected_edge"),
                  Input('up', "value"),Input('down', "value"),Input('filter', "value"),
                  Input('orient', "value"),
                  Input('onclick','value'),
                  Input('node','value')

               ]
               , State('outvar_state','children')
        )
        def display_output( var,
                            selected_node,selected_edge,
                              up,down,filter,
                               orient,
                               onclick,
                               node,
                             outvar_state
                            ):
            # time.sleep(3)
            ctx = dash.callback_context
            # if ctx.triggered[0]['prop_id']== 'gv.selected_node' and ctx.triggered[0]['value']==  None:
            #    raise PreventUpdate                                                                             
           
            # print(f'new trigger :{ctx.triggered=}')
            # breakpoint()
            if ctx.triggered:

                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if self.show_trigger:
                    print('\nNew Trigger:')
                    ctx_msg = json.dumps({
                    'states': ctx.states,
                    'triggered': ctx.triggered,
                    'inputs': ctx.inputs
                          }, indent=2)
                    print(ctx_msg)
                    # print(f'{outvar=},{data_show=}')
                    print(f'{outvar_state=}')
                    print(f'{self.fokusvar=} When triggerd')

                update_var = True    

                if trigger == 'var':
                    outvar=var
                    

             
                # elif trigger == 'gv' :
                   
                    # print(f'{selected_node=}')
                elif selected_node:
                    xvar= selected_node.split('(')[0]
                    if onclick == 'c':
                        outvar = xvar if trigger == 'gv' else self.outvar_state
                        
                    elif onclick == 'd' :
                         if trigger == 'gv':
                            if xvar in self.fokusvar:
                                self.fokusvar = self.fokusvar-{xvar}
                            else: 
                                self.fokusvar = self.fokusvar | {xvar}
                         outvar=xvar   
                
                else:
                    
                    outvar=self.outvar_state

                self.fokusvar = set() if trigger in {'onclick','var'} else self.fokusvar
                
                if onclick == 'c' or trigger == 'var' : #or outvar not in self.mmodel.value_dic.keys() : # or trigger in ['up','down','orient','filter','node'] :   
                    dot_out =  self.mmodel.draw(outvar,up=up,down=down,filter=filter,showatt=False,debug=0,
                                         lag=self.lag,dot=True,HR=orient=='h',last=0,all = False,
                                         attshow=False)
  
                elif onclick == 'd'  :   
                    # print(f'{outvar=}')
                    if self.show_trigger:
                        print(f'{self.fokusvar=} on display ')

                    dot_out =  self.mmodel.draw(self.outvar_state,up=up,down=down,filter=filter,showatt=False,debug=0,
                                         lag=self.lag,dot=True,HR=orient=='h',fokus2=self.fokusvar,all= node == 'all',
                                         attshow= node =='attshow',growthshow= node=='all' or node =='attshow',)
  
                chart_out = get_line(self.mmodel.get_values(outvar).iloc[:2,:],outvar,f'The values for {outvar}:{self.mmodel.var_description.get(outvar,"")}')
                chart_dif_out = get_line(self.mmodel.get_values(outvar).iloc[[2],:],outvar,f'The impact for {outvar}:{self.mmodel.var_description.get(outvar,"")}')
                
                att_pct_out = (get_stack(self.mmodel.get_att_pct(
                outvar.split('(')[0], lag=False, start='', end=''),outvar,f'Attribution of the impact - pct. for {outvar}:{self.mmodel.var_description.get(outvar,"")}'
                                         ,threshold=self.threshold,desdict=self.mmodel.var_description)
                                   if outvar in self.mmodel.endogene else  html.H3(""))
                     
                att_level_out = (get_stack(self.mmodel.get_att_level(
                outvar.split('(')[0], lag=False, start='', end=''),outvar,f'Attribution of the impact - level for {outvar}',
                                           pct=False,threshold=self.threshold,desdict=self.mmodel.var_description) if outvar in self.mmodel.endogene else  html.H3("") )
                 
            else:
                return [dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update, dash.no_update, dash.no_update]
            if self.show_trigger:
                print(f'{self.fokusvar=} before render')
            self.outvar_state = outvar if onclick == 'c' or trigger == 'var' else self.outvar_state
            outvar_state = self.outvar_state
            return [dot_out, chart_out, chart_dif_out, att_pct_out, att_level_out, outvar_state ,dash.no_update,]
        
        app_run(self.app,jupyter=self.jupyter,debug=self.debug,port=self.port,inline=self.inline)
#%test        
if __name__ == "__main__":

#%% testing
       
    if not  'baseline' in locals() or 0 :    
        from modelclass import model 
        madam,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0 )
        # make a simpel experimet VAT
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = madam(scenarie)
        
    
    _ =  Dash_graph(madam,'FY',debug = 0,all=1,filter=30,show_trigger=True,jupyter=True,up=1,port=5006)
