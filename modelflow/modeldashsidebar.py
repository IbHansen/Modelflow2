# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:46:21 2021

@author: bruger
"""

import dash
from jupyter_dash import JupyterDash
from jupyter_dash.comms import _send_jupyter_config_comm_request
try:
    from dash import dcc 
except:
    import dash_core_components as dcc

import dash_bootstrap_components as dbc
try:
    from dash import html
except:     
    import dash_html_components as html
from dash.dependencies import Input, Output, State
from  dash_interactive_graphviz import DashInteractiveGraphviz
import plotly.graph_objs as go
from plotly.graph_objs.scatter.marker import Line

import webbrowser
from threading import Timer

import json
import networkx as nx


from pathlib import Path
import pandas as pd

from modelhelp import cutout

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# log.setLevel(logging.INFO)



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
    "width": rest_width,
    
    "margin-left": sidebar_width,
}
CONTENT_STYLE_GRAPH = {
    "background-color": "f8f9fa",
    "width":"70%",
}
CONTENT_STYLE_TAB = {
    "background-color": "f8f9fa",
    "margin-left": sidebar_width,
    "width": "85%",
    "height": "auto",
}


def app_setup(jupyter=False):  
    if jupyter: 
        # Timer(1,_send_jupyter_config_comm_request).start()
        # JupyterDash.infer_jupyter_proxy_config()
        app = JupyterDash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    return app                 

def app_run(app,jupyter=False,debug=False,port=5000,inline=True):
    def open_browser(port=port):
    	webbrowser.open_new(f"http://localhost:{port}")
    
       
    if jupyter:
        if inline:
            app.run_server(debug=debug,port=port,mode='inline')
        else:     
            Timer(1, open_browser).start()            
            app.run_server(debug=debug,port=port,mode='external')

    else:    
        Timer(1, open_browser).start()            

        app.run_server(debug=debug,port=port)


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




class Dash_Mixin():
    
          
    def modeldash(self,pre_var='FY',debug=False,jupyter=False,show_trigger=False,port=5001,lag= False,filter = 0,up=1,down=0,
                  threshold=0.5,inline=False,time_att=False): 
        print('Still worlking on the layout of this')
        self.dashport = port
        selected_var = pre_var if pre_var else sorted(self.allvar.keys())[0] 
        sidebar = html.Div(
            [ 
              html.H3("Current Variable"),
              dcc.Markdown(
                  id='outvar_state',
                  children=f'{selected_var}'),
              
              html.H3("Select a variable"),             
              dcc.Dropdown(id='var',value=selected_var,options=[
                  dict(label=v, value=v)
                  for v in sorted(self.allvar.keys())]),
             
              html.H3("Up Levels"),
              dcc.Dropdown(id="up",value=up,options=[
                  dict(label=engine, value=engine)
                  for engine in list(range(10))]),

              html.H3("Down levels"),
              dcc.Dropdown(id="down",value=down,options=[
                  dict(label=engine, value=engine)
                  for engine in list(range(10))]),
              
              html.H3("Graph filter%"),
              dcc.Dropdown(id="filter",value=filter,options=[
                  dict(label=t, value=t)
                  for t in list(range(0,100,10))]),
             
              html.H3("Graph orientation"),             
              dcc.RadioItems(id='orient',
              options=[
                  {'label': 'Vertical', 'value':'v'},
                  
                  {'label': 'Horisontal', 'value': 'h'},
                  ],
            value='v',labelStyle={'display': 'block'}),
              
              html.H3("Behavior when clicking on graph"),             
              dcc.RadioItems(id='onclick',
              options=[
                  {'label': 'Center when click', 'value':'c'},                  
                  {'label': 'Display when click', 'value': 'd'},
                  ],
            value='c',labelStyle={'display': 'block'}),
                   
             
            ],
            style=SIDEBAR_STYLE
        )
        # top =   dbc.Col([
        outvar= selected_var    
        tab0 = html.Div([
            dbc.Tabs(id="tabs", children=[
                dbc.Tab(id='Graph',label='Graph', children= [DashInteractiveGraphviz(id="gv" , style=CONTENT_STYLE_GRAPH, 
                        dot_source =   self.draw(selected_var,up=up,down=down,showatt=False,lag=lag,
                                                 debug=0,dot=True,HR=False,filter = filter))],
                        style=CONTENT_STYLE_TOP),

                dbc.Tab(id='Chart',label='Chart', 
                        children = [dcc.Graph(id='chart',
                        figure=get_line(self.value_dic[selected_var].iloc[:2,:],selected_var,f'The values for {selected_var}'))
                        , dcc.Graph(id='chart_dif',
                        figure=get_line(self.value_dic[selected_var].iloc[[2],:],selected_var,f'The impact for {selected_var}'))
                         ],                            
                        style=CONTENT_STYLE_TOP),

                dbc.Tab(id='Attribution',label='Attribution', 
                        children = [dcc.Graph(id='att_pct',
                        figure = (get_stack(self.att_dic[outvar],outvar,f'Attribution of the impact - pct. for {outvar}',
                                            threshold=threshold,desdict=self.var_description)
                                   if outvar in self.endogene else html.H3("Graph orientation")))
                        , dcc.Graph(id='att_level',
                        figure =  (get_stack(self.att_dic_level[outvar],outvar,f'Attribution of the impact - level for {outvar}',
                                           pct=False,threshold=threshold,desdict=self.var_description) if outvar in self.endogene else   html.H3("Graph orientation")))
               
                         ],                            
                        style=CONTENT_STYLE_TOP)


                    ]),
                  ],style = CONTENT_STYLE_TAB)
        # tabbed = tab0
        # tabbed = dbc.Container(tab0,id='tabbed',style={"height": "100vh"},fluid=True)
        tabbed =  html.Div(tab0,id='tabbed',style={"height": "100vh"})
        app  = app_setup(jupyter=jupyter)
    
        
        
        # app.layout = html.Div([sidebar,body2])
        app.layout = dbc.Container([sidebar,tabbed],style={"height": "100vh","width":"100%"},fluid=True)

        @app.callback(
            [Output("gv", "dot_source"),
             Output('chart','figure'), Output('chart_dif','figure'), 
             Output('att_pct','figure'), Output('att_level','figure'), 
             Output('outvar_state','children'),
             Output('tabs','active_tab')],
            
            [Input('var', "value"),
                Input('gv', "selected_node"),Input('gv', "selected_edge"),
                  Input('up', "value"),Input('down', "value"),Input('filter', "value"),
                  Input('orient', "value"),
                  Input('onclick','value')

               ]
               , State('outvar_state','children')
        )
        def display_output( var,
                            selected_node,selected_edge,
                              up,down,filter,
                               orient,
                               onclick,
                             outvar_state
                            ):
            ctx = dash.callback_context
            if ctx.triggered:
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if show_trigger:
                    ctx_msg = json.dumps({
                    'states': ctx.states,
                    'triggered': ctx.triggered,
                    'inputs': ctx.inputs
                          }, indent=2)
                    print(ctx_msg)
                    # print(f'{outvar=},{data_show=}')
                    # print(value)
                update_var = True    
                tab_out = dash.no_update

                if trigger in ['var']:
                    try:
                        outvar=var[:]
                    except:
                        return [dash.no_update, dash.no_update,dash.no_update, dash.no_update,dash.no_update, dash.no_update,dash.no_update]
                    

              
                elif trigger == 'gv':
                    pass
                    try:
                        xvar= selected_node.split('(')[0]
                        if xvar in self.endogene or xvar in self.exogene: 
                            outvar = xvar[:]
                    except:
                        outvar = selected_edge.split('->')[0]
                else:
                    ...
                    
                    outvar=outvar_state
                      
                if onclick == 'c' or outvar not in self.value_dic.keys()  or trigger in ['up','down','orient','filter'] :   
                    dot_out =  self.draw(outvar,up=up,down=down,filter=filter,showatt=False,debug=0,
                                         lag=lag,dot=True,HR=orient=='h')
                    tab_out = dash.no_update
  
                elif onclick == 'd' and trigger in ['gv'] :   
                    dot_out =  self.draw(outvar_state,up=up,down=down,filter=filter,showatt=False,debug=0,
                                         lag=lag,dot=True,HR=orient=='h',fokus2=outvar)
                    update_var  = False 
                    tab_out = 'Attribution'
                else:
                    dot_out = dash.no_update
                    tab_out =dash.no_update
  
                chart_out = get_line(self.value_dic[outvar].iloc[:2,:],outvar,f'The values for {outvar}:{self.var_description.get(outvar,"")}')
                chart_dif_out = get_line(self.value_dic[outvar].iloc[[2],:],outvar,f'The impact for {outvar}:{self.var_description.get(outvar,"")}')
                
                att_pct_out = (get_stack(self.att_dic[outvar],outvar,f'Attribution of the impact - pct. for {outvar}:{self.var_description.get(outvar,"")}'
                                         ,threshold=threshold,desdict=self.var_description)
                                   if outvar in self.endogene else  html.H3(""))
                     
                att_level_out = (get_stack(self.att_dic_level[outvar],outvar,f'Attribution of the impact - level for {outvar}',
                                           pct=False,threshold=threshold,desdict=self.var_description) if outvar in self.endogene else  html.H3("") )
                 
            else:
                return [dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update, dash.no_update, dash.no_update]

            outvar_state = outvar if update_var else outvar_state     
            return [dot_out, chart_out, chart_dif_out, att_pct_out, att_level_out, outvar_state ,dash.no_update,]
       
        app_run(app,jupyter=jupyter,debug=debug,port=self.dashport,inline=inline)
#%test        
if __name__ == "__main__":

        
    if not  'baseline' in locals():    
        from modelclass import model 
        madam,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0 )
        # make a simpel experimet VAT
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = madam(scenarie)
        
   
    setattr(model, "modeldash", Dash_Mixin.modeldash)    
    madam.modeldash('FY',jupyter=False,show_trigger=True,debug=False) 
    #mmodel.FY.draw(up=1, down=1,svg=1,browser=1)

