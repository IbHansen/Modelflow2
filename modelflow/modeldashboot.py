try:
    import dash_interactive_graphviz
    from jupyter_dash import JupyterDash
    import dash
    from dash.dependencies import Input, Output, State
    import dash_html_components as html
    import dash_core_components as dcc
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
except:
    ...

import webbrowser
from threading import Timer

import json
import networkx as nx


from pathlib import Path
import pandas as pd

from modelclass import model 
from modelhelp import cutout


#%%
if 0:
    smallmodel      = '''
    frml <> a = c + b $ 
    frml <> d1 = x + 3 * a(-1)+ c **2 +a  $ 
    frml <> d3 = x + 3 * a(-1)+c **3 $  
    Frml <> x = 0.5 * c +a$'''
    des = {'A':'Bruttonationalprodukt i faste  priser',
           'X': 'Eksport <æøåÆØÅ>;',
           'C': 'Forbrug'}
    mmodel = model(smallmodel,var_description=des,svg=1,browser=1)
    df = pd.DataFrame({'X' : [0.2,0.2] , 'C' :[0.,0.] , 'R':[1.,0.4] , 'P':[0.,0.4]})
    df2 = pd.DataFrame({'X' : [0.2,0.2] , 'C' :[10.,10.] , 'R':[1.,0.4] , 'P':[0.,0.4]})
    
    xx = mmodel(df)
    yy = mmodel(df2)
    
class Dash_Mixin():
    
    def modeldash(self,pre_var='',selected_data_show ='baseline+last run',debug=True,jupyter=False,show_trigger=False): 
        import dash_interactive_graphviz
        from jupyter_dash import JupyterDash
        import dash
        from dash.dependencies import Input, Output, State
        import dash_html_components as html
        import dash_core_components as dcc
        
        import webbrowser
        from threading import Timer
        
        import json
        
        if jupyter: 
            app = JupyterDash(__name__)
        else:
            app = dash.Dash(__name__)
            
        selected_var = pre_var if pre_var else sorted(self.allvar.keys())[0] 
        
        
        app.layout = html.Div(
            [  
                html.Div([
                    dash_interactive_graphviz.DashInteractiveGraphviz(id="gv",engine='dot',
                          dot_source = self.draw(selected_var,dot=True,up=0,down=1,
                            last=0,all = 1 , HR=False)),],
                    style=dict(flexGrow=1, position="relative")
                               ,
                ), 
                
                 
                html.H3("Variable"),
                
                html.Div(
                    [
                       
                        html.H3("Variable"),
                        dcc.Markdown(
                                id='outvar-state',
                                children=f'{selected_var}'),
                        
                        dcc.Dropdown(id='var',value=selected_var,options=[
                                dict(label=v, value=v)
                                for v in sorted(self.allvar.keys())],
                        )
                         ,
                        html.H3("Up, preceeding levels"),
                        dcc.Dropdown(id="up",value=0,options=[
                                dict(label=engine, value=engine)
                                for engine in list(range(10))],
                        ),
                        html.H3("Down, dependent levels"),
                        dcc.Dropdown(id="down",value=1,options=[
                                dict(label=engine, value=engine)
                                for engine in list(range(10))],
                        ),
                        html.H3("Data"),
                        dcc.Dropdown(id="data_show",value=selected_data_show,options=[
                                dict(label=engine, value=engine)
                                for engine in ['Variable names','baseline+last run','last run']],
                        ),
                        html.H3("Graph orientation"),             
                        dcc.RadioItems(id='orient',
                            options=[
                                {'label': 'Vertical', 'value': False},
                                {'label': 'Horisontal', 'value': True},
                                 ],
                       value=True
                       )  
        
                    ],
                    style=dict(display="flex", flexDirection="column"),
                ),
            ],
            style=dict(position="absolute", height="100%", width="100%", display="flex"),
        )
        
        
        @app.callback(
            [Output("gv", "dot_source"),Output('outvar-state', "children")],
            [
             Input('var', "value"),Input('gv', "selected_node"),Input('up', "value"),
             Input('down', "value"), Input('data_show', "value"),
             Input('orient', "value")],
             State('outvar-state','children')
        )
        def display_output( var,select_var,up,down,data_show,orient,outvar_state):
            # value=self.drawmodel(svg=1,all=True,browser=0,pdf=0,des=True,dot=True)
            ctx = dash.callback_context
            if ctx.triggered:
                if show_trigger:
                    ctx_msg = json.dumps({
                    'states': ctx.states,
                    'triggered': ctx.triggered,
                    'inputs': ctx.inputs
                          }, indent=2)
                    print(ctx_msg)
                    # print(f'{outvar=},{data_show=}')
                    print(orient)
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if trigger in ['var']:
                    try:
                        outvar=var[:]
                    except:
                        return [dash.no_update,dash.no_update]
                    


                elif trigger == 'gv':
                    pass
                    try:
                        xvar= select_var.split('(')[0]
                        if xvar in self.endogene or xvar in self.exogene: 
                            outvar = xvar[:]
                    except:
                       outvar= select_node.split('->')[0]
                else: 
                        outvar=outvar_state
                        
                        
                value=self.draw(outvar,dot=True,up=up,down=down,
                                   last=data_show=='last run',all =data_show=='baseline+last run' ,
                                   HR=orient)
                
                # else:     
                #     value=self.draw(outvar,dot=True,up=int(up),down=int(down),all=False)
                   
                 
            
            return [value,outvar]
        
        def open_browser():
        	webbrowser.open_new(f"http://localhost:{5000}")
        
        Timer(1, open_browser).start()
        app.run_server(debug=debug,port=5000)
          
    
if __name__ == "__main__":
    
    from modelclass import model 

    class xmodel(model,Dash_Mixin):
        
        ... 
        
    if not  'baseline' in locals():    
        mmodel,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0 )
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = mmodel(scenarie)
    setattr(model, "modeldash", Dash_Mixin.modeldash) 
    
    mmodel.modeldash('FY',jupyter=False,show_trigger=True,debug=False) 
#%% 