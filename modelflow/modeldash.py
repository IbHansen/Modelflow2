
from pathlib import Path
import pandas as pd

from modelclass import model 



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
    
def modeldash(mmodel,debug=True): 
    import dash_interactive_graphviz
    import dash
    from dash.dependencies import Input, Output
    import dash_html_components as html
    import dash_core_components as dcc
    
    import webbrowser
    from threading import Timer
    
    import json
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                dash_interactive_graphviz.DashInteractiveGraphviz(id="gv",engine='dot'),
                style=dict(flexGrow=1, position="relative"),
            ),
            html.Div(
                [
                   
                    html.H3("var"),
                    dcc.Dropdown(id='var',value=sorted(mmodel.allvar.keys())[0],options=[
                            dict(label=v, value=v)
                            for v in sorted(mmodel.allvar.keys())],
                    )
                     ,
                    html.H3("Up, preceeding levels"),
                    dcc.Dropdown(id="up",value=1,options=[
                            dict(label=engine, value=engine)
                            for engine in list(range(10))],
                    ),
                    html.H3("Down, dependent levels"),
                    dcc.Dropdown(id="down",value=1,options=[
                            dict(label=engine, value=engine)
                            for engine in list(range(10))],
                    ),
                    html.H3("Data"),
                    dcc.Dropdown(id="data_show",value='baseline+last run',options=[
                            dict(label=engine, value=engine)
                            for engine in ['Variable names','baseline+last run','last run']],
                    ),
                    html.H3("Graph orientation"),             
                    dcc.RadioItems(id='orient',
                        options=[
                            {'label': 'Vertical', 'value': False},
                            {'label': 'Horisontal', 'value': True},
                             ],
        value=False
    )  
    
                ],
                style=dict(display="flex", flexDirection="column"),
            ),
        ],
        style=dict(position="absolute", height="100%", width="100%", display="flex"),
    )
    
    
    @app.callback(
        [Output("gv", "dot_source")], #,Output('var', "value")],
        [
         Input('var', "value"),Input('gv', "selected_node"),Input('up', "value"),
         Input('down', "value"), Input('data_show', "value"),
         Input('orient', "value")],
    )
    def display_output( var,select_var,up,down,data_show,orient):
        # value=mmodel.drawmodel(svg=1,all=True,browser=0,pdf=0,des=True,dot=True)
        ctx = dash.callback_context
        try:
            outvar=var[:]
        except:
            return dash.no_update
        if ctx.triggered:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger in ['var','down','up','data_show']:
                pass
                # print('kddddddddddddddkkddd')
            elif trigger == 'gv':
                pass
                xvar= select_var.split('(')[0]
                if xvar in mmodel.endogene or xvar in mmodel.exogene: 
                    outvar = xvar[:]
               
                    
            # this is not elegant but works 
            if 0:         
                if data_show=='name':
                    value=mmodel.draw(outvar,dot=True,up=int(up),down=int(down),all=False,HR=orient)
                elif data_show == 'baseline+last run':   
                    value=mmodel.draw(outvar,dot=True,up=int(up),down=int(down),all=True,HR=orient)
                else:   
                    value=mmodel.draw(outvar,dot=True,up=int(up),down=int(down),last=True,HR=orient)
                    
            value=mmodel.draw(outvar,dot=True,up=int(up),down=int(down),
                              last=data_show=='last run',all =data_show=='baseline+last run' ,
                              HR=orient)
            # else:     
            #     value=mmodel.draw(outvar,dot=True,up=int(up),down=int(down),all=False)
    
            ctx_msg = json.dumps({
            'states': ctx.states,
            'triggered': ctx.triggered,
            'inputs': ctx.inputs
                  }, indent=2)
            # print(ctx_msg)
            # print(f'{outvar=},{data_show=}')
            # print(value)
            
             
        
        return [value] # ,outvar
    
    def open_browser():
    	webbrowser.open_new(f"http://localhost:{5000}")
    
    Timer(1, open_browser).start()
    app.run_server(debug=False,port=5000)
    
if not 'baseline' in locals():
    mmodel,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0 )

modeldash(mmodel)    
