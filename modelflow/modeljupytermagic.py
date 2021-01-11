#!/usr/bin/env python
# coding: utf-8


from IPython.display import display, Math, Latex, Markdown , Image, SVG, display_svg,IFrame    
from IPython.lib.latextools import latex_to_png
from io import StringIO
import pandas as pd
from IPython.core.magic import register_line_magic, register_cell_magic
from subprocess import run 


from model_latex import latextotxt
from modelclass import model
from modelmanipulation import explode


def get_options(line,defaultname = 'test'):
    if line: 
        arglistfull = line.split()
        name= arglistfull[0]
        arglist=arglistfull[1:] if len(arglistfull)>= 1 else []
    else:
        name = defaultname
        arglist=[]

    opt = {o.split('=')[0]: o.split('=')[1] if len(o.split('='))==2 else True for o in arglist}
    
    return name,opt           


@register_cell_magic
def graphviz(line, cell):
    '''Creates a ModelFlow model from a Latex document'''
    name,options = get_options(line,'Testgraph')    
    gv = cell
    model.display_graph(gv,name,**options)
    return

# In[29]:


from IPython.core.magic import register_line_magic, register_cell_magic
@register_cell_magic
def latexflow(line, cell):
    '''Creates a ModelFlow model from a Latex document'''
    name,options = get_options(line)
        
    lmodel = cell
    
    
    display(Markdown(f'# Now creating the model m{name}'))
    display(Markdown(f'''    These variables are created: 
    
     - l{name} - Original latex model
     - f{name} - Template Business logic specification
     - m{name} - Model instance 
    '''))
    
    fmodel = latextotxt(cell)
    mmodel  = model.from_eq(fmodel)
    
    globals()[f'l{name}'] = cell
    globals()[f'm{name}'] = mmodel
    globals()[f'f{name}'] = fmodel
    ia = get_ipython()
    ia.push(f'l{name}',interactive=True)
    ia.push(f'm{name}',interactive=True)
    ia.push(f'f{name}',interactive=True)
    
    display(Markdown('## The model'))
    display(Markdown(cell))
    if options.get('display',False):
        display(Markdown(f'# Creating this Template model'))
        print(fmodel)
        display(Markdown(f'# And this Business Logic Language  model'))
        print(mmodel.equations)

    return


def ibmelt(df,prefix='',per=3):
        temp= df.reset_index().rename(columns={'index':'row'}).melt(id_vars='row',var_name='column')        .assign(var_name=lambda x: prefix+x.row+'_'+x.column)        .loc[:,['value','var_name']].set_index('var_name')
        newdf = pd.concat([temp]*per,axis=1).T
        newdf.index = range(per)
        newdf.index.name = 'Year'
        return newdf

@register_cell_magic
def dataframe(line, cell):
    '''Creates a ModelFlow model from a Latex document'''
    name,options = get_options(line,'Testgraph')    
        
    trans = options.get('t',False )    
    
    prefix = options.get('prefix','') 
    periods = int(options.get('periods','1'))
    
    xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
    xcell= cell.replace('%','')
    mul = 0.01 if '%' in cell else 1.
    sio = StringIO(xcell)
    
    df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python').pipe(xtrans)            .pipe(lambda xx:xx.rename(index = {i:i.upper() for i in xx.index},columns={c:c.upper() for c in xx.columns}))            *mul

    df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
    globals()[f'{name}'] = df
    globals()[f'{name}_melted'] = df_melted
    ia = get_ipython()
    ia.push(f'{name}',interactive=True)
    ia.push(f'{name}_melted',interactive=True)
    display(Markdown(f'## Created the dataframes: {name} and {name}_melted'))
    if options.get('show',False):         
        display(df)
        display(df_melted)
    return 