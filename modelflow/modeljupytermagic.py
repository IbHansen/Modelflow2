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
    
    
    display(Markdown(f'# Now creating the model **{name}**'))
    
    fmodel = latextotxt(cell)
    mmodel  = model.from_eq(fmodel)
    mmodel.equations_latex = cell

    
    globals()[f'{name}'] = mmodel
    
    ia = get_ipython()
    ia.push(f'{name}',interactive=True)
    
    display(Markdown('## The model'))
    display(Markdown(cell))
    if options.get('display',False):
        display(Markdown('# Creating this Template model'))
        print(mmodel.equations_original)
        display(Markdown('# And this Business Logic Language  model'))
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
    '''Converts this cell to a dataframe. and create a melted dataframe
   
    options can be added to the inpput line
    - t transposes the dataframe
    - periods=<number> repeats the melted datframe 
    - prefix=<a string> prefix  columns in the melted dataframe
    - show will show the resulting dataframes''' 
    
    name,options = get_options(line,'Testgraph')    
        
    trans = options.get('t',False )    
    
    prefix = options.get('prefix','') 
    periods = int(options.get('periods','1'))
    melt = options.get('melt',False)
    ia = get_ipython()
    
    xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
    xcell= cell.replace('%','')
    mul = 0.01 if '%' in cell else 1.
    sio = StringIO(xcell)
    
    df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python')\
        .pipe(xtrans)\
        .pipe(lambda xx:xx.rename(index = {i:i.upper() if type(i) == str else i for i in xx.index}
                                  ,columns={c:c.upper() for c in xx.columns}))            *mul
    globals()[f'{name}'] = df
    ia.push(f'{name}',interactive=True)
    if melt:
        df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
        globals()[f'{name}_melted'] = df_melted
        ia.push(f'{name}_melted',interactive=True)
    display(Markdown(f'## Created the dataframes: {name}'))
    if melt: display(Markdown(f'## and {name}_melted'))
    if options.get('show',False):         
        display(df)
        if melt: display(df_melted)
    return 