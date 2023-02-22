#!/usr/bin/env python
# coding: utf-8
'''
This module defines several magic jupyter functions\:

:graphviz: Draw Graphviz graph
:dataframe: Create Pandas Dataframe    
:latexflow: Create a modelflow modelinstance from latex script 

he doc strings can only be displayed when the module is used from Jupyter. So they are not shown in the Sphinx documentation To display the doc strings use the functions in jupyter. 

''' 

from IPython.display import display, Math, Latex, Markdown , Image, SVG, display_svg,IFrame    
from IPython.lib.latextools import latex_to_png
from io import StringIO
import pandas as pd
from IPython.core.magic import register_line_magic, register_cell_magic
from subprocess import run 


from model_latex import latextotxt
from modelclass import model
from modelmanipulation import explode
from model_latex_class import a_latex_model


def get_options(line,defaultname = 'test'):
    '''
    Retrives options from the first line


    Args:
        line (TYPE): DESCRIPTION.
        defaultname (TYPE, optional): DESCRIPTION. Defaults to 'test'.

    Returns:
        name (string): name .
        opt (dict): options.

    '''
    if line: 
        arglistfull = line.split()
        name= arglistfull[0]
        arglist=arglistfull[1:] if len(arglistfull)>= 1 else []
    else:
        name = defaultname
        arglist=[]

    opt = {o.split('=')[0]: o.split('=')[1] if len(o.split('='))==2 else True for o in arglist}
    
    return name,opt           
try:
    # this is to 

    @register_cell_magic
    def graphviz(line, cell):
        '''Creates a ModelFlow model from a Latex document'''
        name,options = get_options(line,'Testgraph')    
        gv = cell
        # print(options)
        model.display_graph(gv,name,**options)
        return

# In[29]:


    @register_cell_magic
    def latexflow(line, cell):
        '''Creates a ModelFlow model from a Latex document'''
        name,options = get_options(line)
            
        lmodel = cell
        
        
        # display(Markdown(f'## Now creating the model **{name}**'))
        
        fmodel = latextotxt(cell)
        if options.get('debug',False):
            display(Markdown('## Creating this Template model'))
            print(fmodel)
        mmodel  = model.from_eq(fmodel)
        mmodel.equations_latex = cell
    
        
        globals()[f'{name}'] = mmodel
        
        ia = get_ipython()
        ia.push(f'{name}',interactive=True)
        if options.get('render',True):
            display(Markdown(cell))
        
        # display(Markdown('## The model'))
        if options.get('display',False):
            display(Markdown(cell))
            display(Markdown('## Creating this Template model'))
            print(mmodel.equations_original)
            display(Markdown('## And this Business Logic Language  model'))
            print(mmodel.equations)
    
        return

    @register_cell_magic
    def latexmodelgrab(line, cell):
       '''Creates a ModelFlow model from a Latex document'''
       name,options = get_options(line)
       ia = get_ipython()

       # print(f'{options=}')
       if options.get('segment',False):
          
           if f'{name}_dict' not in globals():
              globals()[f'{name}_dict'] = {}
              ia.push(f'{name}_dict',interactive=True)

           modelsegment= options.get('segment','rest')
           globals()[f'{name}_dict'][modelsegment]=cell   
           
           if modelsegment.startswith('list'):
               display(Markdown(cell))
               return  
           
           # we want this cell plus all list cells to make this small model
           if options.get('all',False):
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
    
           else:     
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items() 
                                  if name.startswith('list')])+cell 
       else:
           if f'{name}_dict' in globals():
               temp='\n'
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
           else: 
               model_text = cell 
       
       latex_model = a_latex_model(model_text)
       mmodel  = latex_model.mmodel
       mmodel.equations_latex = model_text
   
       
       globals()[f'{name}'] = mmodel
       globals()[f'{name}_latex_model_instance'] = latex_model
       
       ia.push(f'{name}',interactive=True)
       ia.push(f'{name}_latex_model_instance',interactive=True)
       
       if options.get('render',True):
           display(Markdown(cell))
       
       # display(Markdown('## The model'))
       if options.get('display',False):
           display(Markdown(cell))
           try:
               print(f'Model:{name} is created from these segments:\n'+
                     f"{temp.join([s for s in globals()[f'{name}_dict'].keys()])} \n")
           except:
               ...
           display(Markdown('## Creating this Template model'))
           print(mmodel.equations_original)
           display(Markdown('## And this Business Logic Language  model'))
           print(mmodel.equations)
   
       return



    def ibmelt(df,prefix='',per=3):
        # breakpoint()
        temp= df.reset_index().rename(columns={'index':'row'}).melt(id_vars='row',var_name='column').assign(var_name=lambda x: prefix+x.row+'_'+x.column)        .loc[:,['value','var_name']].set_index('var_name')
        newdf = pd.concat([temp]*per,axis=1).T
        newdf.index = range(per)
        newdf.index.name = 'Year'
        return newdf
    
    # @register_cell_magic
    # def dataframe(line, cell):
    #     '''Converts this cell to a dataframe. and create a melted dataframe
       
    #     options can be added to the inpput line\:
        
    #     - t transposes the dataframe
    #     - periods=<number> repeats the melted datframe
    #     - repeat = <number> repeats the original  dataframe - for parameters 
    #     - melt will create a melted dataframe 
    #     - prefix=<a string> prefix  columns in the melted dataframe
    #     - show will show the resulting dataframes
    #     - start=index   will set the index (default 2021) 
    #     ''' 
        
    #     name,options = get_options(line,'Testgraph')    
            
    #     trans = options.get('t',False )    
        
    #     if (options.get('help',False )):
    #         print(__doc__)
            
       
    #     prefix = options.get('prefix','') 
    #     periods = int(options.get('periods','1'))
    #     melt = options.get('melt',False)
    #     start = int(options.get('start','2021'))
    #     silent =  options.get('silent',True)
    #     ia = get_ipython()
        
    #     xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
    #     xcell= cell.replace('%','').replace(',','.')
    #     mul = 0.01 if '%' in cell else 1.
    #     sio = StringIO(xcell)
        
    #     df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python')\
    #         .pipe(xtrans)\
    #         .pipe(lambda xx:xx.rename(index = {i:i.upper() if type(i) == str else i for i in xx.index}
    #                                   ,columns={c:c.upper() for c in xx.columns}))            *mul
    #     # breakpoint()    
    #     if not melt: 
    #         df= pd.concat([df]*periods,axis=0)            
    #         df.index = pd.period_range(start=start,freq = 'Y',periods=len(df))
    #         df.index.name = 'index'
            
        
    #     globals()[f'{name}'] = df
    #     ia.push(f'{name}',interactive=True)
    #     if melt:
    #         df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
    #         df_melted.index = pd.period_range(start=start,freq = 'Y',periods=len(df_melted))
    #         df_melted.index.name = 'index'
    
    #         globals()[f'{name}_melted'] = df_melted
    #         ia.push(f'{name}_melted',interactive=True)
    #     if not silent:
    #         melttext = f' and {name}_melted' if melt else ''
    #         display(Markdown(f'## Created the dataframes: {name}{melttext}'))
    #     if options.get('show',False):         
    #         display(df)
    #         if melt: display(df_melted)
    #     return 
    
    @register_cell_magic
    def modeleviews(line, cell):
        '''Creates a ModelFlow model from a Latex document
        
        In developement 
        '''
        
        try: 
            import pyeviews as evp
        except: 
            print('no pyeviews')
            raise Exception('No pyeviews ')
        from  pathlib import Path, PureWindowsPath
     
        name,options = get_options(line)
                
        
        display(Markdown(f'# Now creating the model **{name}**'))
        
        commands = cell.split('\n')
        print(commands)
        if options.get('debug',False):
            display(Markdown('# executing these eviews commands'))
            print(cell)
     
        eviewsapp = evp.GetEViewsApp(instance='new',showwindow=True)
        for c in commands: 
              print(c)
              evp.Run(c,eviewsapp)
    
    
        evp.Cleanup()

    
        return
    
    
    def ibmelt(df,prefix='',per=3):
        # breakpoint()
        temp= df.reset_index().rename(columns={'index':'row'}).melt(id_vars='row',var_name='column').assign(var_name=lambda x: prefix+x.row+'_'+x.column)        .loc[:,['value','var_name']].set_index('var_name')
        newdf = pd.concat([temp]*per,axis=1).T
        newdf.index = range(per)
        newdf.index.name = 'Year'
        return newdf
    
    @register_cell_magic
    def dataframe(line, cell):
        '''Converts this cell to a dataframe. and create a melted dataframe
        
        Works for yearly data. 
       
        Options can be added to the input line
        - t transposes the dataframe
        - periods=<number> repeats the melted datframe
        - melt will create a melted dataframe 
        - prefix=<a string> prefix  columns in the melted dataframe
        - show will show the resulting dataframes
        - start will set the start index (default to 2021) ''' 
        
        name,options = get_options(line,'Testgraph')    
            
        trans = options.get('t',False )    
        
        prefix = options.get('prefix','') 
        periods = int(options.get('periods','1'))
        melt = options.get('melt',False)
        start = int(options.get('start','2021'))
        silent =  options.get('silent',True)
        ia = get_ipython()
        
        xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
        xcell= cell.replace('%','').replace(',','.')
        mul = 0.01 if '%' in cell else 1.
        sio = StringIO(xcell)
        
        df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python')\
            .pipe(xtrans)\
            .pipe(lambda xx:xx.rename(index = {i:i.upper() if type(i) == str else i for i in xx.index}
                                      ,columns={c:c.upper() for c in xx.columns}))            *mul
        # breakpoint()    
        if not melt: 
            df= pd.concat([df]*periods,axis=0)            
            df.index = pd.period_range(start=start,freq = 'Y',periods=len(df))
            df.index.name = 'index'
            
        
        globals()[f'{name}'] = df
        ia.push(f'{name}',interactive=True)
        if melt:
            df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
            df_melted.index = pd.period_range(start=start,freq = 'Y',periods=len(df_melted))
            df_melted.index.name = 'index'
    
            globals()[f'{name}_melted'] = df_melted
            ia.push(f'{name}_melted',interactive=True)
        if not silent:
            melttext = f' and {name}_melted' if melt else ''
            display(Markdown(f'## Created the dataframes: {name}{melttext}'))
        if options.get('show',False):         
            display(df)
            if melt: display(df_melted)
        return 
except:
    print('no magic')