# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:07:02 2017

@author: hanseni

This module creates functions and classes for visualizing results.

 
"""


import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import fnmatch 
from matplotlib import dates
import matplotlib.ticker as ticker
from IPython.display import display
from dataclasses import dataclass, field
from typing import Any, List, Dict

import numpy 

from modelhelp import cutout,finddec
        
##%%
def meltdim(df,dims=['dima','dimb'],source='Latest'):
    ''' Melts a wide dataframe the variable names are split to dimensions acording 
    to the list of texts in dims. in variablenames 
    the tall dataframe have a variable name for each dimensions 
    also values and source are introduced ac column names in the dataframe ''' 
    splitstring = (r'\|').join(['(?P<'+d+'>[A-Z0-9_]*)' for d in dims])
    melted = pd.melt(df.reset_index().rename(columns={'index':'quarter'}),id_vars='quarter')
    vardf = (melted
      .assign(source=source)      
      .assign(varname = lambda df_ :df_.variable.str.replace('__','|',len(dims)-1))   # just to make the next step more easy 
      .pipe(lambda df_ : pd.concat([df_ ,df_.varname.str.extract(splitstring,expand=True)],axis=1)))
    return vardf


    
class DummyVis:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        # Ignore special methods for Jupyter's display system and other internal methods
        ignored_methods = [
            '_ipython_canary_method_should_not_exist_',
            '_repr_',  # covers all _repr_*_ methods
            '__',     # covers all special __*__ methods
        ]
        if any(name.startswith(prefix) for prefix in ignored_methods):
            return lambda *args, **kwargs: None

        # Return a callable that prints a message and returns another DummyVis instance
        def dummy_method(*args, **kwargs):
            # print(f"Attempt to call '{name}' on an uninitialized vis instance.")
            return DummyVis()

        return dummy_method

    def __call__(self, *args, **kwargs):
        # Define behavior when the instance is called as a function
        # print("DummyVis instance called as a function.")
        return DummyVis

    def __repr__(self):
        return "<Try again>"


class vis():
     ''' Visualization class. used as a method on a model instance. 
        
        The purpose is to select variables acording to a pattern, potential with wildcards
     '''
     def __init__(self, model=None, pat='',names=None,df=None):
         self.model = model
         self.__pat__ = pat
         if type(names) == type(None):
             self.names = self.model.vlist(self.__pat__)
         else: 
             self.names = names 
         
         if not len(self.names):
             raise ValueError(f'The variable specification:"{pat}" did not generate any matches')

             

             
         if isinstance(df,pd.DataFrame):
             self.thisdf = df 
         else:
             self.thisdf = self.model.lastdf.loc[:,self.names] 
         return


     def explain(self,**kwargs):
         for var in self.names:
             x = self.model.explain(var,**kwargs)
         return 
     
     def draw(self,**kwargs):
         for var in self.names:
             x = self.model.draw(var,**kwargs)
         
     def dekomp(self,**kwargs):
         self.model.dekomp.cache_clear()
         for var in self.names:
             x = self.model.dekomp(var,**kwargs)
    
     def heat(self,*args, **kwargs):
         ''' Displays a heatmap of the resulting dataframe'''
         
         name = kwargs.pop('title',self.__pat__)
         a = heatshow(self.thisdf.loc[self.model.current_per,:].T,
                      name=name,*args, **kwargs)
         display(a)
         return a 
 
     def plot(self,*args, **kwargs):
         ''' Displays a plot for each of the columns in the resulting dataframe '''
         
         name = kwargs.get('title','Title')
         a = plotshow(self.thisdf.loc[self.model.current_per,:],
                      name=name,*args,**kwargs)
         display(a)
         return a 
     
        
     def plot_alt(self,title='Title',*args, **kwargs):
         ''' Displays a plot for each of the columns in the resulting dataframe '''
         
         if hasattr(self.model,'var_description'):
                 vtrans = self.model.var_description
         else:
                 vtrans = {}
         a = vis_alt(self.model.basedf.loc[self.model.current_per,self.names].rename(columns=vtrans) ,
                     self.model.lastdf.loc[self.model.current_per,self.names].rename(columns=vtrans) ,
                      title=title,*args,**kwargs)
         return a 
     
     def box(self):
             ''' Displays a boxplot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).box()
     def violin(self):
             ''' Displays a violinplot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).violin()
     def swarm(self):
             ''' Displays a swarmlot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).swarm()
     
     
     @property     
     def df(self):
         ''' Returns the result of this instance as a dataframe'''
         return self.thisdf.loc[self.model.current_per,:]

     @property
     def base(self):
         ''' Returns basedf '''
         return vis(model=self.model,df=self.model.basedf.loc[:,self.names],pat=self.__pat__)

     @property
     def pct(self):
         '''Returns the pct change'''
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(),pat=self.__pat__)
     @property
     def growth(self):
         '''Returns the pct growth'''
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(),pat=self.__pat__)

     @property
     def year_pct(self):
         '''Returns the pct change over 4 periods (used for quarterly data) ''' 
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(periods=4).loc[:,self.names],pat=self.__pat__)
     
     @property
     def yoy_growth(self):
         '''Returns the pct change over 4 periods (used for quarterly data) ''' 
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(periods=4).loc[:,self.names],pat=self.__pat__)

     @property
     def qoq_annualized_growth(self):
         '''Returns the pct change over 4 periods (used for quarterly data) ''' 
         df = (1.+self.thisdf.loc[:,self.names].pct_change().loc[:,self.names])**4-1.
         return vis(model=self.model,df=df,pat=self.__pat__)


     @property
     def frml(self):
         '''Returns formulas '''
         def getfrml(var,l):
             if var in self.model.endogene:
                 t=self.model.allvar[var]['frml'].replace('\n',' ').replace('  ',' ')
                 return f'{var:<{l}} : {t}'
             else: 
                 return f'{var:{l}} : Exogenous'
         mlength = max([len(v) for v in self.names]) 
         out = '\n'.join(getfrml(var,mlength) for var in self.names)
         print(out)
         
     @property
     def des(self):
         '''Returns variable descriptions '''
         def getdes(var,l):
             return f'{var:<{l}} : {self.model.var_description[var]}'
         mlength = max([len(v) for v in self.names]) 

         out = '\n'.join(getdes(var,mlength) for var in self.names)
         print(out)

     @property
     def eviews(self):
         '''Returns variable descriptions '''
         def geteviews(var,l):
             if var in self.model.endogene: 
                 ev = self.model.eviews_dict.get(var,'Not avaible')
                 return f'{var:<{l}} : {ev}'
             else: 
                return f'{var:<{l}} : Exogen'
                 
         mlength = max([len(v) for v in self.names]) 
         out = '\n'.join(geteviews(var,mlength) for var in self.names)
         print(out)
          

     @property
     def dif(self):
         ''' Returns the differens between the basedf and lastdf'''
         difdf = self.thisdf-self.model.basedf.loc[:,self.names]
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     
     @property
     def difpctlevel(self):
         ''' Returns the differens between the basedf and lastdf in percent '''
         difdf = (self.thisdf-self.model.basedf.loc[:,self.names])/ self.model.basedf.loc[:,self.names]
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     
     @property
     def difpct(self):
         ''' Returns the differens between the pct changes in basedf and lastdf'''
         difdf = self.thisdf.pct_change()-self.model.basedf.loc[:,self.names].pct_change()
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     @property
     def difgrowth(self):
         ''' Returns the differens between the pct changes in basedf and lastdf'''
         difdf = self.thisdf.pct_change()-self.model.basedf.loc[:,self.names].pct_change()
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     
     @property 
     def endo(self):
         '''Selects only endogenous variables'''
         endonames = [v for v in self.names if v in self.model.endogene]
         thisdf = self.thisdf.loc[:,endonames] 

         return vis(model=self.model,df = thisdf, names= endonames, pat= self.__pat__ )
     @property 
     def exo(self):
         '''Selects only exogenous variables''' 
         endonames = [v for v in self.names if v in self.model.exogene]
         thisdf = self.thisdf.loc[:,endonames] 

         return vis(model=self.model,df = thisdf, names= endonames, pat= self.__pat__ )
     
     @property
     def print(self):
         ''' prints the current result'''
         print('\n',self.thisdf.loc[self.model.current_per,:].to_string())
         return 
     
     def __repr__(self):

#         return self.thisdf.loc[self.model.current_per,:].to_string() 

         if self.model.in_notebook():
           #  from modelwidget import visshow
            # visshow(self.model,self.__pat__)
             return ''
         else:     
             return self.thisdf.loc[self.model.current_per,:].to_string() 
             
        

     def _repr_html_(self):
         '''Displays a nice summary of the results when called in a Jupyter enviorement'''

         from modelwidget import visshow
         pat = ' '.join(self.names)
         # visshow(self.model,self.__pat__)
         visshow(self.model,pat)
         return ''

         # return self.model.ibsstyle(self.thisdf.loc[self.model.current_per,:]).to_html(doctype_html=True) 


     # @property
     # def show(self):
     #     if self.model.in_notebook():
     #         display(self.model.ibsstyle(self.thisdf.loc[self.model.current_per,:],transpose=True))
     #     else:     
     #         print(self.thisdf.loc[self.model.current_per,:].to_string()) 
     @property
     def show(self):
         if self.model.in_notebook():
             pat = ' '.join(self.names)

             from modelwidget import visshow
             visshow(self.model,pat)
         else:     
             print(self.thisdf.loc[self.model.current_per,:].to_string()) 

        
     def __mul__(self,other):
         ''' Multiply the curent result with other '''
         muldf = self.thisdf * other 
         return vis(model=self.model,df=muldf,pat=self.__pat__)
     
     def rename(self,other=None):
         ''' rename columns '''
         if type(other) == type(None):
             if hasattr(self.model,'var_description'):
                 vtrans = self.model.var_description
             else:
                 vtrans = {}
         else:
             vtrans = other
         muldf = self.thisdf.rename(columns=vtrans) 
         return vis(model=self.model,df=muldf,pat=self.__pat__)
     
     @property    
     def endo(self):
         ''' only endogennous variables.  columns '''
         endovar = [v for v in self.names if v in self.model.endogene]
         muldf = self.thisdf.loc[:,endovar] 
         # print(f'{endovar=}')
         return vis(model=self.model,df=muldf,pat = ' '.join( endovar ))


     def mul(self,other):
         ''' Multiply the curent result with other '''
         return self.__mul__(other)
     
     @property
     def mul100(self):
         '''Multiply the current result with 100 '''
         return self.__mul__(100.0) 
         
class compvis() :
     ''' Class to compare to runs in boxplots''' 
     def __init__(self, model=None, pat=None):
         ''' Combines basedf and lastdf to one tall dataframe useful for the Seaborn library''' 
         self.model    = model
         self.__pat__  = pat
         self.names    = self.model.vlist(self.__pat__)
         self.lastdf   = self.model.lastdf.loc[self.model.current_per,self.names] 
         self.basedf   = self.model.basedf.loc[self.model.current_per,self.names]
         self.lastmelt = melt(self.lastdf,source='Scenario') 
         self.basemelt = melt(self.basedf ,source='Base') 
         self.melted   = pd.concat([self.lastmelt,self.basemelt])
         return
     
     def box(self,*args, **kwargs): 
        '''Displays a boxplot'''
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.boxplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
     def swarm(self,*args, **kwargs): 
        '''Displays a swarmplot '''
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.swarmplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
     def violin(self,*args, **kwargs):
        '''Displays a violinplot''' 
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.violinplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
        
   
class  container():
     '''A container, used if to izualize dataframes without a model'''
      
     def __init__(self,lastdf,basedf):
        self.lastdf = lastdf
        self.basedf = basedf
        
     def smpl(self,start='',end='',df=None):
        ''' Defines the model.current_per which is used for calculation period/index
        when no parameters are issues the current current period is returned \n
        Either none or all parameters have to be provided '''
        if start =='' and end == '':
            pass
        else:
            istart,iend= self.lastdf.index.slice_locs(start,end)
            per=self.lastdf.index[istart:iend]
            self.current_per =  per 
        return self.current_per
     def vlist(self,pat):
        '''returns a list of variable matching the pattern'''
        if isinstance(pat,list):
               ipat=pat
        else:
               ipat = [pat]
        out = [v for  p in ipat for v in sorted(fnmatch.filter(self.lastdf.columns,p.upper()))]  
        return out    


##%%  
class varvis():
     ''' Visualization class. used as a method on a model instance. 
        
        The purpose is to select variables acording to a pattern, potential with wildcards
     '''
     def __init__(self, model=None, var=''):
         # print(f' varvis called {var=}')
         self.model = model
         self.var = var
         if var not in model.allvar:
            raise ValueError(f'The specification:"{var}" did not match a method, property or variable name')
         self.endo = model.allvar[var]['endo']
     def explain(self,**kwargs):
         x = self.model.explain(self.var,**kwargs)
         return x
     
     def draw(self,**kwargs):
         x = self.model.draw(self.var,**kwargs)

     def tracedep(self,down=1,**kwargs):
       '''Trace dependensies of name down to level down'''
       self.model.draw(self.var,down=down,up=0,source=self.var,**kwargs)
       
     def tracepre(self,up=1,**kwargs):
       '''Trace dependensies of name down to level down
 - `showdata|sd=True` will include a table of values for each variable 
 - `showdata|sd='pattern of variable names'` will include a table of values for each variable matching the pattern (including wildcharts
 - `attshow|ats = True` will include a table of attributions for each variable
 - `growthshow|gs = True` will include a table of growth for each variable
 - `HR = True` will reorient the dependency graph 
 - `up = <integer>` will determine how many levels of parents to include
 - `pgn = True` will display as a png picture
 - `svg = True` will display as a svg picture which can be zoomed
 - `pdf = True` will display as a pdf picture
 - `eps = True` will create a eps file
 - `browser = True` will open a browser with the resulting dependency graph - useful for zooming on a big graph 
 - `saveas = <a file name without extension>` will save the picture wit the filename with an added extension reflection the picture type 


To allow the use of the display in presentations or publications The resulting file(s) are placed in the graph/subfolder 
       '''
       self.model.draw(self.var,down=0,up=up,sink=self.var,**kwargs)

     @property
     def dash(self):
       '''Trace dependensies of name down to level down'''
       self.model.modeldash(self.var)

     @property
     def dash2(self):
       '''Trace dependensies of name down to level down poer = 5002'''
       self.model.modeldash(self.var,dashport=5002)

     def dashport(self,port=5002):
       '''Trace dependensies of name down to level down'''
       self.model.modeldash(self.var,dashport=port)


     def get_att(self,start='',end='',dec=None,bare=True,**kwargs):
         '''
Retrieve and display the attribution for a variable within a specified period.

Parameters:
    start (str): Start date of the period (default: '').
    end (str): End date of the period (default: '').
    dec (int): Number of decimal places for formatting (default: None).
    bare (bool): If True, display only the attribution result; if False, display both the difference and attribution results (default: True).
    type (str) : One of 'pct', 'growth', 'level' (default: 'pct') 
    **kwargs: Additional keyword arguments for specifying attribution type and other options.

Returns:
    None (displays the attribution result)

Note:
    - The method retrieves the difference and attribution results using the specified period and attribution type.
    - The `dec` parameter controls the number of decimal places for formatting. If not provided, the default number of decimal places is determined based on the attribution type.
    - The `bare` parameter determines whether to display only the attribution result or both the difference and attribution results.
    - The `type` parameter wether to display the level, pct or growth attribution determines whether to display level, growth or pct attribution
'''

         diff = self.model.get_att_diff(self.var,start=start,end=end,**kwargs)
         res = self.model.get_att(self.var,start=start,end=end,**kwargs)
         percent=(kwargs.get('type','pct') in {'growth'}) or \
               ((kwargs.get('type','pct') in {'pct'}) and  bare) 
         if type(dec) == type(None):
             xdec =  2 if (kwargs.get('type','pct') in {'level','growth'}) else 0
         else: 
             xdec = dec 
         if bare: 
             out = res
             out.index.name= 'Growth percent' if kwargs.get('type','pct') == 'growth' else 'Percent att.' if kwargs.get('type','pct') == 'pct' else 'Level att.'
  
         else:     
             out = pd.concat([diff,res])
             out.index.name= 'Growth percent' if kwargs.get('type','pct') == 'growth' else 'Level/percent' if kwargs.get('type','pct') == 'pct' else 'Level/level'
             
         sout = self.model.ibsstyle(out,percent=percent,dec=xdec )   
         display(sout)
             
     
     def dekomp(self,**kwargs):
         if kwargs.get('lprint','False'):
             self.model.dekomp.cache_clear()
         x = self.model.dekomp(self.var,**kwargs)
         return x
     
     def var_des(self,var):
         des = self.model.var_description[var]
         return des if des != var else ''
         
     def _showall_old(self,all=1,dif=0,last=0,show_all=True):
            if self.endo:
                des_string = self.model.get_eq_des(self.var,show_all)
                out1,out2 = '',''
                out0   = f'Endogeneous: {self.var}: {self.var_des(self.var)} \nFormular: {self.model.allvar[self.var]["frml"]}\n\n{des_string}\n'
                try:
                    if dif:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                                            
                    if all:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                        out1 = f'\nInput last run: \n {self.model.get_eq_values(self.var)}\n\nInput base run: \n {self.model.get_eq_values(self.var,last=False)}\n'
                    elif last:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                        out1 = f'\nInput last run: \n {self.model.get_eq_values(self.var)}\n'
                    if all or dif:            
                        out2 = f'\nDifference for input variables: \n {self.model.get_eq_dif(self.var,filter=False)}'
                except Exception as e:
                    print(e)
                    pass 
                out=out0+out1+out2
            else: 
                out   = f'Exogeneous : {self.var}: {self.var_des(self.var)}  \n Values : \n{self.model.get_values(self.var)}\n' 
            return out
 
     # def ibsstyle(self,df,dec=2):
     #     ''' display a dataframe with tooltip'''
         
     #     tt = pd.DataFrame([[self.model.var_description[v] for c in df.columns] for v in df.index ],index=df.index,columns=df.columns) 
     #     xdec = f'{dec}'
     #     result = df.style.format('{:.'+xdec+'f}').\
     #     set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
     #                     'background-color: white; color: #000066; font-size: 0.8em;'
     #                     'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
     #     return result
 
     def _showall(self,all=1,dif=0,last=0,show_all=True):
            from IPython.display import SVG, display, Image, IFrame, HTML, Markdown
            if self.endo:
                des_string = self.model.get_eq_des(self.var,show_all)
                out0,out1,out2 = '','',''
                print(f'Endogeneous: {self.var}: {self.var_des(self.var)}')
                print(f'Formular: {self.model.allvar[self.var]["frml"]}\n\n{des_string}\n')
                try:                                            
                    if dif or all or last: print('Values :')
                    if dif or all or last: display(HTML(self.model.ibsstyle(self.model.get_values(self.var)).to_html() )) 
                    if        all or last: print('Input last run:')
                    if        all or last: display(self.model.ibsstyle(self.model.get_eq_values(self.var)))
                    if        all        : print('Input base run:')
                    if        all        : display(self.model.ibsstyle(self.model.get_eq_values(self.var,last=False)))
                    if        all or dif: print('Difference for input variables')
                    if        all or dif: display(self.model.ibsstyle(self.model.get_eq_dif(self.var,filter=False)))
                except Exception as e:
                    print(e)
                    pass 
                out=out0+out1+out2
            else: 
                out   = f'Exogeneous : {self.var}: {self.var_des(self.var)}  \n Values : \n{self.model.get_values(self.var)}\n' 
            return out
     
    
     @property
     def show(self):
         out = self._showall(all=1)
         print(out)
         return 
     
     @property
     def showdif(self):
         out = self._showall(all=0,dif=1)
         print(out)
         return 
     @property
     def frml(self):
         out = self._showall(all=0,dif=0)
         print(out)
         return 
     @property
     def eviews(self):
         out = self.model.eviews_dict.get(self.var,'Not avaiable')
         print(out)
         return 

     
     def __repr__(self):
             
            out = self._showall(all=0,last=1)
            return out
            

@dataclass
class tabledef:
    mmodel: Any = None
    options: Dict = field(default_factory=dict)
    lines: List = field(default_factory=list)
    dflast: pd.DataFrame = None  # New field with default value None
    dfbase: pd.DataFrame = None  # New field with default value None
    timeslice : List = field(default_factory=list)
    
    
    def __post_init__(self):
        self.timeslice = self.timeslice if self.timeslice else self.mmodel.current_per
        self.outdfs = [self.makedf(self.dflast,line) for line in self.lines ] 
        self.outdf     = pd.concat( self.outdfs ) 
        self.displaydf = pd.concat( self.outdfs ) 
        
        
        
        # self.outstyled = [self.makestyle(linedf,line) for linedf,line in zip(self.outdfs,self.lines)] 
            
        # self.datawidget = self.make_one_style(self.displaydf.loc[:,self.timeslice],self.lines )
       
        return 
    
    @property
    def datawidget(self): 
        return self.make_one_style(self.displaydf.loc[:,self.timeslice],self.lines )
        
    def makedf(self, data, line):   
        linetype = line.get('type', 'growth')
    
        # Pre-process for cases that use linevars and linedes
        if linetype in ['growth', 'gdppct']:
            linevars = self.mmodel.vlist(line['pat'])
            linedes = [self.mmodel.var_description[v] for v in linevars]
    
        match linetype:
            case 'growth':
                linedf = self.mmodel[linevars].pct\
                    .rename().mul(100).df.pipe(lambda df: df.rename(columns={c: f'{c} %' for c in df.columns})).T
    
            case 'gdppct':
                gdpvars = [self.findgdpvar(v) for v in linevars]
                pct_des = [f'{self.mmodel.var_description[v].split(",")[0].split("mill")[0]} % ' for v in linevars]
                linedf = pd.DataFrame((govdf := self.mmodel[linevars].df).values / self.mmodel[gdpvars].df.values * 100,
                                      index=govdf.index, columns=pct_des).T
    
            case 'line':
                linetext = line.get('text', 'A line ')
                linedf = pd.DataFrame(float('nan'), index=self.mmodel.current_per, columns=[linetext]).T
    
            case _:
                raise Exception(f'No such line type: {linetype}')
    
        return linedf
   
    
    
    # def make_one_style_old(self,df,lines)  :
    #     format_decimal = [ line.get('dec',2) for line,df  in zip(self.lines,self.outdfs) for row in range(len(df))]

    #     for i, dec in enumerate(format_decimal):
    #          df.iloc[i] = df.iloc[i].apply(lambda x: f"{x:,.{dec}f}")
             
             

    def make_one_style(self,df,lines)  :
        format_decimal = [  line.get('dec',2)  for line,df  in zip(self.lines,self.outdfs) for row in range(len(df))]

        for i, dec in enumerate(format_decimal):
              df.iloc[i] = df.iloc[i].apply(lambda x:  " " if pd.isna(x) else f"{x:,.{dec}f}")
        
        
        out = df.style.set_table_styles([           
    {
        'selector': '.row_heading, .corner',
        'props': [
            ('position', 'sticky'),
            ('left', '0'),
            ('background-color', 'salmon'),
            ('width', '300px'),  # Set the width of the row headings
            ('min-width', '200px'),  # Ensure the minimum width is respected
            ('max-width', '400px')  # Ensure the maximum width is respected
        ]
    },
    {
        'selector': '.col_heading',
        'props': [
            ('position', 'sticky'),
            ('top', '0'),
            ('background-color', 'white')
        ]
    },
    {
        'selector': 'th',
        'props': [
            ('text-align', 'left')  # Align text to the left
        ]
    },
    {
        'selector': 'caption',
        'props': [
            ('font-size', '16px'),  # Make the font larger
            ('font-weight', 'bold')  # Make the font bold
        ]
    }
],overwrite=False)
        out= out.set_caption(self.options.get('caption' , 'Summary'))
        
        return out 
        
        
    def findgdpvar(self,varname):
            gdpname = varname[:3]+'NYGDPMKTP'+varname[-2:]
            if gdpname in self.mmodel.endogene: 
                return gdpname
            else: 
                raise Exception(f"{varname} don't have a matching GDP variable ")
                
    def _repr_html_(self):  
        return self.datawidget._repr_html_()
                
                
                    


        
def vis_alt(grund,mul,title='Show variables',ttop=None):
    ''' Graph of one of more variables each variable is displayed for 3 banks'''
    avar = grund.columns
    antal=len(avar)
    fig, axes = plt.subplots(nrows=antal, ncols=1,figsize=(15,antal*6))  #,sharex='col' ,sharey='row')
    fig.suptitle(title, fontsize=20)
    ax2 = [axes] if antal == 1 else axes
    for i,(var,ax) in enumerate(zip(avar,ax2)):
        grunddata = grund.loc[:,var]
        muldata = mul.loc[:,var]
        # breakpoint() 
        grunddata.plot(ax=ax,legend=False,fontsize=14)
        muldata.plot (ax=ax,legend=False,fontsize=14)
        ax.set_title(var,fontsize=14)
        x_pos = grunddata.index[-1]
        ax.text(x_pos, grunddata.values[-1],'Baseline',fontsize=14)
        ax.text(x_pos, muldata.values[-1]  ,'Alternative',fontsize=14)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,}'))

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', labelleft=True)
    fig.subplots_adjust(top=ttop if type(ttop)  != type(None) else 0.98-(0.2/antal))

    return fig
    

def plotshow(df,name='',ppos=-1,kind='line',colrow=2,sharey=False,top=None,
             splitchar='__',savefig='',*args,**kwargs):
    '''
    
Plots a subplot for each column in a datafra.
    ppos determins which split by __ to use 
    kind determins which kind of matplotlib chart to use 
     
    Args:
        df (TYPE): Dataframe .
        name (TYPE, optional): title. Defaults to ''.
        ppos (TYPE, optional): # of position to use if split. Defaults to -1.
        kind (TYPE, optional): matplotlib kind . Defaults to 'line'.
        colrow (TYPE, optional): columns per row . Defaults to 6.
        sharey (TYPE, optional): Share y axis between plots. Defaults to True.
        splitchar (TYPE, optional): if the name should be split . Defaults to '__'.
        savefig (TYPE, optional): save figure. Defaults to ''.
        xsize  (TYPE, optional): x size default to 10 
        ysize  (TYPE, optional): y size per row, defaults to 2

    Returns:
        a matplotlib fig.

    '''
    
   
    plt.ioff()
    if splitchar:
        out=df.pipe(lambda df_: df_.rename(columns={v: v.split(splitchar)[ppos] for v in df_.columns}))
    else:
        out=df
    number = out.shape[1] 
    row=-((-number)//colrow)
    # breakpoint()
    
    axes=out.plot(kind=kind,subplots=True,layout=(row,colrow),figsize = (kwargs.get('xsize',10), row*kwargs.get('ysize',2)),
                 use_index=True,title=name,sharey=sharey)
    for ax in axes.flatten():
        pass
        # dec=finddec(dfatt)
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,}'))

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', labelleft=True)
        if out.index.dtype == 'int64':
            fmtr = ticker.StrMethodFormatter('{x:.0f}')
            ax.xaxis.set_major_formatter(fmtr)
        
    fig = axes.flatten()[0].get_figure()
    fig.set_constrained_layout(True)
    fig.suptitle(name,fontsize=20)
    # fig.tight_layout()
    
    # top = (row*(2-0.1)-0.2)/(row*(2-0.1))
#    print(top)
    # fig.subplots_adjust(top=top if type(top)  != type(None) else 0.98-(0.2/row))
    if savefig:
        fig.savefig(savefig)
   # plt.subplot_tool()
    plt.ion() 
    plt.close('all')
    return fig
    
def melt(df,source='Latest'):
    ''' melts a wide dataframe to a tall dataframe , appends a soruce column ''' 
    melted = pd.melt(df.reset_index().rename(columns={'index':'time'}),id_vars='time').assign(source=source) 
    return melted

def heatshow(df,name='',cmap="Reds",mul=1.,annot=False,size=(11.69,8.27),dec=0,cbar=True,linewidths=.5):
    ''' A heatmap of a dataframe ''' 
    xx=(df.astype('float'))*mul 
#    fig, ax = plt.subplots(figsize=(11,8))
    fig, ax = plt.subplots(figsize=size)  #A4 
    sns.heatmap(xx,cmap=cmap,ax=ax,fmt="."+str(dec)+"f",annot=annot,annot_kws ={"ha": 'center'},linewidths=linewidths,cbar=cbar)
    ax.set_title(name, fontsize=20) 
    
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), ha = 'left',rotation=0)
    yax = ax.get_yaxis()
    pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=pad)
    
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), va = 'top' ,rotation=70.)
    fig.subplots_adjust(bottom=0.15)
    #ax.tick_paraOms(axis='y',direction='out', length=3, width=2, colors='b',labelleft=True)
    plt.close('all')
    return fig
##%%
def attshow(df,treshold=False,head=5000,tail=0,t=True,annot=False,showsum=False,sort=True,size=(11.69,8.27),title='',
            tshow=True,dec=0,cbar=True,cmap='jet',savefig=''):
    '''Shows heatmap of impacts of exogeneous variables
    :df: Dataframe with impact 
    :treshold: Take exogeneous variables with max impact of treshold or larger
    :numhigh: take the numhigh largest impacts
    :t: transpose the heatmap
    :annot: Annotate the heatmap
    :head: take the head largest 
    .tail: take the tail smalest 
    :showsum: Add a column with the sum 
    :sort: Sort the data
    .tshow: Show a longer title
    :cbar:  if a colorbar shoud be displayes
    :cmap: the colormap
    :save: Save the chart (in png format) '''
    
    
    selectmin = df.min().sort_values(ascending=False).tail(tail).index.tolist() 
    selectmax = df.max().sort_values(ascending=False).head(head).index.tolist() 
    select=selectmax+selectmin
    yy      = df[select].pipe(
                lambda df_ : df_[select] if sort else df_[sorted(list(df_.columns))])
    if showsum:
        asum= yy.sum(axis=1)
        asum.name = '_Sum'
        yy = pd.concat([yy,asum],axis=1)
 
    yy2 = yy.T if t  else yy
    if sort and tshow: 
        txt= ' Impact from exogeneous variables. '+(
         str(head)+' highest. ' if head >= 1 else '')+( str(tail)+' smallest. ' if tail >= 1 else '')
    else:
        txt= '' 
    f=heatshow(yy2,cmap=cmap,name=title+txt ,annot=annot,mul=1.,size=size,dec=dec,cbar=cbar)
    f.subplots_adjust(bottom=0.16) 
    if savefig:
        f.savefig(savefig)
    return yy2



def attshowone(df,name,pre='',head=5,tail=5):
    ''' shows the contribution to row=name from each column 
    the coulumns can optional be selected as starting with pre'''
    
    res = df.loc[name,[n for n in df.columns if n.startswith(pre)]].sort_values(ascending=False).pipe(
                   lambda df_: df_.head(head).append(df_.tail(tail)))
    ax =  res.plot(kind='bar')
    txt= ( str(head)+' highest. ' if head >= 1 else '')+( str(tail)+' smallest. ' if tail >= 1 else '')
    ax.set_title('Contributions to '+name+'.   '+txt)
    return ax


def water(serxinput,sort=False,ascending =True,autosum=False,allsort=False,threshold=0.0):
    ''' Creates a dataframe with information for a watrfall diagram 
    
    :serx:  the input serie of values
    :sort:  True if the bars except the first and last should be sorted (default = False)
    :allsort:  True if all bars should be sorted (default = False)
    :autosum:  True if a Total bar are added in the end  
    :ascending:  True if sortorder = ascending  
    
    Returns a dataframe with theese columns:
    
    :hbegin: Height of the first bar
    :hend: Height of the last bar
    :hpos: Height of positive bars
    :hneg: Height of negative bars
    :start: Ofset at which each bar starts 
    :height: Height of each bar (just for information)
    '''
    # get the height of first and last column 
    
   
    total=serxinput.sum() 
    serx = cutout(serxinput,threshold)
    if sort or allsort :   # sort rows except the first and last
        endslice   = None if allsort else -1
        startslice = None if allsort else  1
        i = serx[startslice:endslice].sort_values(ascending =ascending ).index
        if allsort:            
            newi =i.tolist() 
        else:
            newi =[serx.index.tolist()[0]]  + i.tolist() + [serx.index.tolist()[-1]]# Get the head and tail 
        ser=serx[newi]
    else:
        ser=serx.copy()
        
    if autosum:
        ser['Total'] = total
        
    hbegin = ser.copy()
    hbegin[1:]=0.0
    hend   = ser.copy()
    hend[:-1] = 0.0
         
    
    height = ser 
    start = ser.cumsum().shift().fillna(0.0)  # the  starting point for each bar
    start[-1] = start[-1] if allsort and not autosum else 0     # the last bar should start at 0
    end = start + ser

    hpos= height*(height>=0.0)
    hneg= height*(height<=0.0)
    dfatt = pd.DataFrame({'start':start,'hbegin':hbegin,'hpos':hpos,'hneg':hneg,'hend':hend,'height':height}).loc[ser.index,:]

    return dfatt

def waterplot(basis,sort=True,ascending =True,autosum=False,bartype='bar',threshold=0.0,
              allsort=False,title=f'Attribution ',top=0.9, desdic = {},zero=True,  ysize=5,**kwarg):
    att = [(name,water(ser,sort=sort,autosum=autosum,allsort=allsort,threshold=threshold)) 
                       for name,ser in basis.transpose().iterrows()]
    # print(att[0][1])
    fig, axis = plt.subplots(nrows=len(att),ncols=1,figsize=(10,ysize*len(att)),constrained_layout=True)
    width = 0.5  # the width of the barsser
    laxis = axis if isinstance(axis,numpy.ndarray) else [axis]
    for i,((name,dfatt),ax) in enumerate(zip(att,laxis)):
        _ = dfatt.hpos.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,
                            color='green',width=width)
        _ = dfatt.hneg.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,
                            color='red',width=width)
        _ = None if allsort else dfatt.hbegin.plot(ax=ax,kind=bartype,bottom=dfatt.start,
                stacked=True,color=('green' if dfatt.hbegin[0] > 0 else 'red') if zero else 'blue',width=width)
        _ = None if allsort and not autosum else dfatt.hend.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,color='blue',width=width)
        ax.set_ylabel(name,fontsize='x-large')
        dec=finddec(dfatt)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,.{dec}f}'))

        ax.set_title(desdic.get(name,name))
        ax.set_xticklabels(dfatt.index.tolist(), rotation = 70,fontsize='x-large')
#        plt.xticks(rotation=45, horizontalalignment='right', 
#                   fontweight='light', fontsize='x-large'  )
    fig.suptitle(title,fontsize=20)
    if 1:
        ...
        # plt.tight_layout()
        # fig.subplots_adjust(top=top)

#    plt.show()
    plt.close('all')


    return fig

vis.plot.__doc__ = plotshow.__doc__ 


if __name__ == '__main__' and 1:
    basis = pd.DataFrame([[100,100.],[-10.0,-12], [12,-10],[-10,10]],index=['nii','cost','credit','fee'],columns=['ex','ex2'])
    basis.loc['total'] = basis.sum()
    waterplot(basis)
    
 