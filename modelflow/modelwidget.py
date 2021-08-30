# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To inject methods and properties into a asia model instance, so we dont have to recreate it

@author: bruger
"""

import pandas as pd
import  ipywidgets as widgets  

from  ipysheet import sheet, cell, current 
from ipysheet.pandas_loader import from_dataframe, to_dataframe

from IPython.display import display, clear_output,Latex, Markdown
from dataclasses import dataclass,field


from modelclass import insertModelVar
from modelclass import model 


@dataclass
class basewidget:
    ''' basis for widget updating in jupyter'''
    
    ibstest : str = 'ddd'
    
    datachildren : list = field(default_factory=list) # list of children widgets 
    
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        for w in self.datachildren:
            w.update_df(df,current_per)
    
    
@dataclass
class tabwidget(basewidget):
    
    tabdefdict : any =  field(default_factory = lambda: ({}))

    
    def __post__init(self):
        
        
        self.datachildren = [tabcontent 
                             for tabcontent  in self.tabdefdict.values()]
        self.tabs = widgets.Tab(self.datachrildren, 
                                layout={'height': 'max-content'})  
         
        for i,key in enumerate(self.tabdefdict.keys()):
           self.tabs.set_title(i,key)


@dataclass
class sheetwidget(basewidget):
    ''' class defefining a widget which updates from a sheet '''
    df_var         : any = pd.DataFrame()         # definition 
    trans          : any = lambda x : x      # renaming of variables 
    transpose      : bool = False  
    
    
    def __post_init__(self):
        ...
    
        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = self.newnamedf.T if self.transpose else newnamedf
        
        self.wsheet = sheet(from_dataframe(self.org_df_var))

    def update_df(self,df,current_per=None):
        # breakpoint()
        self.df_new_var = to_dataframe(self.wsheet).T if self.transpose else to_dataframe(self.wsheet) 
        self.df_new_var.columns = self.df_var.columns
        self.df_new_var.index = self.df_var.index
        
        df.loc[self.df_new_var.index,self.df_new_var.columns] =  df.loc[ self.df_new_var.index,self.df_new_var.columns] + self.df_new_var 
        pass
        return df
 

@dataclass
class slidewidget:
    ''' class defefining a widget with lines of slides '''
    slidedef     : dict         # definition 
    altname      : str = 'Alternative'
    basename     : str =  'Baseline'
    expname      : str =  "Carbon tax rate, US$ per tonn "
    def __post_init__(self):
        ...
        
        wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
        walt  = widgets.Label(value = self.altname,layout={'width':'8%', 'border':"hide"})
        wbas  = widgets.Label(value = self.basename,layout={'width':'10%', 'border':"hide"})
        whead = widgets.HBox([wexp,walt,wbas])
        #
        wset  = [widgets.FloatSlider(description=des,
                min=cont['min'],
                max=cont['max'],
                value=cont['value'],
                step=cont.get('step',0.01),
                layout={'width':'60%'},style={'description_width':'40%'},
                readout_format = f":>,.{cont.get('dec',2)}f",
                continuous_update=False
                )
             for des,cont in self.slidedef.items()]
        
             
            
        for w in wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:>,.{cont.get('dec',2)}f}",
                layout=widgets.Layout(display="flex", justify_content="center", width="10%", border="hide"))
            for des,cont  in self.slidedef.items()
            
            ]
        self.wslide = [widgets.HBox([s,v]) for s,v in zip(wset,waltval)]
        self.slidewidget =  widgets.VBox([whead] + self.wslide)
      
        # define the result object 
        self.current_values = {des:
             {key : v.split() if key=='var' else v for key,v in cont.items() if key in {'value','var','op'}} 
             for des,cont in self.slidedef.items()} 
            
    def update_df(self,df,current_per):
        ''' updates a dataframe with the values from the widget'''
        for i,(des,cont) in enumerate(self.current_values.items()):
            op = cont.get('op','=')
            value = cont['value']
            for var in cont['var']:
                if  op == '+':
                    df.loc[current_per,var]    =  df.loc[current_per,var] + value
                elif op == '+impulse':    
                    df.loc[current_per[0],var] =  df.loc[current_per[0],var] + value
                elif op == '=start-':   
                    startindex = df.index.get_loc(current_per[0])
                    varloc = df.columns.get_loc(var)
                    df.iloc[:startindex,varloc] =  value
                elif op == '=':    
                    df.loc[current_per,var]    =   value
                elif op == '=impulse':    
                    df.loc[current_per[0],var] =   value
                else:
                    print(f'Wrong operator in {cont}.\nNot updated')
                    assert 1==3,'wRONG OPERATOR'
        
  
    def set_slide_value(self,g):
        ''' updates the new values to the self.current_vlues
        will be used in update_df
        '''
        line_des = g['owner'].description
        if 0: 
          print()
          for k,v in g.items():
             print(f'{k}:{v}')
         
        self.current_values[line_des]['value'] = g['new']
  
@dataclass
class update_tab:
    ''' class defefining a tabbed widget with slide widgets '''
    tabdef     : dict         # definition 
    
    
    
    def __post_init__(self):
        
        self.inputs = {tabname : slidewidget(slidedef) 
                   for tabname,slidedef in self.tabdef.items()}
        
        
        self.tabs = widgets.Tab([slides.slidewidget 
                   for slides in self.inputs.values()],
                                layout={'height': 'max-content'})  
        
        for i,key in enumerate(self.inputs.keys()):
          self.tabs.set_title(i,key)

    def update_df(self,df,current_per):
        for wid in self.inputs.values():
            wid.update_df(df,current_per)
                      
@dataclass
class update_input:   
    ''' class to input and run a model'''
    mmodel : any     # a model 
    tabdef  : dict 
    basename : str ='Baseline'
    varpat   : str = '*'
    showvar  : bool = True 
    
    def __post_init__(self):
        
        self.baseline = self.mmodel.basedf.copy()
        
        wgo   = widgets.Button(description="Run scenario")
        wgo.on_click(self.run)
        
        wreset   = widgets.Button(description="Reset to start")
        wzero   = widgets.Button(description="Set all to 0")
        wsetbas   = widgets.Button(description="Use as baseline")
        wbut  = widgets.HBox([wgo,wreset,wzero,wsetbas])
        
        self.wtab = update_tab(self.tabdef)
        
        self.wname = widgets.Text(value=self.basename,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
        self.wpat = widgets.Text(value= self.varpat,placeholder='Type something',description='Output variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        self.wpat.layout.visibility = 'visible' if self.showvar else 'hidden'
        

        winputstring = widgets.HBox([self.wname,self.wpat])
       
        self.wtotal = widgets.VBox([self.wtab.tabs,winputstring,wbut])
    
    
    
    def run(self,g):
        self.wtab.update_df(self.baseline,self.mmodel.current_per)
        self.mmodel(self.baseline,progressbar=1)
        self.exodif = self.mmodel.exodif()
         


  
 
if __name__ == '__main__':
    if 'masia' not in locals(): 
        print('loading model')
        masia,baseline = model.modelload('C:/wb Dawn/UNESCAP-/Asia/Asia.pcim',run=0,silent=1)    
    # test = slidewidget(masia,{})
     