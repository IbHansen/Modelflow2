# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To inject methods and properties into a asia model instance, so we dont have to recreate it

@author: bruger
"""

import pandas as pd
import  ipywidgets as widgets  
from IPython.display import display, clear_output,Latex, Markdown
from dataclasses import dataclass


from modelclass import insertModelVar
from modelclass import model 



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
        walt  = widgets.Label(value = self.altname,layout={'width':'10%'})
        wbas  = widgets.Label(value = self.basename,layout={'width':'20%'})
        whead = widgets.HBox([wexp,walt,wbas])
        #
        wset  = [widgets.FloatSlider(description=des,
                min=cont['min'],
                max=cont['max'],
                value=cont['value'],
                step=cont.get('step',0.01),
                layout={'width':'60%'},style={'description_width':'40%'},
                readout_format = f":<,.{cont.get('dec',2)}f",
                continuous_update=False
                )
             for des,cont in self.slidedef.items()]
        
             
            
        for w in wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:<,.{cont.get('dec',2)}f}",
                layout={'width':'10%'})
            for des,cont  in self.slidedef.items()]
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
    ''' class defefining a widget with lines of slides '''
    tabdef     : dict         # definition 
    
    
    
    def __post_init__(self):
        
        self.inputs = {tabname : slidewidget(slidedef) 
                   for tabname,slidedef in self.tabdef.items()}
        
        
        self.tabs = widgets.Tab([slides.slidewidget 
                   for tabname,slides in self.inputs.items()])  
        
        for i,key in enumerate(self.inputs.keys()):
          self.tabs.set_title(i,key)

    def update_df(self,df,current_per):
        for wid in self.input.values():
            wid.update_df(df,current_per)
            
        
 
if __name__ == '__main__':
    if 'masia' not in locals(): 
        print('loading model')
        masia,baseline = model.modelload('C:/wb Dawn/UNESCAP-/Asia/Asia.pcim',run=1,silent=1)    
    # test = slidewidget(masia,{})
