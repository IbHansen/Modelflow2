# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To define Jupyter widgets which accept user input and can update dataframes and show variables. 

A user can define new widgets to suit the need for her use case. This requires some familliarity with the 
ipywidget library https://ipywidgets.readthedocs.io/en/stable/


This module is a revision of the modelwidget module. Modelwidget is still around and is used for legacy and output widgets. 

Ipywidgets are used 
to define widgets which can be used in jupyter notebooks and in shiny. 

 :Tabwidget and basewidget:  defines a **container of widgets**.
 :updatawidget: defines a **widget to update** variables using the widgets defined in the classes mentioned abowe
 :keep_plot_widget:  Defines a **widget to display** the results from the model run made in the updatewidget
 :sheetwidget slidewidget ect: defines a **basic widget** which can be dispayed and which might have the ability to 
                   update a dataframe

 
Some methods and properties of widgets.  
     :datawidget: The ipywidget which can be dispayed and might accept input used for updating a dataframe 
     :datachildren: A list of widgets in a container widget
     :update_df: a function which updates a dataframe 
     :reset: A function which resets the input values to the initial values. 
     

 

@author: Ib 
"""
# from shiny import App, event, html_dependencies, http_staticfiles, input_handler, module, reactive, render, render_image, render_plot, render_text, render_ui, req, run_app, session, types, ui
# from shinywidgets import output_widget, reactive_read, register_widget, render_widget
import ipywidgets as ipy

import pandas as pd

from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider, SelectMultiple, Layout
from ipywidgets import Select
from ipywidgets import interactive, ToggleButtons, SelectionRangeSlider, RadioButtons
from ipywidgets import interactive_output, HBox, VBox, link, Dropdown,Output
from copy import copy 

import pandas as pd
import  ipywidgets as widgets  

from  ipysheet import sheet, cell, current 
from ipysheet.pandas_loader import from_dataframe, to_dataframe

from IPython.display import display, clear_output,Latex, Markdown
from dataclasses import dataclass,field
import matplotlib.pylab  as plt 
import yaml 

from typing import Tuple



class singelwidget:
    '''Parent class for widgets
    
    Will do some common stuff''' 
    
    def __init__(self,widgetdef):
        '''
        Some common properties and methods are defines. 
        
        **Sets these properties:** 
    
         - self.widgetdef = widgetdef 
         - self.content =  self.widgetdef['content'] the content of this widget
         - self heading = self.widgetdef['heading'] the heading for this widget. 

        
        
        Parameters
        ----------
        widgetdef : dict 
            A dict with the descrition of a widget
        
    
        Returns
        -------
        None.

    

        '''
        # breakpoint() 
        self.widgetdef = widgetdef
        self.content = widgetdef['content']
        self.heading = widgetdef.get('heading','')


    def update_df(self,df,current_per):
        ''' will update container widgets'''
        try: 
            for w in self.datachildren:
                w.update_df(df,current_per)
        except: 
            ... 
            
            
    def reset(self,g):
        ''' will reset  container widgets '''
        try: 
            for w in self.datachildren:
                w.reset(g)
        except: 
            ... 



class basewidget(singelwidget):
    ''' basis for widget updating in jupyter'''
    
    

    def __init__(self,widgetdef ):
        super().__init__(widgetdef)

        # breakpoint()
        self.datachildren = [globals()[widgettype+'widget'](widgetdict)  
                            for widgettype,widgetdict in self.content  ]  
        
        self.datawidget = VBox([child.datawidget for child in self.datachildren])

    
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        for w in self.datachildren:
            w.update_df(df,current_per)
    
            
            
    def reset(self,g):
        ''' will reset  container widgets'''
        for w in self.datachildren:
            w.reset(g)

        
    
class tabwidget(singelwidget):
    '''A widget to create tab or acordium contaners'''
    
    # widgetdef : dict  
    # expname      : str =  "Carbon tax rate, US$ per tonn "
    # tab: bool =True 

    def __init__(self,widgetdef):
        # self.tabdefdict = self.widgetdef['content'] 
        super().__init__(widgetdef)
        self.selected_index  = widgetdef.get('selected_index','0')
        self.tab = widgetdef.get('tab',True)
        self.tabdefheadings = [subtabdef[0] for subtabdef in self.content] 
        
        thiswidget = widgets.Tab if self.tab else widgets.Accordion
        # breakpoint() 
        
        self.datachildren = [globals()[subwidgettype+'widget'](subwidgetdict) 
        for tabheading,(subwidgettype,subwidgetdict) in self.content ]  
        
        self.datawidget = thiswidget([child.datawidget for child in self.datachildren], 
                                     selected_index = self.selected_index,                               
                                     layout={'height': 'max-content'})  
         
        for i,key in enumerate(self.tabdefheadings):
           self.datawidget.set_title(i,key)
           
           
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        for w in self.datachildren:
            w.update_df(df,current_per)
            
            
    def reset(self,g):
        ''' will reset  container widgets'''
        for w in self.datachildren:
            w.reset(g)


        
   
class sheetwidget(singelwidget):
    ''' class defefining a widget with lines of slides '''


    def __init__(self,widgetdef ):
        ...
        super().__init__(widgetdef)    
        # print(f'{self.content=}')
        self.df_var =self.content             # input dataframe to update
        self.trans  = widgetdef.get('trans',lambda x:x)    # Translation of variable names 
        self.transpose  = widgetdef.get('transpose',False)# if the dataframs should be transpossed 
        # print(f'{self.transpose=}')
        self.wexp  = widgets.Label(value = self.heading,layout={'width':'54%'})

    
        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = newnamedf.T if self.transpose else newnamedf
        
        self.wsheet = sheet(from_dataframe(self.org_df_var),column_resizing=True,row_headers=False)
        
        self.org_values = [c.value for c in self.wsheet.cells]
        # breakpoint()
        self.datawidget=widgets.VBox([self.wexp,self.wsheet]) if len(self.heading) else self.wsheet


    def update_df(self,df,current_per=None):
        # breakpoint()
        self.df_new_var = to_dataframe(self.wsheet).T if self.transpose else to_dataframe(self.wsheet) 
        self.df_new_var.columns = self.df_var.columns
        self.df_new_var.index = self.df_var.index
        
        df.loc[self.df_new_var.index,self.df_new_var.columns] =  df.loc[ self.df_new_var.index,self.df_new_var.columns] + self.df_new_var 
        pass
        return df
    
    
    def reset(self,g):
        for c,v in zip(self.wsheet.cells,self.org_values):
            c.value = v
     

class slidewidget(singelwidget):
    ''' class defefining a widget with lines of slides '''
    


    def __init__(self,widgetdef ):
        ...
        super().__init__(widgetdef)
        self.altname = self.widgetdef .get('altname','Alternative')
        self.basename = self.widgetdef .get('basename','Baseiline')
        # breakpoint() 
        
        wexp  = widgets.Label(value = self.heading,layout={'width':'54%'})
        walt  = widgets.Label(value = self.altname,layout={'width':'8%', 'border':"hide"})
        wbas  = widgets.Label(value = self.basename,layout={'width':'10%', 'border':"hide"})
        whead = widgets.HBox([wexp,walt,wbas])
        #
        # breakpoint()
        self.wset  = [widgets.FloatSlider(description=des,
                min=cont['min'],
                max=cont['max'],
                value=cont['value'],
                step=cont.get('step',0.01),
                layout={'width':'60%'},style={'description_width':'40%'},
                readout_format = f":>,.{cont.get('dec',2)}f",
                continuous_update=False
                )
             for des,cont in self.content.items()]
        
            
        for w in self.wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:>,.{cont.get('dec',2)}f}",
                layout=widgets.Layout(display="flex", justify_content="center", width="10%", border="hide"))
            for des,cont  in self.content.items()
            
            ]
        self.wslide = [widgets.HBox([s,v]) for s,v in zip(self.wset,waltval)]
        self.slidewidget =  widgets.VBox([whead] + self.wslide)
        self.datawidget =  widgets.VBox([whead] + self.wslide)
      
        # define the result object 
        self.current_values = {des:
             {key : v.split() if key=='var' else v for key,v in cont.items() if key in {'value','var','op'}} 
             for des,cont in self.widgetdef ['content'].items()} 
            
        # register_widget(self.widget_id, self.datawidget)
    
            
    def reset(self,g): 
        for i,(des,cont) in enumerate(self.content .items()):
            self.wset[i].value = cont['value']
            
        
            
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
                elif op == '%':    
                    df.loc[current_per,var] =   df.loc[current_per,var] * (1-value/100)
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

class sumslidewidget(singelwidget):
    ''' class defefining a widget with lines of slides  the sum of the input variables can max be the parameter
    maxsum (default=1.0) '''
    def __init__(self,widgetdef ):
        ...
        super().__init__(widgetdef)
        self.altname = self.widgetdef .get('altname','Alternative')
        self.basename = self.widgetdef .get('basename','Baseiline')
        self.maxsum = self.widgetdef .get('maxsum',1.0)
       # breakpoint() 
       
        
        self.first = list(self.content.keys())[:-1]     
        self.lastdes = list(self.content.keys())[-1]     

        wexp  = widgets.Label(value = self.heading,layout={'width':'54%'})
        walt  = widgets.Label(value = self.altname,layout={'width':'8%', 'border':"hide"})
        wbas  = widgets.Label(value = self.basename,layout={'width':'10%', 'border':"hide"})
        whead = widgets.HBox([wexp,walt,wbas])
        #
        self.wset  = [widgets.FloatSlider(description=des,
                min=cont['min'],
                max=cont['max'],
                value=cont['value'],
                step=cont.get('step',0.01),
                layout={'width':'60%'},style={'description_width':'40%'},
                readout_format = f":>,.{cont.get('dec',2)}f",
                continuous_update=False,
                disabled= False
                )
             for des,cont in self.content.items()]
        
             
            
        for w in self.wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:>,.{cont.get('dec',2)}f}",
                layout=widgets.Layout(display="flex", justify_content="center", width="10%", border="hide"))
            for des,cont  in self.content.items()
            
            ]
        self.wslide = [widgets.HBox([s,v]) for s,v in zip(self.wset,waltval)]
        self.slidewidget =  widgets.VBox([whead] + self.wslide)
        self.datawidget =  widgets.VBox([whead] + self.wslide)
      
        # define the result object 
        self.current_values = {des:
             {key : v.split() if key=='var' else v for key,v in cont.items() if key in {'value','var','op','min','max'}} 
             for des,cont in self.content.items()} 
            
    def reset(self,g): 
        for i,(des,cont) in enumerate(self.content.items()):
            self.wset[i].value = cont['value']
            
        
            
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
        line_index = list(self.current_values.keys()).index(line_des)
        if 0: 
          print()
          for k,v in g.items():
             print(f'{k}:{v}')
         
        self.current_values[line_des]['value'] = g['new']
        # print(f'{self.current_values=}')
        if type(self.maxsum) == float:
            allvalues = [v['value'] for v  in self.current_values.values()]
            thissum = sum(allvalues)
            if thissum > self.maxsum:
                # print(f'{allvalues=}')
                # print(f"{self.current_values[self.lastdes]['min']=}")
                newlast = self.maxsum-sum(allvalues[:-1])
                newlast = max(newlast,self.current_values[self.lastdes]['min'])
                # print(f'{newlast=}')
                self.current_values[self.lastdes]['value']= newlast
    
                newsum= sum([v['value'] for v  in self.current_values.values()]) 
                # print(f'{newsum=}')
                # print(f'{line_index=}')
                if newsum > self.maxsum:
                    self.current_values[line_des]['value']=self.wset[line_index].value-newsum +self.maxsum
                    self.wset[line_index].value = self.current_values[line_des]['value'] 
                    
                self.wset[-1].value = newlast
                

class radiowidget(singelwidget):
    ''' class defining a widget with a number of radiobutton groups 
    
    When df_update is executed the cheked variable will be set to 1 the other to 0
    
    
    
    {'<heading 1> [['text1','variable1'],
                   ['text2','variable2']
                     .....]
     'heading 2' [['text1','variable1'],
                    ['text2','variable2']
                      .....]
     
     
     }
    
    
    
    '''
    
    def __init__(self,widgetdef):
        ...
        super().__init__(widgetdef)
        
        wexp  = widgets.Label(value = self.heading,layout={'width':'54%'})
        whead = widgets.HBox([wexp])
        #
        self.wradiolist = [widgets.RadioButtons(options=[i for i,j in cont],description=des,layout={'width':'70%'},
                                           style={'description_width':'37%'}) for des,cont in self.content.items()]
        if len(self.wradiolist) <=2:
            self.wradio = widgets.HBox(self.wradiolist)
        else: 
            self.wradio  = widgets.VBox(self.wradiolist)            
        
        self.datawidget =  widgets.VBox([whead] + [self.wradio])
            
    def reset(self,g): 
            for wradio in self.wradiolist:
                wradio.index = 0
        
            
    def update_df(self,df,current_per):
        ''' updates a dataframe with the values from the widget'''
        for wradio,(des,cont) in zip(self.wradiolist,self.content.items()):
            # print(f'{des=} {wradio.value=},{wradio.index=},{cont[wradio.index]=}')
            # print(f'{cont=}')
            for varlabel,variable in cont:
                df.loc[current_per,variable] = 0
            selected_variable = cont[wradio.index][1]
            df.loc[current_per,selected_variable] = 1 
            # print(df.loc[current_per,:])
        

class checkwidget(singelwidget):
     ''' class defefining a widget with a number of radiobutton groups 
     
     When df_update is executed the cheked variable will be set to 1 the other to 0
     
     
     
     {
      '<heading 1>:['variable1 variable2 ---',value ],
      '<heading 2>:['variable10 variable20 ---',value ]
      }
     
     
     
     '''
     def __init__(self,widgetdef):
         ...
         super().__init__(widgetdef)

         wexp  = widgets.Label(value = self.heading,layout={'width':'54%'})
         whead = widgets.HBox([wexp])
         #
         self.wchecklist = [widgets.Checkbox(description=des,value=val)   for des,(var,val) 
                            in self.content.items()]
         self.wcheck  = widgets.VBox(self.wchecklist)   
         
         self.datawidget =  widgets.VBox([whead] + [self.wcheck])
             
     def reset(self,g): 
             for wcheck,(des,(variable,val)) in zip(self.wchecklist,self.content.items()):
                 wcheck.value = val
         
             
     def update_df(self,df,current_per):
         ''' updates a dataframe with the values from the widget'''
         for wcheck,(des,(variable,val)) in zip(self.wchecklist,self.content.items()):
             for selected_variable in  variable.split(): 
                 df.loc[current_per,selected_variable] = 1.0 if wcheck.value else 0.0
 

                      
@dataclass
class updatewidget:   
    ''' class to input and run a model
    
    - display(wtotal) to activate the widget 
    
    '''
    
    mmodel : any     # a model 
    datawidget : any # a widget  to update from  
    basename : str ='Business as usual'
    keeppat   : str = '*'
    varpat    : str ='*'
    showvarpat  : bool = True    # Show varpaths to 
    exodif   : any = field(default_factory=pd.DataFrame)          # definition 
    lwrun    : bool = True
    lwupdate : bool = False
    lwreset  :  bool = True
    lwsetbas  :  bool = True
    outputwidget : str  = 'jupviz'
    display_first :any = None 
  # to the plot widget  

    prefix_dict : dict = field(default_factory=dict)
    vline  : list = field(default_factory=list)
    relativ_start : int = 0 
    short :bool = False 
    
    
    
    def __post_init__(self):
        self.baseline = self.mmodel.basedf.copy()
        self.wrun    = widgets.Button(description="Run scenario")
        self.wrun .on_click(self.run)
        self.wrun .tooltip = 'Click to run'
        self.wrun.style.button_color = 'Lime'

        
        wupdate   = widgets.Button(description="Update the dataset ")
        wupdate.on_click(self.update)
        
        wreset   = widgets.Button(description="Reset to start")
        wreset.on_click(self.reset)

        
        wsetbas   = widgets.Button(description="Use as baseline")
        wsetbas.on_click(self.setbasis)
        self.experiment = 0 
        
        lbut = []
        
        if self.lwrun: lbut.append(self.wrun )
        if self.lwupdate: lbut.append(wupdate)
        if self.lwreset: lbut.append(wreset)
        if self.lwsetbas : lbut.append(wsetbas)
        
        wbut  = widgets.HBox(lbut)
        
        
        self.wname = widgets.Text(value=self.basename,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
        self.wselectfrom = widgets.Text(value= self.varpat,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        self.wselectfrom.layout.visibility = 'visible' if self.showvarpat else 'hidden'

        winputstring = widgets.HBox([self.wname,self.wselectfrom])
       
    
        self.mmodel.keep_solutions = {}
        self.mmodel.keep_solutions = {self.wname.value : self.baseline}
        self.mmodel.keep_exodif = {}
        
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        
        self.wtotal = VBox([widgets.HTML(value="Hello <b>World</b>")])  
        
        def init_run(g):
            # print(f'{g=}')
            self.varpat = g['new']
            self.keep_ui = keep_plot_widget(mmodel = self.mmodel, 
                                      selectfrom = self.varpat, prefix_dict=self.prefix_dict,
                                      vline=self.vline,relativ_start=self.relativ_start,
                                      short = self.short) 
            
            # self.wtotal = widgets.VBox([self.datawidget.datawidget,winputstring,wbut,
            #                             self.keep_ui.datawidget])
            self.wtotal.children  = [self.datawidget.datawidget,winputstring,wbut,
                                        self.keep_ui.datawidget]
            
            self.start = copy(self.mmodel.current_per[0])
            self.end = copy(self.mmodel.current_per[-1])
            
        self.wselectfrom.observe(init_run,names='value',type='change')

        init_run({'new':self.varpat})
    
    def update(self,g):
        self.thisexperiment = self.baseline.copy()
        # print(f'update smpl  {self.mmodel.current_per=}')
        
        self.datawidget.update_df(self.thisexperiment,self.mmodel.current_per)
        self.exodif = self.mmodel.exodif(self.baseline,self.thisexperiment)
        # print(f'update 2 smpl  {self.mmodel.current_per[0]=}')


              
        
    def run(self,g):
        self.update(g)
        # print(f'run  smpl  {self.mmodel.current_per[0]=}')
        # self.start = self.mmodel.current_per[0]
        # self.end = self.mmodel.current_per[-1]
        # print(f'{self.start=}  {self.end=}')
        self.wrun .tooltip = 'Running'
        self.wrun.style.button_color = 'Red'

        self.mmodel(self.thisexperiment,start=self.start,end=self.end,progressbar=0,keep = self.wname.value,                    
                keep_variables = self.keeppat)
        self.wrun .tooltip = 'Click to run'
        self.wrun.style.button_color = 'Lime'
        # print(f'run  efter smpl  {self.mmodel.current_per[0]=}')
        self.mmodel.keep_exodif[self.wname.value] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True
        self.current_experiment = self.wname.value
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        self.keep_ui.trigger(None) 
        
    def setbasis(self,g):
        self.mmodel.keep_solutions={self.current_experiment:self.mmodel.keep_solutions[self.current_experiment]}
        
        self.mmodel.keep_exodif[self.current_experiment] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True

        
        
         
    def reset(self,g):
        self.datawidget.reset(g) 

def fig_to_image(fig,format='svg'):
    from io import StringIO
    f = StringIO()
    fig.savefig(f,format=format,bbox_inches="tight")
    f.seek(0)
    image= f.read()
    return image
  
    


@dataclass
class keep_plot_widget:
    
    """
      Returns an interactive widget to plots the keept dataframes

      Args:
          mmodel : a modelflow model instance 
          smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
          selectfrom (list, optional): the variables to select from, Defaults to [] -> all keept  variables .
          legend (bool, optional): DESCRIPTION. legends or to the right of the curve. Defaults to 1.
          dec (string, optional): decimals on the y-axis. Defaults to '0'.
          use_descriptions : Use the variable descriptions from the model 
          vline: List of vertical lines (position,text)
          add_var_name: Add the variable name to description  
          short: Short, 1 2 cut down on the inpout fields 
          select_scenario: If True, select the scenarios which has to be displayed
          switch : if True use the scenarios in mmodel.basedf and mmodel.lastdf 
          var_groups: a dictionary of variable pattern  to select for instance countries (if not present var_groups from model is used)
          use_var_groups: use the var:groupos 
          
          displaytype : string type one of ['tab','accordion','anything']
          save_location: Default save location 
          
      property: 
          show: Display the widget
          datawidget: the actual widget 

      Returns:
          keep_plot_widget instance .

      self.keep_wiz_figs is set to a dictionary containing the figures. Can be used to produce publication
      quality files. 

    """

    
    mmodel : any     # a model 
    # pat : str ='*'
    # showvarpat : bool = True 
    smpl : Tuple[int, int] = ('', '')
    relativ_start : int = 0 
    selected  : str =''
    
    selectfrom  : str ='*'
    showselectfrom : bool = True 

    legend : bool = False 
    dec=''
    use_descriptions : bool = True
    select_width=''
    select_height='200px'
    vline : any = None

    var_groups : dict = field(default_factory=dict)
    use_var_groups : bool = True 
    add_var_name : bool = False
    short :any  = 0 
    multi :any = False 
    select_scenario : bool = False
    displaytype : str = 'tab' # or '' or accordion
    save_location :str = './graph'
    switch :bool = False
    use_smpl : bool = False 
    
  
    
    def __post_init__(self):
        from copy import copy 
        
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.first_prefix = True 
        
        self.old_current_per = copy(self.mmodel.current_per) 
        # print(f'Før {self.mmodel.current_per=}')
        
        self.select_scenario = False if self.switch else self.select_scenario
       
        with self.mmodel.set_smpl(*self.smpl):
            # print(f'efter set smpl  {self.mmodel.current_per=}')
            with self.mmodel.set_smpl_relative(self.relativ_start,0):
                # print(f'efter set smpl relativ  {self.mmodel.current_per=}')
                ...
                show_per = copy(list(self.mmodel.current_per)[:])
        self.mmodel.current_per = copy(self.old_current_per)       
        
        self.save_dialog = savefigs_widget(location = self.save_location)
        # Now variable selection 

        allkeepvar = [set(df.columns) for df in self.mmodel.keep_solutions.values()]
        keepvar = sorted(allkeepvar[0].intersection(*allkeepvar[1:]))   # keept variables in all experiments

        
        wselectfrom = widgets.Text(value= self.selectfrom,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        wselectfrom.layout.visibility = 'visible' if self.showselectfrom else 'hidden'
        
        self.wxopen = widgets.Checkbox(value=False,description = 'Allow save figures',disabled=False,
                                     layout={'width':'25%'}    ,style={'description_width':'5%'})
        
        gross_selectfrom = []        
        def changeselectfrom(g):
            nonlocal gross_selectfrom
            _selectfrom = [s.upper() for s in self.mmodel.vlist(wselectfrom.value) if s in keepvar  ] if self.selectfrom else keepvar
            gross_selectfrom =  [(f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}',v)
                                 for v in _selectfrom] 
            try: # Only if this is done after the first call
                selected_vars.options=gross_selectfrom
                selected_vars.value  = [gross_selectfrom[0][1]]
            except: 
                ...
        wselectfrom.observe(changeselectfrom,names='value',type='change')   
        self.wxopen.observe(self.trigger,names='value',type='change')
        
        changeselectfrom(None)
            
            

        # Now the selection of scenarios 
        
        
        with self.mmodel.keepswitch(switch=self.switch,scenarios= '*'):
            gross_keys = list(self.mmodel.keep_solutions.keys())
       
            scenariobase = Select(options = gross_keys, 
                                  value = gross_keys[0],
                                               description='First scenario',
                                               layout=Layout(width='50%', font="monospace"))
            scenarioselect = SelectMultiple(options = [s for s in gross_keys if not s == scenariobase.value],
                                            value=  [s for s in gross_keys if not s == scenariobase.value],
                                            description = 'Next',
                                               layout=Layout(width='50%', font="monospace"))
            keep_keys = list(self.mmodel.keep_solutions.keys())
            self.scenarioselected = '|'.join(keep_keys)
            keep_first = keep_keys[0]

        
        def changescenariobase(g):
            scenarioselect.options=[s for s in gross_keys if not s == scenariobase.value]
            scenarioselect.value  =[s for s in gross_keys if not s == scenariobase.value]
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value) )
            # print(f'{self.scenarioselected=}')
           
            
        def changescenarioselect(g):
            # print(f'{scenarioselect.value}')
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value) )
            self.trigger(None)
            diff.description = fr'Difference to: "{scenariobase.value}"'
            # print(f'{self.scenarioselected=}')
            
        scenariobase.observe(changescenariobase,names='value',type='change')
        scenarioselect.observe(changescenarioselect,names='value',type='change')
        
        wscenario = HBox([scenariobase, scenarioselect])
        
        
        wscenario.layout.visibility = 'visible' if self.select_scenario else 'hidden'
            
        
        
        
        
        # print(f'efter context  set smpl  {self.mmodel.current_per=}')
        # breakpoint() 
        init_start = self.mmodel.lastdf.index.get_loc(show_per[0])
        init_end = self.mmodel.lastdf.index.get_loc(show_per[-1])
        # print(f'{gross_selectfrom=}')
        width = self.select_width if self.select_width else '50%' if self.use_descriptions else '50%'
    

        description_width = 'initial'
        description_width_long = 'initial'
        
        if self.use_var_groups:
            if len(self.var_groups):
                self.prefix_dict = self.var_groups
            elif hasattr(self.mmodel,'var_groups') and len(self.mmodel.var_groups):
                self.prefix_dict = self.mmodel.var_groups
            else: 
                self.prefix_dict = {}
                
        
        select_prefix = [(iso,c) for iso,c in self.prefix_dict.items()]
        # print(select_prefix)
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        selected_vars = SelectMultiple( options=gross_selectfrom, layout=Layout(width=width, height=self.select_height, font="monospace"),
                                        description='Select one or more', style={'description_width': description_width})
        
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=False, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                              value='linear', style={'description_width': description_width})
        # 
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=self.legend, style={
                              'description_width': description_width},layout=Layout(width='auto',margin="0% 0% 0% 5%"))
        
        self.widget_dict = {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend}
        
        
        
        for wid in self.widget_dict.values():
            wid.observe(self.trigger,names='value',type='change')
            
        # breakpoint()
        self.out_widget =VBox([widgets.HTML(value="Hello <b>World</b>",
                                  placeholder='',
                                  description='',)])
        
       
        
        def get_prefix(g):
            from modelclass import model
            ''' this function is triggered when the prefix selection is changed. Used only when a  prefix_dict has ben set. 
            
            g['new]'] contains the new prefix
            
            It sets the variables the user can select from 
            
            '''
            # print(f'{g=}')
            try:
                # fint the suffix for the variables in the current selection of variables
                # to ensure we get the same variables for the next prefix 
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except:
                # we are at the first call to getprefix 
                current_suffix = ''
                
            new_prefix = g['new']  # returns the selected prefix as tuple as therre can be more 
            # print(new_prefix)
            # find the variables which starts with the prefix 
            if 0: # only prefix
                selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                        if any([variable.startswith(n) 
                                                
                                                for n in new_prefix]))
            else: 
                gross_selectfrom_vars = [variable for des,variable in gross_selectfrom ]
                gross_pat  = ' '.join([ppat for ppat in new_prefix])
                selected_match_var = set(model.list_names(gross_selectfrom_vars,gross_pat))
                
                selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                        if variable in selected_match_var)
                
            # print(f'{selected_prefix_var=}')
            # An exception is trigered but has no consequences     
            try:                      
                selected_vars.options = selected_prefix_var # Sets the variables to be selected from 
            except: 
                ...
                
                
            # print(f'{current_suffix=} \n{selected_prefix_var=}')

            if not self.first_prefix:
                new_selection   = [f'{n}{c}' for c in current_suffix for n in new_prefix
                                        if f'{n}{c}' in {s  for p,s in selected_prefix_var}]
                selected_vars.value  = new_selection 
                # print(f"{new_selection=}{current_suffix=}{g['old']=}")
            else:    
                # we are where no prefix has been selected
                self.first_prefix = False 
                if len(selected_prefix_var):
                    selected_vars.value  = [varname for des,varname in  selected_prefix_var]
                
                   
        if len(self.prefix_dict): 
            selected_prefix = SelectMultiple(value=[select_prefix[0][1]], options=select_prefix, 
                                              layout=Layout(width='25%', height=self.select_height, font="monospace"),
                                        description='')
               
            selected_prefix.observe(get_prefix,names='value',type='change')
            select = HBox([selected_vars,selected_prefix])
            # print(f'{select_prefix=}')
            # we call get_prefix when the widget is set up. 
            get_prefix({'new':select_prefix[0]})
        else: 
            # no prefix, so we only have to select variable names. 
            select = VBox([selected_vars])
            selected_vars.value  = [gross_selectfrom[0][1]]

        options1 = HBox([diff]) if self.short >=2 else HBox([diff,legend])
        options2 = HBox([scale, showtype, self.wxopen])
        if self.short:
            vui = [select, options1,]
        else:
            if self.select_scenario:
                vui = [wscenario, select, options1, options2,]  
            else: 
                vui = [           select, options1, options2,]  
        vui =  vui + [i_smpl] if self.use_smpl else vui 
        
        self.datawidget= VBox(vui+[self.out_widget])
        # show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
        #                                     'scale': scale, 'legend': legend})
        # print('klar til register')
        
        
    @property
    def show(self):
        display(self.datawidget)

        
        

    def _repr_html_(self):
        display( self.datawidget)
        
    def __repr__(self):
        return ' '

        
    def explain(self, i_smpl=None, selected_vars=None, diff=None, showtype=None, scale=None, legend= None):
        # print(f'Start explain:\n {self.mmodel.current_per[0]=}\n{selected_vars=}\n')
        variabler = ' '.join(v for v in selected_vars)
        smpl = (self.mmodel.lastdf.index[i_smpl[0]], self.mmodel.lastdf.index[i_smpl[1]])
        if type(diff) == str:
            diffpct = True
            ldiff = False
        else: 
            ldiff = diff
            diffpct = False
        self.save_dialog.waddname.value =(
                    ('_level'   if showtype == 'level' else '_growth') +   
                    ('_diff'    if ldiff     else ''  )+
                    ('_diffpct' if diffpct   else '' ) + 
                    ('_log'     if scale == 'log' else '')
                    )
        # print(self.save_dialog.addname)
        
        # clear_output()
        with self.mmodel.keepswitch(switch=self.switch,scenarios= self.scenarioselected):
            with self.mmodel.set_smpl(*smpl):
    
                self.keep_wiz_figs = self.mmodel.keep_plot(variabler, diff=ldiff, diffpct = diffpct, 
                                                    scale=scale, showtype=showtype,
                                                    showfig=False,
                                                    legend=legend, dec=self.dec, vline=self.vline)
                plt.close('all')
                return self.keep_wiz_figs  
            
            
        # print(f'Efter plot {self.mmodel.current_per=}')


    def  trigger(self,g):
        self.mmodel.current_per = copy(self.old_current_per)     
        

        values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'Triggerd:\n{values} \n Start trigger {self.mmodel.current_per=}\n')
        if len(values['selected_vars']):

            figs = self.explain(**values)
            figlist = [widgets.HTML(fig_to_image(a_fig),width='100%',format = 'svg',layout=Layout(width='90%'))  
                      for key,a_fig in figs.items() ]
            
            if self.displaytype in { 'tab' , 'accordion'}:
                ...
                wtab = widgets.Tab(figlist) if  self.displaytype  == 'tab' else  widgets.Accordion(figlist)
                for i,v in enumerate(figs.keys()):
                   wtab.set_title(i,f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}')
                wtab.selected_index = 0   
                res = [wtab]

            else:     
                res = figlist 
            
            self.save_dialog.figs = figs
            self.out_widget.children = (res + [self.save_dialog.datawidget]) if self.wxopen.value else res
            self.out_widget.layout.visibility = 'visible'
            # print(f'end  trigger {self.mmodel.current_per[0]=}')
        else: 
            self.out_widget.layout.visibility = 'hidden'

@dataclass
class savefigs_widget:
    '''
    A widget ó save figs from a dictionary of matplotlib figures
    

    '''
    figs : dict =  field(default_factory=dict)
    location : str = './graph'
    addname : str  =  ''
        
    def __post_init__(self):
        
        wgo     = widgets.Button(description = 'Save charts to file',colour='green')
        
        wlocation =  widgets.Text(value = self.location,description='Save location:',
                            layout={'width':'55%'},style={'description_width':'35%'})
        wexperimentname = widgets.Text(value='Experiment_1',description='Name of these experiments:',
                            layout={'width':'55%'},style={'description_width':'35%'})
        self.waddname  = widgets.Text(value=self.addname,placeholder='If needed type a suffix',description='suffix for these charts:',
                                    layout={'width':'55%'},style={'description_width':'35%'})

        wextensions = widgets.SelectMultiple(value = ('svg',),
                                             options =  ['svg', 'pdf', 'png', 'eps'],
                                             description='Output type:',
                                    layout={'width':'55%'},style={'description_width':'35%'},rows=4)
        wxopen = widgets.Checkbox(value=True,description = 'Open location',disabled=False,
                                     layout={'width':'25%'}    ,style={'description_width':'5%'})
        wsavelocation =  widgets.Text(value = self.location,description='Saved at:',
                            layout={'width':'55%'},style={'description_width':'35%'})
        wsavelocation.layout.visibility = 'hidden'
        
        def go(g):
            from modelclass import model 
            # print(f'{self.figs.keys()}')
            result = model.savefigs(self.figs,
                           location       = wlocation.value,
                           experimentname = wexperimentname.value,
                           addname        = self.waddname.value,
                           extensions    =  wextensions.value,
                           xopen           = wxopen.value)
            wsavelocation.description,wsavelocation.value = result
            wsavelocation.layout.visibility = 'visible'
            
                           
                           
        wgo.on_click(go)
        
        self.datawidget = widgets.VBox([HBox([wgo,wxopen]),
                        wexperimentname,wlocation,self.waddname,wextensions,wsavelocation])
        
        
        
        
   
        
@dataclass
class shinywidget:
    '''A to translate a widget to shiny widget '''

    a_widget : any # The datawidget to wrap 
    widget_id  : str 
    
    def __post_init__(self):
        from shinywidgets import register_widget
        ...        
        register_widget(self.widget_id, self.a_widget.datawidget)
       
           
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        self.a_widget.update_df(df,current_per)
            
            
    def reset(self,g):
        ''' will reset  container widgets'''
        self.a_widget.reset(g) 

def make_widget(widgetdef):
    '''
    '''
    
    widgettype,widgetdict  = widgetdef
    reswidget = globals()[widgettype+'widget'](widgetdict)
    return reswidget


if __name__ == '__main__':

    slidedef =  ['slide', {'heading':'Dette er en prøve', 
                'content' :{'Productivity'    : {'var':'ALFA',             'value': 0.5 ,'min':0.0, 'max':1.0},
                 'DEPRECIATES_RATE' : {'var':'DEPRECIATES_RATE', 'value': 0.05,'min':0.0, 'max':1.0}, 
                 'LABOR_GROWTH'     : {'var':'LABOR_GROWTH',     'value': 0.01,'min':0.0, 'max':1.0},
                 'SAVING_RATIO'     : {'var': 'SAVING_RATIO',    'value': 0.05,'min':0.0, 'max':1.0}
                        }}  ]
    
    radiodef =   ['radio', {'heading':'Dette er en anden prøve', 
                'content' : {'Ib': [['Arbejdskraft','LABOR_GROWTH'],
                       ['Opsparing','SAVING_RATIO'] ],
               'Søren': [['alfa','ALFA'],
                       ['a','A'] ],
               'Marie': [['Hest','HEST'],
                       ['Ko','KO'] ],
               
               }} ]
    
    checkdef =   ['check', {'heading':'Dette er en treidje prøve', 
                'content' :{'Arbejdskraft':['LABOR_GROWTH KO',1],
                           'Opsparing'  :['SAVING_RATIO',0] }}]
    
    alldef = [slidedef] + [radiodef] + [checkdef]
                  
    
    tabdef = ['tab',{'content':[['Den første',['base',{'content':alldef}]],
                                ['Den anden' ,['base',{'content':alldef}]]]
                                ,
                    'tab':False}]
    
    basedef = ['base',{'content':[slidedef,radiodef]}]
        
    xx = make_widget(tabdef)
     
    # print(yaml.dump(tabdef,indent=4))
    
