# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To define Jupyter widgets to update and show variables. 

this version is made to make is easy to define input widgets as dictikonaries. 

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


class singelwidget:
    '''Parent class for widgets
    
    Will do some common stuff''' 
    
    def __init__(self,widgetdef):
        '''
        

        Parameters
        ----------
        widgetdef : dict 
            A dict with the descrition of a widget
        
        Sets properties 
        ----------------
         widgetdef['content'] the content of this widget
         widgetdef['heading'] the heading for this widget. 
    
        Returns
        -------
        None.

        '''
        # breakpoint() 
        self.widgetdef = widgetdef
        self.content = widgetdef['content']
        self.heading = widgetdef.get('heading','')


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


        
# @dataclass
# class sheetwidget:
#     ''' class defining a widget which updates from a sheet '''
#     df_var         : any = pd.DataFrame()         # definition 
#     trans          : any = lambda x : x      # renaming of variables 
#     transpose      : bool = False            # orientation of dataframe 
#     expname      : str =  "Carbon tax rate, US$ per tonn "
   
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
    ''' class defefining a widget with lines of slides '''
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
    ''' class defefining a widget with a number of radiobutton groups 
    
    When df_update is executed the cheked variable will be set to 1 the other to 0
    
    
    
    {'<heading 1> [['text1','variable1'],
                   ['text2','variable2']
                     .....]
     'heading 2' [['text1','variable1'],
                    ['text2','variable2']
                      .....]
     
     ....
     ....
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
    exodif   : any = pd.DataFrame()         # definition 
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
        self.wpat = widgets.Text(value= self.varpat,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        self.wpat.layout.visibility = 'visible' if self.showvarpat else 'hidden'
        

        winputstring = widgets.HBox([self.wname,self.wpat])
       
    
        self.mmodel.keep_solutions = {}
        self.mmodel.keep_solutions = {self.wname.value : self.baseline}
        self.mmodel.keep_exodif = {}
        
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        
        self.keep_ui = keep_plot_shiny(mmodel = self.mmodel, 
                                  selectfrom = self.varpat, prefix_dict=self.prefix_dict,
                                  vline=self.vline,relativ_start=self.relativ_start,
                                  short = self.short) 
        
        self.wtotal = widgets.VBox([self.datawidget.datawidget,winputstring,wbut,
                                    self.keep_ui.datawidget])
        
        self.start = copy(self.mmodel.current_per[0])
        self.slut = copy(self.mmodel.current_per[-1])


    
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
        # self.slut = self.mmodel.current_per[-1]
        # print(f'{self.start=}  {self.slut=}')
        self.wrun .tooltip = 'Running'
        self.wrun.style.button_color = 'Red'

        self.mmodel(self.thisexperiment,start=self.start,slut=self.slut,progressbar=0,keep = self.wname.value,                    
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
class htmlwidget_df:
    ''' class displays a dataframe in a html widget '''
    
    mmodel : any     # a model   
    df_var         : any = pd.DataFrame()         # definition 
    trans          : any = lambda x : x      # renaming of variables 
    transpose      : bool = False            # orientation of dataframe 
    expname      : str =  "Test"
    percent      : bool = False 
   
    
    def __post_init__(self):
        ...
        
   
    
        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = newnamedf.T if self.transpose else newnamedf
        self.wexp  = widgets.Label(value = f'{self.expname} {self.org_df_var.shape}',layout={'width':'54%'})
        self.datawidget=self.wexp
        # return
        image = self.mmodel.ibsstyle(self.org_df_var,percent = self.percent).to_html()
        # image = self.org_df_var.to_html()
        self.whtml = widgets.HTML(image)
        # self.datawidget=widgets.VBox([self.wexp,self.whtml]) if len(self.expname) else self.whtml
        self.datawidget=widgets.VBox([self.wexp,self.whtml]) if len(self.expname) else self.whtml
        
    @property
    def show(self):
        display(self.datawidget)
        

@dataclass
class htmlwidget_fig:
    ''' class displays a dataframe in a html widget '''
    
    figs : any     # a model   
    expname      : str =  ""
    format       : str =  "svg"
  
    
    def __post_init__(self):
        ...
        
        self.wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
   
    
        image = fig_to_image(self.figs,format=self.format)
        print(image)
        # self.whtml = widgets.HTML(image,layout={'width':'54%'})
        self.whtml = widgets.Image(image,width=300,format = 'svg')
        self.datawidget=widgets.VBox([self.wexp,self.whtml]) if len(self.expname) else self.whtml
        
        
        
@dataclass
class htmlwidget_label:
    ''' class displays a dataframe in a html widget '''
    
    expname      : str =  ""
    format       : str =  "svg"
  
    
    def __post_init__(self):
        ...
        
        self.wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
 
    
        self.datawidget=self.wexp

@dataclass
class visshow:
    mmodel : any     # a model 
    varpat    : str ='*' 
    showvarpat  : bool = True 
    show_on   : bool = True  # Display when called 
        
    def __post_init__(self):
        ...
        from IPython import get_ipython
        try:
            get_ipython().magic('matplotlib notebook')
        except: 
            ...
        # print(plt.get_backend())
        this_vis = self.mmodel[self.varpat]
        self.out_dict = {}
        self.out_dict['Baseline'] ={'df':this_vis.base}        
        self.out_dict['Alternative'] ={'df':this_vis} 
        self.out_dict['Difference'] ={'df':this_vis.dif} 
        self.out_dict['Diff. pct. level'] ={'df':this_vis.difpctlevel.mul100,'percent':True} 
        self.out_dict['Base growth'] ={'df':this_vis.base.pct.mul100,'percent':True} 
        self.out_dict['Alt. growth'] ={'df':this_vis.pct.mul100,'percent':True} 
        self.out_dict['Diff. in growth'] ={'df':this_vis.difpct.mul100,'percent':True} 
        
        out = widgets.Output() 
        self.out_to_data = {key:
            htmlwidget_df(self.mmodel,value['df'].df.T,expname=key,

                          percent=value.get('percent',False))                           
                           for key,value in self.out_dict.items()}
            
        with out: # to suppress the display of matplotlib creation 
            self.out_to_figs ={key:
                 htmlwidget_fig(value['df'].rename().plot(top=0.9,title=''),expname='')               
                              for key,value in self.out_dict.items() }
                
       
        tabnew =  {key: tabwidget({'Charts':self.out_to_figs[key],'Data':self.out_to_data[key]},selected_index=0) for key in self.out_to_figs.keys()}
       
        exodif = self.mmodel.exodif()
        exonames = exodif.columns        
        exoindex = exodif.index
        if len(exonames):
            exobase = self.mmodel.basedf.loc[exoindex,exonames]
            exolast = self.mmodel.lastdf.loc[exoindex,exonames]
            
            out_exodif = htmlwidget_df(self.mmodel,exodif.T)
            out_exobase = htmlwidget_df(self.mmodel,exobase.T)
            out_exolast = htmlwidget_df(self.mmodel,exolast.T)
        else: 
            out_exodif = htmlwidget_label(expname = 'No difference in exogenous')
            out_exobase =  htmlwidget_label(expname = 'No difference in exogenous')
            out_exolast =  htmlwidget_label(expname = 'No difference in exogenous')
            
        
        out_tab_info = tabwidget({'Delta exogenous':out_exodif,
                                  'Baseline exogenous':out_exobase,
                                  'Alt. exogenous':out_exolast},selected_index=0)
        
        tabnew['Exo info'] = out_tab_info
        
        this =   tabwidget(tabnew,selected_index=0)
            
            
        
        self.datawidget = this.datawidget  
        if self.show_on:
            ...
            display(self.datawidget)
        try:    
            get_ipython().magic('matplotlib inline') 
        except:
            ...

    def __repr__(self):
        return ''

    def _html_repr_(self):  
        return self.datawidget 
         
    @property
    def show(self):
        display(self.datawidget)
    

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

@dataclass
class keep_plot_shiny:
    mmodel : any     # a model 
    pat : str ='*'
    smpl=('', '')
    relativ_start : int = 0 
    selectfrom  : str ='*'
    legend=0
    dec=''
    use_descriptions=True
    select_width=''
    select_height='200px'
    vline : list = field(default_factory=list)
    prefix_dict : dict = field(default_factory=dict)
    add_var_name=False
    short :any  = 0 
    multi :any = False 
    
    """
     Plots the keept dataframes

     Args:
         pat (str, optional): a string of variables to select pr default. Defaults to '*'.
         smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
         selectfrom (list, optional): the variables to select from, Defaults to [] -> all keept  variables .
         legend (bool, optional)c: DESCRIPTION. legends or to the right of the curve. Defaults to 1.
         dec (string, optional): decimals on the y-axis. Defaults to '0'.
         use_descriptions : Use the variable descriptions from the model 

     Returns:
         None.

     self.keep_wiz_figs is set to a dictionary containing the figures. Can be used to produce publication
     quality files. 

    """

    
    def __post_init__(self):
        from copy import copy 
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.old_current_per = copy(self.mmodel.current_per) 
        # print(f'Før {self.mmodel.current_per=}')
        with self.mmodel.set_smpl(*self.smpl):
            # print(f'efter set smpl  {self.mmodel.current_per=}')
            with self.mmodel.set_smpl_relative(self.relativ_start,0):
               # print(f'efter set smpl relativ  {self.mmodel.current_per=}')
               ...
               show_per = copy(list(self.mmodel.current_per)[:])
        self.mmodel.current_per = copy(self.old_current_per)        
        # print(f'efter context  set smpl  {self.mmodel.current_per=}')
        # breakpoint() 
        init_start = self.mmodel.lastdf.index.get_loc(show_per[0])
        init_end = self.mmodel.lastdf.index.get_loc(show_per[-1])
        keepvar = sorted (list(self.mmodel.keep_solutions.values())[0].columns)
        defaultvar = ([v for v in self.mmodel.vlist(self.pat) if v in keepvar][0],) # Default selected
        # print('******* selectfrom')
        # variables to select from 
        _selectfrom = [s.upper() for s in self.mmodel.vlist(self.selectfrom)] if self.selectfrom else keepvar
        # _selectfrom = keepvar
        # print(f'{_selectfrom=}')
        gross_selectfrom =  [(f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}',v) for v in _selectfrom] 
        # print(f'{gross_selectfrom=}')
        width = self.select_width if self.select_width else '50%' if self.use_descriptions else '50%'
    

        description_width = 'initial'
        description_width_long = 'initial'
        keep_keys = list(self.mmodel.keep_solutions.keys())
        keep_first = keep_keys[0]
        select_prefix = [(c,iso) for iso,c in self.prefix_dict.items()]
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        selected_vars = SelectMultiple( options=gross_selectfrom, value=gross_selectfrom[0], layout=Layout(width=width, height=self.select_height, font="monospace"),
                                       description='Select one or more', style={'description_width': description_width})
        
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=False, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                             value='linear', style={'description_width': description_width})
        # 
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=self.legend, style={
                              'description_width': description_width})
        
        self.widget_dict = {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend}
        for wid in self.widget_dict.values():
            wid.observe(self.trigger,names='value',type='change')
            
        # breakpoint()
        self.out_widget = VBox([widgets.HTML(value="Hello <b>World</b>",
                                  placeholder='',
                                  description='',)])
        
        # values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'i post_init:\n{values} \n Start trigger {self.mmodel.current_per=}')
        # self.explain(**values)
        self.trigger(None)
        
        
       
        # selected_vars.observe(get_value_selected_vars,names='value',type='change')    
            
        
        def get_prefix(g):
            
            try:
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except:
                current_suffix = ''
            new_prefix = g['new']
            selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                    if any([variable.startswith(n)  for n in new_prefix]))
            # An exception is trigered but has no consequences     
            try:                      
                selected_vars.options = selected_prefix_var
            except: 
                ...
                
                
            # print('here')

            if current_suffix:
                new_selection   = [f'{n}{c}' for c in current_suffix for n in new_prefix
                                        if f'{n}{c}' in {s  for p,s in selected_prefix_var}]
                selected_vars.value  = new_selection 
                # print(f"{new_selection=}{current_suffix=}{g['old']=}")
            else:    
                selected_vars.value  = [selected_prefix_var[0][1]]
                
                   
        if len(self.prefix_dict): 
            selected_prefix = SelectMultiple(value=[select_prefix[0][1]], options=select_prefix, 
                                             layout=Layout(width='25%', height=self.select_height, font="monospace"),
                                       description='')
               
            selected_prefix.observe(get_prefix,names='value',type='change')
            select = HBox([selected_vars,selected_prefix])
            # print(f'{select_prefix=}')
            get_prefix({'new':select_prefix[0]})
        else: 
            select = VBox([selected_vars])
   
        options1 = HBox([diff]) if self.short >=2 else HBox([diff,legend])
        options2 = HBox([scale, showtype])
        if self.short:
            vui = [select, options1, i_smpl]
        else:
            vui = [select, options1, options2, i_smpl]  
        vui =  vui[:-1] if self.short >= 2 else vui  
        
        self.datawidget= VBox(vui+[self.out_widget])
        # show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
        #                                     'scale': scale, 'legend': legend})
        # print('klar til register')
        
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
            
        # clear_output()
        with self.mmodel.set_smpl(*smpl):

            if self.multi: 
                self.keep_wiz_fig = self.mmodel.keep_plot_multi(variabler, diff=ldiff, diffpct = diffpct, scale=scale, showtype=showtype,
                                                    legend=legend, dec=self.dec, vline=self.vline)
            else: 
                self.keep_wiz_figs = self.mmodel.keep_plot(variabler, diff=ldiff, diffpct = diffpct, 
                                                    scale=scale, showtype=showtype,
                                                    legend=legend, dec=self.dec, vline=self.vline)
                self.keep_wiz_fig = self.keep_wiz_figs[selected_vars[0]]   
                plt.close('all')
        return self.keep_wiz_figs        
        # print(f'Efter plot {self.mmodel.current_per=}')


    def  trigger(self,g):
        self.mmodel.current_per = copy(self.old_current_per)        

        values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'Triggerd:\n{values} \n Start trigger {self.mmodel.current_per=}\n')
        figs = self.explain(**values)
        if 0:
            xxx = fig_to_image(self.keep_wiz_fig,format='svg')
            self.out_widget.children  =  [widgets.HTML(xxx)]
        else:
            yyy = [widgets.HTML(fig_to_image(a_fig),width='100%',format = 'svg',layout=Layout(width='75%'))  
                     for key,a_fig in figs.items() ]
            self.out_widget.children = yyy
        # print(f'end  trigger {self.mmodel.current_per[0]=}')




def make_widget(widgetdef):
    '''
    '''
    
    widgettype,widgetdict  = widgetdef
    reswidget = globals()[widgettype+'widget'](widgetdict)
    return reswidget

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

