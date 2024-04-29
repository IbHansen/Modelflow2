# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To define Jupyter widgets  show variables. 
@author: Ib 
"""

import pandas as pd
import  ipywidgets as widgets  

from  ipysheet import sheet, cell, current 
from ipysheet.pandas_loader import from_dataframe, to_dataframe

from IPython.display import display, clear_output,Latex, Markdown
from dataclasses import dataclass,field
import matplotlib.pylab  as plt 



from modelclass import insertModelVar
from modelclass import model 
from modeljupyter import jupviz


@dataclass
class basewidget:
    ''' basis for widget updating in jupyter'''
    
    
    datachildren : list = field(default_factory=list) # list of children widgets 
    
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        for w in self.datachildren:
            w.update_df(df,current_per)
    
    
@dataclass
class tabwidget:
    '''A widget to create tab or acordium contaners'''
    
    tabdefdict : dict # =  field(default_factory = lambda: ({}))
    tab : bool = True 
    selected_index :any = None
    
    def __post_init__(self):
        
        thiswidget = widgets.Tab if self.tab else widgets.Accordion
        self.datachildren = [tabcontent 
                             for tabcontent  in self.tabdefdict.values()]
        self.datawidget = thiswidget([child.datawidget for child in self.datachildren], 
                                     selected_index = self.selected_index,                               
                                     layout={'height': 'max-content'})  
         
        for i,key in enumerate(self.tabdefdict.keys()):
           self.datawidget.set_title(i,key)
           
           
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        for w in self.datachildren:
            w.update_df(df,current_per)
            
            
    def reset(self,g):
        ''' will reset  container widgets'''
        for w in self.datachildren:
            w.reset(g)


@dataclass
class sheetwidget:
    ''' class defining a widget which updates from a sheet '''
    df_var         : pd.DataFrame = field(default_factory=pd.DataFrame)        # definition 
    trans          : any = lambda x : x      # renaming of variables 
    transpose      : bool = False            # orientation of dataframe 
    expname      : str =  "Carbon tax rate, US$ per tonn "
   
    
    def __post_init__(self):
        ...
        
        self.wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})

    
        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = self.newnamedf.T if self.transpose else newnamedf
        
        self.wsheet = sheet(from_dataframe(self.org_df_var),column_resizing=True)
        self.org_values = [c.value for c in self.wsheet.cells]
        self.datawidget=widgets.VBox([self.wexp,self.wsheet]) if len(self.expname) else self.wsheet

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
        


@dataclass
class slidewidget:
    ''' class defefining a widget with lines of slides '''
    slidedef     : dict         # definition 
    altname      : str = 'Alternative'
    basename     : str =  'Baseline'
    expname      : str =  "Carbon tax rate, US$ per tonn "
    def __post_init__(self):
        ...
        # plt.ioff()
        wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
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
                continuous_update=False
                )
             for des,cont in self.slidedef.items()]
        
            
        for w in self.wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:>,.{cont.get('dec',2)}f}",
                layout=widgets.Layout(display="flex", justify_content="center", width="10%", border="hide"))
            for des,cont  in self.slidedef.items()
            
            ]
        self.wslide = [widgets.HBox([s,v]) for s,v in zip(self.wset,waltval)]
        self.slidewidget =  widgets.VBox([whead] + self.wslide)
        self.datawidget =  widgets.VBox([whead] + self.wslide)
      
        # define the result object 
        self.current_values = {des:
             {key : v.split() if key=='var' else v for key,v in cont.items() if key in {'value','var','op'}} 
             for des,cont in self.slidedef.items()} 
            
    def reset(self,g): 
        for i,(des,cont) in enumerate(self.slidedef.items()):
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

@dataclass
class sumslidewidget:
    ''' class defefining a widget with lines of slides '''
    slidedef     : dict         # definition
    maxsum          : any  = None 
    altname      : str = 'Alternative'
    basename     : str =  'Baseline'
    expname      : str =  "Carbon tax rate, US$ per tonn "
    def __post_init__(self):
        ...
        
        self.first = list(self.slidedef.keys())[:-1]     
        self.lastdes = list(self.slidedef.keys())[-1]     

        wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
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
             for des,cont in self.slidedef.items()]
        
             
            
        for w in self.wset: 
            w.observe(self.set_slide_value,names='value',type='change')
        
        waltval= [widgets.Label(
                value=f"{cont['value']:>,.{cont.get('dec',2)}f}",
                layout=widgets.Layout(display="flex", justify_content="center", width="10%", border="hide"))
            for des,cont  in self.slidedef.items()
            
            ]
        self.wslide = [widgets.HBox([s,v]) for s,v in zip(self.wset,waltval)]
        self.slidewidget =  widgets.VBox([whead] + self.wslide)
        self.datawidget =  widgets.VBox([whead] + self.wslide)
      
        # define the result object 
        self.current_values = {des:
             {key : v.split() if key=='var' else v for key,v in cont.items() if key in {'value','var','op','min','max'}} 
             for des,cont in self.slidedef.items()} 
            
    def reset(self,g): 
        for i,(des,cont) in enumerate(self.slidedef.items()):
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
        # print(self.current_values)
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
                

                      
@dataclass
class updatewidget:   
    ''' class to input and run a model'''
    
    mmodel : any     # a model 
    a_datawidget : any # a tab to update from  
    basename : str ='Business as usual'
    keeppat   : str = '*'
    varpat    : str ='*'
    showvarpat  : bool = True 
    exodif   : pd.DataFrame = field(default_factory=pd.DataFrame)        # definition 
    lwrun    : bool = True
    lwupdate : bool = False
    lwreset  :  bool = True
    lwsetbas  :  bool = True
    lwshow    :bool = True 
    outputwidget : str  = 'jupviz'
    prefix_dict : dict = field(default_factory=dict)
    display_first :any = None 
    vline  : list = field(default_factory=list)
    relativ_start : int = 0 
    short :bool = False 
    legend :bool = False
    
    def __post_init__(self):
        self.baseline = self.mmodel.basedf.copy()
        wrun   = widgets.Button(description="Run scenario")
        wrun.on_click(self.run)
        
        wupdate   = widgets.Button(description="Update the dataset ")
        wupdate.on_click(self.update)
        
        wreset   = widgets.Button(description="Reset to start")
        wreset.on_click(self.reset)

        wshow   = widgets.Button(description="Show results")
        wshow.on_click(self.show)
        
        wsetbas   = widgets.Button(description="Use as baseline")
        wsetbas.on_click(self.setbasis)
        self.experiment = 0 
        
        lbut = []
        
        if self.lwrun: lbut.append(wrun)
        if self.lwshow:  lbut.append(wshow)
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
       
        self.wtotal = widgets.VBox([self.a_datawidget.datawidget,winputstring,wbut])
    
        self.mmodel.keep_solutions = {}
        self.mmodel.keep_solutions = {self.wname.value : self.baseline}
        self.mmodel.keep_exodif = {}
        
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'

    
    def update(self,g):
        self.thisexperiment = self.baseline.copy()
        self.a_datawidget.update_df(self.thisexperiment,self.mmodel.current_per)
        self.exodif = self.mmodel.exodif(self.baseline,self.thisexperiment)


    def show(self,g=None):
        if self.outputwidget == 'jupviz':
            clear_output()
            display(self.wtotal)

            displaydict =  {k :v.loc[self.mmodel.current_per,self.wpat.value.split()] 
                            for k,v in self.mmodel.keep_solutions.items()}
            jupviz(displaydict,legend=0)()
            
        elif self.outputwidget == 'keep_viz':

            selectfrom = [v for v in self.mmodel.vlist(self.wpat.value) if v in 
                          set(list(self.mmodel.keep_solutions.values())[0].columns)]
            clear_output()
            display(self.wtotal)
            plt.close('all')
            _ = self.mmodel.keep_viz(pat=selectfrom[0],selectfrom=selectfrom,vline=self.vline)
            
        elif self.outputwidget == 'keep_viz_prefix':

            selectfrom = [v for v in self.mmodel.vlist(self.wpat.value) if v in 
                          set(list(self.mmodel.keep_solutions.values())[0].columns)]
            clear_output()
            if self.display_first:
                display(self.display_first)
            display(self.wtotal)
            plt.close('all')
            with self.mmodel.set_smpl_relative(self.relativ_start,0):
                _ = self.mmodel.keep_viz_prefix(pat=selectfrom[0],
                        selectfrom=selectfrom,prefix_dict=self.prefix_dict,vline=self.vline,short=self.short,legend=self.legend)
            
        
    def run(self,g):
        clear_output(True)
        display(self.wtotal)
        self.update(g)
        self.mmodel(self.thisexperiment,progressbar=1,keep = self.wname.value,                    
                keep_variables = self.keeppat)
        self.mmodel.keep_exodif[self.wname.value] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True
        self.current_experiment = self.wname.value
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        self.show(g)
        
    def setbasis(self,g):
        clear_output(True)
        display(self.wtotal)
        self.mmodel.keep_solutions={self.current_experiment:self.mmodel.keep_solutions[self.current_experiment]}
        
        self.mmodel.keep_exodif[self.current_experiment] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True

        
        
         
    def reset(self,g):
        self.a_datawidget.reset(g) 

def fig_to_image(figs,format='svg'):
    from io import StringIO
    f = StringIO()
    figs.savefig(f,format=format)
    f.seek(0)
    image= f.read()
    return image
  
@dataclass
class htmlwidget_df:
    ''' class displays a dataframe in a html widget '''
    
    mmodel : any     # a model   
    df_var         : pd.DataFrame = field(default_factory=pd.DataFrame)        # definition 
    trans          : any = lambda x : x      # renaming of variables 
    transpose      : bool = False            # orientation of dataframe 
    expname      : str =  ""
    percent      : bool = False 
    style        : any = ''
   
    
    def __post_init__(self):
        ...
        
        self.wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
   
    
        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = newnamedf.T if self.transpose else newnamedf
        if 0:
            image = self.mmodel.ibsstyle(self.org_df_var,percent = self.percent).to_html()
        else: 
            style_html = """
<style>
table, th, td {
    border: none;     
    border-collapse: collapse;
    padding: 10px;  # Adjust padding as needed
    text-align: left;
}
</style>
"""         
            if self.style:
                image_html = self.style(self.org_df_var).to_html()
            else:
                image_html = self.mmodel.ibsstyle(self.org_df_var, percent=self.percent).to_html()
            image = f"{style_html}{image_html}"

            self.whtml = widgets.HTML(image)
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
        # print(f'Create {self.expname}')
        
        self.wexp  = widgets.Label(value = self.expname,layout={'width':'54%'})
   
        image = fig_to_image(self.figs,format=self.format)
        self.whtml = widgets.HTML(image)
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
        # from IPython import get_ipython
        # get_ipython().magic('matplotlib notebook') 
        # print(plt.get_backend())
        # print('hej haj')
        # plt.close('all')

        # print(f'{self.show_on=}')
        # plt.ioff() 
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
                 htmlwidget_fig(value['df'].rename().plot(top=1.0,title=''),expname='')               
                              for key,value in self.out_dict.items() }
                
       
        tabnew =  {key: tabwidget({'Charts':self.out_to_figs[key],'Data':self.out_to_data[key]},selected_index=0) for key in self.out_to_figs.keys()}
        # print('tabnew created')
        exodif = self.mmodel.exodif()
        exonames = exodif.columns        
        exoindex = exodif.index
        
        
        if 1:
            
            if len(exonames):
                exoindexstart = max(self.mmodel.lastdf.index.get_loc(exodif.index[0]),self.mmodel.lastdf.index.get_loc(self.mmodel.current_per[0]))
                exoindexend   = min(self.mmodel.lastdf.index.get_loc(exodif.index[-1]),self.mmodel.lastdf.index.get_loc(self.mmodel.current_per[-1]))
                exoindexnew = self.mmodel.lastdf.index[exoindexstart:exoindexend]
                rows,cols  =exodif.loc[exoindexnew,:].shape
                exosize = rows*cols 

                if exosize < 2000 : 
                    exobase = self.mmodel.basedf.loc[exoindexnew,exonames]
                    exolast = self.mmodel.lastdf.loc[exoindexnew,exonames]
                    
                    out_exodif = htmlwidget_df(self.mmodel,exodif.loc[exoindexnew,:].T)
                    out_exobase = htmlwidget_df(self.mmodel,exobase.T)
                    out_exolast = htmlwidget_df(self.mmodel,exolast.T)
                else: 
                    out_exodif = htmlwidget_label(expname = 'To many to display ')
                    out_exobase =  htmlwidget_label(expname = 'To many to display ')
                    out_exolast =  htmlwidget_label(expname = 'To many to display ')
            else: 
                out_exodif = htmlwidget_label(expname = 'No difference in exogenous')
                out_exobase =  htmlwidget_label(expname = 'No difference in exogenous')
                out_exolast =  htmlwidget_label(expname = 'No difference in exogenous')
                
            
            out_tab_info = tabwidget({'Delta exogenous':out_exodif,
                                      'Baseline exogenous':out_exobase,
                                      'Alt. exogenous':out_exolast},selected_index=0)
            
            tabnew['Exo info'] = out_tab_info
            # print('exo inf  created')

        this =   tabwidget(tabnew,selected_index=0)
        # print('this created ')   
            
        
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
    

if __name__ == '__main__':
    # if 'masia' not in locals(): 
    #     print('loading model')
    #     masia,baseline = model.modelload('C:/wb Dawn/UNESCAP-/Asia/Asia.pcim',run=0,silent=1)    
    # test = slidewidget(masia,{})
    ...
     