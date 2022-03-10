# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:26:13 2019

@author: hanseni
"""

import pandas as pd
import  ipywidgets as widgets  
from IPython.display import display, clear_output,Latex, Markdown
import matplotlib.pylab  as plt 
import seaborn as sns
import os
import webbrowser as wb
import sys
import re
import fnmatch 
import matplotlib.ticker as ticker



#import qgrid 


 
from  modelhelp import insertModelVar, finddec
import modelpattern as pt


def vis_alt3(dfs,model,title='Show variables',basename='Baseline',altname='Alternative',trans={},legend=True):
    ''' display tabbed widget with results from different dataframes, usuallly 2 but more can be shown
    
    dfs is a list of dataframes. They should be of same dimensionalities 
    
    '''
    avar = dfs[0].columns
    outlist =     [widgets.Output() for var in avar]
    outdiflist =  [widgets.Output() for var in avar]    
    deplist =     [widgets.Output() for var in avar]
    reslist =     [widgets.Output() for var in avar]
    attlist =     [widgets.Output() for var in avar]
    varouttablist =  [widgets.Tab(children = [out,outdif,att,dep,res]) 
                      for out,outdif,att,dep,res  in zip(outlist,outdiflist,attlist,deplist,reslist)]
    for var,varouttab in zip(avar,varouttablist):
        for i,tabtext in enumerate(['Level','Impact','Attribution','Dependencies','Results']):
            varouttab.set_title(i,tabtext)

    controllist = [widgets.VBox([varouttab]) for varouttab in varouttablist]
    tab = widgets.Tab(children = controllist)
    for i,(var,out) in enumerate(zip(avar,controllist)):
        tab.set_title(i, var) 
    
    def showvar(b):
        sel = b['new']
        out = outlist[sel]
        outdif = outdiflist[sel]
        dep = deplist[sel]
        res = reslist[sel]
        att = attlist[sel]
        var = avar[sel]
        with out:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
#            ax.spines['top'].set_visible(False)
            for i,df in enumerate(dfs):
                alt=i if len(dfs) >= 3 else ''
                data = df.loc[:,var]
                data.name = f'{basename}' if i ==0 else f'{altname}{alt}'
                data.plot(ax=ax,legend=legend,fontsize=14)
                x_pos = data.index[-1]
                if not legend:
                    if i == 0:
                        basevalue = data.values[-1]
                    else:
                        if (abs(data.values[-1]-basevalue) < 0.01 or 
                          (abs(basevalue) > 100. and (1.-abs(data.values[-1]/basevalue) < 0.008))):
                            if i == 1:
                                ax.text(x_pos, data.values[-1]  ,f' {basename} and {altname}{alt}',fontsize=14)
                        else:
                            ax.text(x_pos, data.values[-1]  ,f' {altname}{alt}',fontsize=14)
                            ax.text(x_pos, basevalue  ,f' {basename}',fontsize=14)
            plt.show(fig)
        with outdif:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
            for i,df in enumerate(dfs):
                if i == 0:
                    basedata = df.loc[:,var]
                    x_pos = data.index[-1]
                else:
                    data = df.loc[:,var]-basedata
                    data.plot(ax=ax,legend=False,fontsize=14)
                    x_pos = data.index[-1]
                    alt=i if len(dfs) >= 3 else ''
                    ax.text(x_pos, data.values[-1]  ,f' Impact of {altname}{alt}',fontsize=14)
            plt.show(fig)
        with dep:
            clear_output()
            model.draw(var,up=2,down=2,svg=1)
        with res:
            clear_output()
            out = model.get_values(var).T.rename(columns={'Base':basename,'Last':altname})
            out.style.set_caption(trans.get(var,var))
            print(trans.get(var,var))
            print(out)
        with att:
            clear_output()
            #model.smpl(N-20,N)
            if var in model.endogene:
                print(f'What explains the difference between the {basename} and the {altname} run ')
                print(model.allvar[var]['frml'])
                model.explain(var,up=0,dec=1,size=(9,12),svg=1,HR=0)
            else:
                print(f'{var} is exogeneous and attribution can not be performed')
            
    display(tab) 

    showvar({'new':0})
    tab.observe(showvar,'selected_index')
    return tab

def vis_alt4(dfs,model,title='Show variables',trans={},legend=True):
    ''' display tabbed widget with results from different dataframes, usuallly 2 but more can be shown
    
    dfs is a list of dataframes. They should be of same dimensionalities 
    
    '''
    avar = dfs[list(dfs.keys())[0]].columns
    outlist =     [widgets.Output() for var in avar]
    outdiflist =  [widgets.Output() for var in avar]    
    deplist =     [widgets.Output() for var in avar]
    reslist =     [widgets.Output() for var in avar]
    attlist =     [widgets.Output() for var in avar]
    varouttablist =  [widgets.Tab(children = [out,outdif,att,dep,res]) 
                      for out,outdif,att,dep,res  in zip(outlist,outdiflist,attlist,deplist,reslist)]
    for var,varouttab in zip(avar,varouttablist):
        for i,tabtext in enumerate(['Level','Impact','Attribution','Dependencies','Results']):
            varouttab.set_title(i,tabtext)

    controllist = [widgets.VBox([varouttab]) for varouttab in varouttablist]
    tab = widgets.Tab(children = controllist)
    for i,(var,out) in enumerate(zip(avar,controllist)):
        tab.set_title(i, var) 
    
    def showvar(b):
        sel = b['new']
        out = outlist[sel]
        outdif = outdiflist[sel]
        dep = deplist[sel]
        res = reslist[sel]
        att = attlist[sel]
        var = avar[sel]
        with out:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
#            ax.spines['top'].set_visible(False)
            for i,(k,df) in enumerate(dfs.items()):
                data = df.loc[:,var]
                data.name = k
                data.plot(ax=ax,legend=legend,fontsize=14)
                x_pos = data.index[-1]
                if not legend:
                    if i == 0:
                        basevalue = data.values[-1]
                    else:
                        if (abs(data.values[-1]-basevalue) < 0.01 or 
                          (abs(basevalue) > 100. and (1.-abs(data.values[-1]/basevalue) < 0.008))):
                            if i == 1:
                                ax.text(x_pos, data.values[-1]  ,f' {basename} and {altname}{alt}',fontsize=14)
                        else:
                            ax.text(x_pos, data.values[-1]  ,f' {altname}{alt}',fontsize=14)
                            ax.text(x_pos, basevalue  ,f' {basename}',fontsize=14)
            plt.show(fig)
        with outdif:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
            for i,(k,df) in enumerate(dfs.items()):
                if i == 0:
                    basedata = df.loc[:,var]
                    x_pos = data.index[-1]
                else:
                    data = df.loc[:,var]-basedata
                    data.name = f' Impact of: {k}'
                    data.plot(ax=ax,legend=legend,fontsize=14)
                    x_pos = data.index[-1]
                    if not legend:
                        ax.text(x_pos, data.values[-1]  ,f' Impact of: {k}',fontsize=14)
            plt.show(fig)
        with dep:
            clear_output()
            model.draw(var,up=2,down=2,svg=1)
        with res:
            clear_output()
            out = pd.concat([df.loc[:,var] for k,df in dfs.items()],axis=1)
            out.columns = [k for k in dfs.keys()]
            out.style.set_caption(trans.get(var,var))
            print(trans.get(var,var))
            print(out.to_string())
        with att:
            clear_output()
            #model.smpl(N-20,N)
            if var in model.endogene:
                print(f'What explains the difference between the {basename} and the {altname} run ')
                print(model.allvar[var]['frml'])
                model.explain(var,up=0,dec=1,size=(9,12),svg=1,HR=0)
            else:
                print(f'{var} is exogeneous and attribution can not be performed')
            
    display(tab) 

    showvar({'new':0})
    tab.observe(showvar,'selected_index')
    
class jup_keepviz() :
    ''' Class to vizualize a number of runs, primary in Jupyter 
    :dfs: A dict with runs {name :{'result' : df}}
    :title: A title 
    :trans: a translation of variable names to more redable names
    :legend: if legends has to be shown 
    
    '''



    def __init__(self,dfs,title='Show variables',trans={},legend=True,showfig=False):
        self.dfs = dfs
        self.title = title
        self.trans = trans
        self.legend = legend
        self.showfig = showfig
        
        return 
        
    def plot_level(self,var):
        fig,ax = plt.subplots(figsize=(10,6))
        ax.set_title(f'Level: {self.trans.get(var,var)}',fontsize=14)
        if  self.legend :
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
#            ax.spines['top'].set_visible(False)
        for i,(k,df) in enumerate(self.dfs.items()):
            data = df.loc[:,var]
            data.name = k
            data.plot(ax=ax,legend=self.legend,fontsize=14)
            dec=finddec(df)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,.{dec}f}'))

            x_pos = data.index[-1]
            if not self.legend:
                if i == 0:
                    basename = k
                    basevalue = data.values[-1]
                    ax.text(x_pos, basevalue  ,f' {basename}',fontsize=14)
                else:
                    if (abs(data.values[-1]-basevalue) < 0.01 or 
                      (abs(basevalue) > 100. and (1.-abs(data.values[-1]/basevalue) < 0.008))):
                        if i == 1:
                            ax.text(x_pos, data.values[-1]  ,f' {basename} and {k}',fontsize=14)
                        else: 
                            ax.text(x_pos, data.values[-1]  ,f' {k}',fontsize=14)
                            
                    else:
                        ax.text(x_pos, data.values[-1]  ,f' {k}',fontsize=14)
        return fig

    def plot_dif(self,var):
         fig,ax = plt.subplots(figsize=(10,6))
         ax.set_title(f'{self.trans.get(var,var)}',fontsize=14)
         if  self.legend :
            ax.spines['right'].set_visible(True)
         else:
                ax.spines['right'].set_visible(False)
         for i,(k,df) in enumerate(self.dfs.items()):
             if i == 0:
                 basedata = df.loc[:,var]
             else:
                 data = df.loc[:,var]-basedata
                 data.name = f' Impact of: {k}'
                 dec=finddec(data)

                 data.plot(ax=ax,legend=self.legend,fontsize=14)
                 ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,.{dec}f}'))
                 
                 x_pos = basedata.index[-1]
                 if not self.legend:
                     ax.text(x_pos, data.values[-1]  ,f' Impact of: {k}',fontsize=14)
         return fig 

    def formatnumber(self,var,out):
        if out.abs().max(axis=1).max() >= 30.0:
            return lambda number : f'{number:,.0f}'
        else:
            return lambda number : f'{number:,.6f}'
 
    
class jupviz(jup_keepviz) :
    '''
    Class to vizualize a number of experiments in an tabbed ipywidget in a jupyter notebook
    ''' 
             
    def __call__(self):
        self.vis()
   
    def vis(self):
        ''' display tabbed widget with results from different dataframes, usuallly 2 but more can be shown
        
        dfs is a list of dataframes. They should be of same dimensionalities 
        
        '''
        self.basekey = [key for key in  self.dfs.keys()][0] 

        wexperiment_name = widgets.Text(value=self.basekey,placeholder='Type something',description='Name of these experiments:',
                            layout={'width':'55%'},style={'description_width':'45%'})
        
        wsavefig   = widgets.Button(description="Save figure")
        wbut  = widgets.HBox([wsavefig,wexperiment_name])
        
        wfile_folder = widgets.Text(value = 'test2',placeholder='Type something',description='Figure saved in:',
                            layout={'width':'65%'},style={'description_width':'25%'},visible=False,disabled=True)
        wfile_folder.layout.visibility = 'hidden'
        
        wopen = widgets.Button(description="Open the saved figures")
        wopen.layout.visibility = 'hidden'
        
        winputstring = widgets.VBox([wfile_folder,wopen])
          
        
        avar = self.dfs[list(self.dfs.keys())[0]].columns
        outlist =     [widgets.Output() for var in avar]
        outdiflist =  [widgets.Output() for var in avar]    
        reslist =     [widgets.Output() for var in avar]
        resdiflist =  [widgets.Output() for var in avar]
        varouttablist =  [widgets.Tab(children = [out,outdif,res,resdif]) 
                          for out,outdif,res,resdif  in zip(outlist,outdiflist,reslist,resdiflist)]
        for var,varouttab in zip(avar,varouttablist):
            for i,tabtext in enumerate(['Level','Impact','Level data','Impact data']):
                varouttab.set_title(i,tabtext)
    
        controllist = [widgets.VBox([varouttab]) for varouttab in varouttablist]
        tab = widgets.Tab(children = controllist)
        
        for i,(var,out) in enumerate(zip(avar,controllist)):
            tab.set_title(i, var) 
        
        def showvar(b):
            wfile_folder.layout.visibility = 'hidden'
            wopen.layout.visibility = 'hidden'

            sel = b['new']
            out    = outlist[sel]
            outdif = outdiflist[sel]
            res    = reslist[sel]
            resdif = resdiflist[sel]
            var    = avar[sel]
            self.selected = sel
            self.selected_var = var 
            with out:
                clear_output()
                self.fig_level = self.plot_level(var)
                plt.show(self.fig_level)
            with outdif:
                clear_output()
                self.fig_dif = self.plot_dif(var)
                plt.show(self.fig_dif)
            with res:
                clear_output()
                out = pd.concat([df.loc[:,var] for k,df in self.dfs.items()],axis=1)
                out.columns = [k for k in self.dfs.keys()]
                print(self.trans.get(var,var))
                print(out.to_string(float_format= self.formatnumber(var,out)))
            with resdif:
                clear_output()
                try:
                    baseline = self.dfs[[k for i,k in enumerate(self.dfs.keys()) if i == 0][0]].loc[:,[var]]
                    out = pd.concat([df.loc[:,[var]]-baseline for i,(k,df) in enumerate(self.dfs.items()) if i >= 1],axis=1)
                    out.columns = [k for i,k in enumerate(self.dfs.keys()) if i >= 1]
                    # out.style.set_caption(self.trans.get(var,var))
                    print(self.trans.get(var,var))
                    print(out.to_string(float_format= self.formatnumber(var,out)))
                except:
                    print('No Data')
     
        outtab = widgets.VBox([tab,wbut,winputstring])
        def savefig(s):
            self.graph_folde=os.path.join(os.getcwd(),'experiments',wexperiment_name.value,'graph')
            self.figlocation        = os.path.join(self.graph_folde,f'{self.selected_var}') 
            self.figlocation_impact = os.path.join(self.graph_folde,f'{self.selected_var}-impact') 
            wfile_folder.layout.visibility = 'visible'
            if self.showfig:
                wopen.layout.visibility = 'visible'
            else:
                wopen.layout.visibility = 'hidden'

            try:
                if not os.path.exists(self.graph_folde):
                    os.makedirs(self.graph_folde)
            except: 
                wfile_folder.value = f"Can't create  {self.graph_folde}"
                return 

            try:       
                self.fig_level.savefig(self.figlocation+'.svg')
                self.fig_dif.savefig(self.figlocation_impact+'.svg')
                self.fig_level.savefig(self.figlocation+'.pdf')
                self.fig_dif.savefig(self.figlocation_impact+'.pdf')
                wfile_folder.value = self.graph_folde
            except: 
                wfile_folder.value = f"Can't write to {self.graph_folde}, remember to close"

        def openfig_svg(s):
            try:
                wb.open(self.figlocation_impact+'.svg',new=2)
                wb.open(self.figlocation+'.svg',new=2)
                # os.system(self.figlocation)
                # os.system(self.figlocation_impact)
            except: 
                wfile_folder.layout.visibility = 'visible'
                wfile_folder.value = f"Can't open to {self.figlocation}.svg Try to download the file"
                
        def openfig_pdf(s):
            try:
                wb.open(self.figlocation_impact+'.pdf')
                wb.open(self.figlocation+'.pdf')
            except: 
                wfile_folder.layout.visibility = 'visible'
                wfile_folder.value = f"Can't open to {self.figlocation}.svg Try to download the file"
                


        if self.showfig : wopen.on_click(openfig_svg)
            
        wsavefig.on_click(savefig)
      
        display(outtab) 
    
        showvar({'new':0})
        tab.observe(showvar,'selected_index')
        
            
     
# Define data extraction
        
        
def get_alt(mmodel,pat,onlyendo=False):
    ''' Retrieves variables matching pat from a model '''
    varnames = mmodel.vlist(pat)
    modelvar = mmodel.endogene if onlyendo else mmodel.exogene | mmodel.endogene 
    modelvarnames = list(dict.fromkeys([v for v in varnames if v in modelvar])) # to awoid dublicate names
    per = mmodel.current_per
    return [mmodel.basedf.loc[per,modelvarnames],mmodel.lastdf.loc[per,modelvarnames]]

def get_alt_dic(mmodel,pat,dfs,onlyendo=False):
    ''' Retrieves variables matching pat from a model '''
    varnames = mmodel.vlist(pat)
    modelvar = mmodel.endogene if onlyendo else mmodel.exogene | mmodel.endogene 
    modelvarnames = list(dict.fromkeys([v for v in varnames if v in modelvar])) # to awoid dublicate names
    per = mmodel.current_per
    return {k: df_dic['results'].loc[per,modelvarnames] for k,df_dic in dfs.items()}






def inputwidget(model,basedf,slidedef={},radiodef=[],checkdef=[],modelopt={},varpat='RFF XGDPN RFFMIN GFSRPN DMPTRSH XXIBDUMMY'
                 ,showout=1,trans={},base1name='',alt1name='',go_now=True,showvar=False):
    '''Creates an input widgets for updating variables 
    
    :df: Baseline dataframe 
    :slidedef: dict with definition of variables to be updated by slider
    :radiodef: dict of lists. each at first level defines a collection of radiobuttoms
               second level defines the text for each leved and the variable to set or reset to 0
    :varpat: the variables to show in the output widget
    :showout: 1 if the output widget is to be called '''
    
    lradiodef= len(radiodef)
    lslidedef = len(slidedef)
    lcheckdef = len(checkdef)
    basename = base1name if base1name else 'Baseline' 
    altname = alt1name if alt1name else 'Alternative' 
    
    if lradiodef: 
        wradiolist = [widgets.RadioButtons(options=[i for i,j in cont],description=des,layout={'width':'70%'},
                                           style={'description_width':'37%'}) for des,cont in radiodef.items()]
        if len(wradiolist) <=2:
            wradio = widgets.HBox(wradiolist)
        else: 
            wradio = widgets.VBox(wradiolist)

            

# define slidesets 
    if lslidedef:     
        wexp  = widgets.Label(value="Input new parameter ",layout={'width':'52%'})
        walt  = widgets.Label(value=f'{altname}',layout={'width':'10%'})
        wbas  = widgets.Label(value=f'{basename}',layout={'width':'20%'})
        whead = widgets.HBox([wexp,walt,wbas])
        
        wset  = [widgets.FloatSlider(description=des,
                                    min=cont['min'],max=cont['max'],value=cont['value'],step=cont.get('step',0.01),
                                    layout={'width':'60%'},style={'description_width':'40%'},readout_format = f":<,.{cont.get('dec',2)}f")
                 for des,cont in slidedef.items()]
        formattest = ':>.2f'
        waltval= [widgets.Label(value=f"{cont['value']:<,.{cont.get('dec',2)}f}",layout={'width':'10%'})
                  for des,cont  in slidedef.items()]
        wslide = [widgets.HBox([s,v]) for s,v in zip(wset,waltval)]
       
# cheklist  
    if lcheckdef:         
        wchecklist = [widgets.Checkbox(description=des,value=val)   for des,var,val in checkdef]
        wcheck  = widgets.HBox(wchecklist)   

# some buttons and text     
    wname = widgets.Text(value=basename,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
    wpat = widgets.Text(value=varpat,placeholder='Type something',description='Output variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
    if showvar:
        wpat.layout.visibility = 'visible'
    else:
        wpat.layout.visibility = 'hidden'
        

    winputstring = widgets.HBox([wname,wpat])
    
    wgo   = widgets.Button(description="Run scenario")
    wreset   = widgets.Button(description="Reset to start")
    wzero   = widgets.Button(description="Set all to 0")
    wsetbas   = widgets.Button(description="Use as baseline")
    wbut  = widgets.HBox([wgo,wreset,wzero,wsetbas])
    
    wvar = [whead]+wslide if lslidedef else []
    if lradiodef: wvar = wvar + [wradio]
    if lcheckdef: wvar = wvar + [wcheck]
        
    w     = widgets.VBox(wvar+[winputstring] +[wbut])

    # This function is run when the button is clecked 
    firstrun = True
    model.rundic = {}
    
    def run(b):
        nonlocal firstrun
        nonlocal altname
        # mulstart       = model.basedf.copy()
        mulstart       = insertModelVar(basedf.copy(deep=True),model)
        # model.smpl(df=mulstart)
        
        # First update from the sliders 
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                op = cont.get('op','=')
                var = cont['var']
                for var in cont['var'].split():
                    if  op == '+':
                        mulstart.loc[model.current_per,var]    =  mulstart.loc[model.current_per,var] + wset[i].value
                    elif  op == '%':
                        mulstart.loc[model.current_per,var]    =  mulstart.loc[model.current_per,var] * (1+wset[i].value/100)
                    elif op == '+impulse':    
                        mulstart.loc[model.current_per[0],var] =  mulstart.loc[model.current_per[0],var] + wset[i].value
                    elif op == '=start-':   
                        startindex = mulstart.index.get_loc(model.current_per[0])
                        varloc = mulstart.columns.get_loc(var)
                        mulstart.iloc[:startindex,varloc] =  wset[i].value
                    elif op == '=':    
                        mulstart.loc[model.current_per,var]    =   wset[i].value
                    elif op == '=impulse':    
                        mulstart.loc[model.current_per[0],var] =   wset[i].value
                    else:
                        print(f'Wrong operator in {cont}.\nNot updated')
                        assert 1==3,'wRONG OPERATOR'
                
        # now  update from the radio buttons 
        if lradiodef:
            for wradio,(des,cont) in zip(wradiolist,radiodef.items()):
                # print(des,wradio.value,wradio.index,cont[wradio.index])
                for v in cont:
                    mulstart.loc[model.current_per,v[1]] = 0.0
                mulstart.loc[model.current_per,cont[wradio.index][1]] = 1.0  
 
        if lcheckdef:           
            for box,(des,var,_) in zip(wchecklist,checkdef):
                mulstart.loc[model.current_per,var] = 1.0 * box.value

        #with out:
        clear_output()
        mul = model(mulstart,**modelopt)
        # model.mulstart=mulstart


        clear_output()
        display(w)
        #_ = mfrbus['XGDPN RFF RFFMIN GFSRPN'].dif.rename(trans).plot(colrow=1,sharey=0)
        
        
        if firstrun:
            model.experiment_results = {}
            model.experiment_results[basename] = {'results':model.lastdf.copy()}
            firstrun = False
            wname.value = f'{altname}'
        else:
            altname = wname.value
            walt.value = f'{altname}'
            model.experiment_results[altname] = {'results':model.lastdf.copy()}

        
        if showout:
            varpat_this =  wpat.value
            resdic = get_alt_dic(model,varpat_this,model.experiment_results)
            a = jupviz(resdic,trans=trans)()
        else:  
            a = vis_alt4(get_alt_dic(model,wpat.value,model.experiment_results),model,trans=trans)

    def reset(b):

        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                wset[i].value  =   cont['value']
            
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
            
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue
                
    def zeroset(b):
        basename = base1name if base1name else 'Baseline' 
        walt.value = f'{altname}'
        wbas.value = f'{basename}'

        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                wset[i].value  =   type(cont['value'])(0.0)
            
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
            
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue


                

    def setbas(b):
        nonlocal basename
        nonlocal firstrun 
        model.basedf = model.lastdf.copy(deep=True)
        basename = wname.value
        walt.value = f'{altname}'
        wbas.value = f'{basename}'
        wname.value = f'{basename}'
        model.rundic = {}
        firstrun = True 
        
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                waltval[i].value=  f"{cont['value']:<,.{cont.get('dec',2)}f}"
                
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
                
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue


    # Assign the function to the button  
    wgo.on_click(run)
    wreset.on_click(reset)
    wzero.on_click(zeroset)
    wsetbas.on_click(setbas)
    # out = widgets.Output()
    
    
    if go_now:
        run(None)
    return w





def get_att_gui(totdif,var='FY',spat = '*',desdic={},use='level',kind='bar',perselect='per',ysize=10):
    '''Creates a jupyter ipywidget to display model level 
    attributions ''' 
    xvar=var
    # print(f'{var=}  {xvar}')
    def show_all2(Variable,Periode,Save,Use):
         # global fig1,fig2
         fig1 = totdif.totexplain(pat=Variable,top=0.87,use=Use,axvline=Periode,kind=kind)
         fig2 = totdif.totexplain(pat=Variable,vtype='per',per = Periode,top=0.85,use=Use,ysize=ysize) 
         if Save:
            fig1.savefig(f'Attribution-{Variable}-{use}.pdf')
            fig2.savefig(f'Attribution-{Variable}-{Periode}-{use}.pdf')
            print(f'Attribution-{Variable}-{use}.pdf and Attribution-{Variable}-{Periode}-{use}.pdf aare saved' )
    show = widgets.interactive(show_all2,
              Variable = widgets.Dropdown(options = sorted(totdif.model.endogene),value=xvar),
              Periode  = widgets.Dropdown(options = totdif.model.current_per) if perselect=='per'
                                          else [f'{t.year}-{t.month}-{t.day}' for t in totdif.model.current_per],
              Use = widgets.RadioButtons(options= ['level', 'growth'],description='Use'),
              Save = widgets.Checkbox(description='Save the charts',value=False)
              )
    return show 

def get_att_gui2(totdif,var='RP',spat = '*',desdic={},use='level',kind='bar'):
    '''Creates a jupyter ipywidget to display model level 
    attributions for datily dates ''' 
    def show_all2(Variable,Periode,Save,Use):
         global fig1,fig2
         fig1 = totdif.explain_all(pat=Variable,top=0.87,use=Use,kind='line',stacked=False,axvline=Periode,threshold=0.01,resample='m')
         fig2 = totdif.totexplain(pat=Variable,vtype='per',per = Periode,top=0.85,use=Use,threshold=0.01) 
         if Save:
            fig1.savefig(f'Attribution-{Variable}-{use}.pdf')
            fig2.savefig(f'Attribution-{Variable}-{Periode}-{use}.pdf')
            print(f'Attribution-{Variable}-{use}.pdf and Attribution-{Variable}-{Periode}-{use}.pdf aare saved' )

    show = widgets.interactive(show_all2,
              Variable = widgets.Dropdown(options = sorted(totdif.model.endogene),value=var),
              Periode  = widgets.Dropdown(options = [f'{t.year}-{t.month}-{t.day}' for t in mddm.totdekomp.model.current_per]),
              Use = widgets.RadioButtons(options= ['level', 'growth'],description='Use'),
              Save = widgets.Checkbox(description='Save the charts',value=False)
              )
    return show 

def vtol(var):
    ''' replaces special characters in variable name to latex'''
    return var.replace(r'_',r'\_').replace('{','\{').replace('}','\}')


def an_expression_to_latex(exp,funks=[]):
    ''' Returns a latex string from a list of terms (defined in the modelpattern module)
    
    funks is a list of localy defines functions
    '''
    def t_to_latex(t):
        if t.var:
            var =vtol(t.var)
            if t.lag:
                return f'{var}_{{t{t.lag}}}'
            else:
                return f'{var}_t'
        elif t.number:
            return t.number
        else:
            if t.op == '*':
                op=r'\times'
            elif t.op == '$':
                op=''
            else:
                op = t.op
            return op

    return ' '.join([t_to_latex(t) for t in pt.udtryk_parse(exp,funks=funks)])
    
def expressions_to_latex(expressions,funks=[],allign = True, disp = False):
    ''' Returns a latex string from a list of a list of terms
    
    :funks: a list of local functions in the model 
    :allign: the first = is enclosed in & for alligning several equations in latex
    :disp: the result is displayed 
    '''
    
    texp = expressions if type(expressions) == list else [expressions ]
    latex_list = [an_expression_to_latex(exp,funks=funks) for exp in  texp]
    if allign: 
        latex_list = [l.replace('=',' & = & ',1) for l in latex_list]
    latex_out = r'\\'.join(latex_list) 
    latex_out = r'\begin{eqnarray*}'+latex_out+r'\end{eqnarray*}'
    
    if disp:
        display(Latex('$'+latex_out+'$'))
    else:
        return latex_out
    
def frml_as_latex(frml_in,funks=[],allign= True, name=True,disp=True,linespace = False):
    ''' Display formula
    
    :funks: local functions 
    :allign: allign =
    :name: also display the frml name
    '''
    frmls =    frml_in if type(frml_in) == list else [frml_in ]
    out = []
    for i,frml in enumerate(frmls):
        a,fr,n,udtryk= pt.split_frml(frml)
        out_udtryk = an_expression_to_latex(udtryk,funks = funks)
        out_frmlname = vtol(n) if name and n != '<>' else ''
        if allign:
            out_udtryk = out_udtryk.replace('=',' & = & ',1)+(r'\\[1em]' if linespace else '')
        out.append(f'{out_frmlname} {out_udtryk}')
    latex_out = r'\\'.join(out)
    if allign:
        latex_out = r'\begin{eqnarray*}'+latex_out+r'\end{eqnarray*}'
    
    if disp:
        display(Latex('$'+latex_out+'$'))
    else: 
        return latex_out

def get_frml_latex(model,pat='*',name=True):
    
    variabler =  [var for  p in pat.split() for var in fnmatch.filter(model.nrorder,p)]
    frmls = [model.allvar[var]['frml'] for var in variabler if not model.allvar[var]['dropfrml']]
    _=frml_as_latex(frmls,funks=model.funks,name=name)
    