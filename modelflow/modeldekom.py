# -*- coding: utf-8 -*-
"""
Module for making attribution analysis of a model. 

The main function is attribution 

Created on Wed May 31 08:50:51 2017

@author: hanseni

"""


import pandas as pd
import fnmatch 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.dates as mdates

import numpy
import ipywidgets as ip
import pdb



from modelhelp import cutout
import modelclass as mc 
import modeldekom as mk 
import modelvis as mv


idx= pd.IndexSlice
    
def attribution(model,experiments,start='',end='',save='',maxexp=10000,showtime=False,
                summaryvar=['*']
                ,silent=False,msilent=True,type='level'):
    """ Calculates an attribution analysis on a model 
    accepts a dictionary with experiments. the key is experiment name, the value is a list 
    of variables which has to be reset to the values in the baseline dataframe. """  
    summaryout = model.vlist(summaryvar)
    adverseny = model.lastdf
    base = model.basedf
    if type == 'level':
        adverse0=adverseny[summaryout].loc[start:end,:].copy() 
    elif type ==  'growth':
        adverse0=adverseny[summaryout].pct_change().loc[start:end,:].copy() * 100.
    ret={}
    modelsave = model.save  # save the state of model.save 
    model.save = False      # no need to save the experiments in each run 
    with model.timer('Total dekomp',showtime):
        for i,(e,var) in enumerate(experiments.items()):
            if i >= maxexp : break     # when we are testing 
            oldvar=adverseny[var].copy()
            if not silent:
                print(i,'Experiment :',e,'\n','Touching: \n', var)
            adverseny[var] = base[var]
            tempdf = model(adverseny   ,start,end,samedata=True,
                silent=msilent)[summaryout]
            adverseny[var] = oldvar
            
            if type == 'level':
                ret[e] = tempdf[summaryout].loc[start:end,:]
            elif type ==  'growth':
                ret[e]  = tempdf.pct_change().loc[start:end,:] * 100.

        difret = {e : adverse0-ret[e]           for e in ret}
    
    df = pd.concat([difret[v] for v in difret],keys=difret.keys()).T
    if save:
         df.to_pickle('data\\' +save +r'.pc')    
         
    model.save = modelsave # restore the state of model.save 
    return df

def attribution_new(model,experiments,start='',end='',save='',maxexp=10000,showtime=False,
                summaryvar=['*']
                ,silent=False,msilent=True,type='level'):
    """ Calculates an attribution analysis on a model 
    accepts a dictionary with experiments. the key is experiment name, the value is a list 
    of variables which has to be reset to the values in the baseline dataframe. """  
    summaryout = model.vlist(summaryvar)
    adverseny = model.lastdf
    base = model.basedf
    adverse0_level  = adverseny[summaryout].loc[start:end,:].copy() 
    adverse0_growth = adverseny[summaryout].pct_change().loc[start:end,:].copy() * 100.
    ret_level  = {}
    ret_growth = {}
    modelsave = model.save  # save the state of model.save 
    model.save = False      # no need to save the experiments in each run 
    with model.timer('Total dekomp',showtime):
        for i,(e,var) in enumerate(experiments.items()):
            if i >= maxexp : break     # when we are testing 
            oldvar=adverseny[var].copy()
            if not silent:
                print(i,'Experiment :',e,'\n','Touching: \n', var)
            adverseny[var] = base[var]
            tempdf = model(adverseny   ,start,end,samedata=True,
                silent=msilent)[summaryout]
            adverseny[var] = oldvar
            
            ret_level[e]   = tempdf.loc[start:end,:]
            ret_growth[e]  = tempdf.pct_change().loc[start:end,:] * 100.

        difret_level  = {e : adverse0_level  - ret_level[e]         for e in ret_level}
        difret_growth = {e : adverse0_growth - ret_growth[e]        for e in ret_growth}
    
    df_level = pd.concat([difret_level[v] for v in difret_level],keys=difret_level.keys()).T
    df_growth = pd.concat([difret_growth[v] for v in difret_growth],keys=difret_growth.keys()).T
    if save:
         df_level.to_pickle('data\\' +save +r'_level.pc')    
         df_level.to_pickle('data\\' +save +r'_growth.pc')    
         
    model.save = modelsave # restore the state of model.save 
    return {'level':df_level, 'growth':df_growth}


def ilist(df,pat):
    '''returns a list of variable in the model matching the pattern, 
    the pattern can be a list of patterns of a sting with patterns seperated by 
    blanks
    
    This function operates on the index names of a dataframe. Relevant for attribution analysis
    '''
    if isinstance(pat,list):
           upat=pat
    else:
           upat = [pat]
           
    ipat = upat
    out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(df.index,up.upper()))]  
    return out

    
def GetAllImpact(impact,sumaryvar):
    ''' get all the impact from at impact dataframe''' 
    exo = list({v for v,t in impact.columns})
    df = pd.concat([impact.loc[sumaryvar,c]  for c in exo],axis=1)
    df.columns = exo
    return df 

def GetSumImpact(impact,pat='PD__*'):
    """Gets the accumulated differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:].groupby(level=[0],axis=1).sum()   
    return a 

def GetLastImpact(impact,pat='RCET1__*'):
    """Gets the last differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:].groupby(level=[0],axis=1).last()   
    return a
 
def GetAllImpact(impact,pat='RCET1__*'):
    """Gets the last differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:] 
    return a 

def GetOneImpact(impact,pat='RCET1__*',per=''):
    """Gets differences attributet to each impact group in period:per """ 
    a = impact.loc[ilist(impact,pat),idx[:,per]] 
    a.columns = [v[0] for v in a.columns]
    return a 

def AggImpact(impact):
    """ Calculates the sum of impacts and place in the last column
    
    This function is applied to the result iof a Get* function""" 
    asum= impact.sum(axis=1)
    asum.name = '_Sum'
    aout = pd.concat([impact,asum],axis=1)
    return aout 


class totdif():
    ''' Class to make modelvide attribution analysis 
    
    '''
    
    def __init__(self, model,summaryvar='*',desdic={},experiments = None):
       
       self.diffdf  = model.exodif()
       self.diffvar = self.diffdf.columns
       if len(self.diffvar) == 0:
           print('No variables to attribute to ')
           self.go = False 
           self.typetext = 'Unknown'

       else: 
           self.go = True 
           self.experiments = {v:v for v in self.diffvar} if experiments == None else experiments
           self.model = model 
           self.start = self.model.current_per.tolist()[0]
           self.end = self.model.current_per.tolist()[-1]
           
           self.desdic = desdic 
           self.summaryvar = summaryvar
           self.summaryout = model.vlist(self.summaryvar)
           
           self.res = attribution_new(self.model,self.experiments,self.start,self.end,
            summaryvar=self.summaryvar,showtime=1,silent=1,type=type)
       
    def explain_last(self,pat='',top=0.9,title='',use='level',threshold=0.0,ysize=5):  
        '''
        Explains last period 

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to ''.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.
            ysize (TYPE, optional): DESCRIPTION. Defaults to 5.

        Returns:
            fig (TYPE): DESCRIPTION.

        '''
        if self.go:         
            self.impact = mk.GetLastImpact(self.res[use],pat=pat).T.rename(index=self.desdic)
            ntitle = f'Decomposition last period, {use}' if title == '' else title
            fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title= ntitle,desdic=self.desdic,
                               threshold=threshold,ysize=ysize)
            return fig
   
    def explain_sum(self,pat='',top=0.9,title='',use='level',threshold=0.0,ysize=5): 
        '''
        Explains the sum
        

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to ''.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.
            ysize (TYPE, optional): DESCRIPTION. Defaults to 5.

        Returns:
            fig (TYPE): DESCRIPTION.

        '''
        if self.go:          
           self.impact = mk.GetSumImpact(self.res[use],pat=pat).T.rename(index=self.desdic)
           ntitle = f'Decomposition, sum over all periods, {use}' if title == '' else title
           fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title=ntitle,desdic=self.desdic,
                              threshold=threshold,ysize=ysize )
           return fig
   
    def explain_per(self,pat='',per='',top=0.9,title='',use='level',threshold=0.0,ysize=5):   
        '''
        Explains a periode

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to ''.
            per (TYPE, optional): DESCRIPTION. Defaults to ''.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.
            ysize (TYPE, optional): DESCRIPTION. Defaults to 5.

        Returns:
            fig (TYPE): DESCRIPTION.

        '''
        if self.go:        
           tper = self.res[use].columns.get_level_values(1)[0] if per == '' else per
           self.impact = mk.GetOneImpact(self.res[use],pat=pat,per=tper).T.rename(index=self.desdic) 
           t2per = str(tper.date()) if type(tper) == pd._libs.tslibs.timestamps.Timestamp else tper
           ntitle = f'Decomposition, {use}: {t2per}' if title == '' else title
           fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title=ntitle,desdic=self.desdic ,
                              threshold=threshold,ysize=ysize)
           return fig
   
            
    def explain_allold(self,pat='',stacked=True,kind='bar',top=0.9,title='',use='level',
                    threshold=0.0,resample='',axvline=None): 
        if self.go:
            years = mdates.YearLocator()   # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')
            
            selected =   GetAllImpact(self.res[use],pat) 
            grouped = selected.stack().groupby(level=[0])
            fig, axis = plt.subplots(nrows=len(grouped),ncols=1,figsize=(10,5*len(grouped)),constrained_layout=False)
            width = 0.5  # the width of the barsser
            ntitle = f'Decomposition, {use}' if title == '' else title
            laxis = axis if isinstance(axis,numpy.ndarray) else [axis]
            for j,((name,dfatt),ax) in enumerate(zip(grouped,laxis)):
                dfatt.index = [i[1] for i in dfatt.index]
                if resample=='':
                    tempdf=cutout(dfatt.T,threshold).T
                else:
                    tempdf=cutout(dfatt.T,threshold).T.resample(resample).mean()
#                pdb.set_trace()
                tempdf.plot(ax=ax,kind=kind,stacked=stacked,title=self.desdic.get(name,name))
                ax.set_ylabel(name,fontsize='x-large')
#                ax.set_xticklabels(tempdf.index.tolist(), rotation = 45,fontsize='x-large')
##                ax.xaxis.set_minor_locator(plt.NullLocator())
##                ax.tick_params(axis='x', labelleft=True)
#                ax.xaxis.set_major_locator(years)
#                ax.xaxis_date()
#                ax.xaxis.set_major_formatter(years_fmt)
#                ax.xaxis.set_minor_locator(months)
#                ax.tick_params(axis='x', labelrotation=45,right = True)
            if type(axvline) != type(None): axis.axvline(axvline)    
            fig.suptitle(ntitle,fontsize=20)
            if 1:
                # plt.tight_layout()
                # fig.subplots_adjust(top=top)
                fig.set_constrained_layout(True)

            return fig

    def explain_all(self,pat='',stacked=True,kind='bar',top=0.9,title='',use='level',
                    threshold=0.0,resample='',axvline=None): 
        '''
        Explains all

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to ''.
            stacked (TYPE, optional): DESCRIPTION. Defaults to True.
            kind (TYPE, optional): DESCRIPTION. Defaults to 'bar'.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.
            resample (TYPE, optional): DESCRIPTION. Defaults to ''.
            axvline (TYPE, optional): DESCRIPTION. Defaults to None.

        Returns:
            None.

        '''
        import warnings 
        if self.go:
            years = mdates.YearLocator()   # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')
            
            selected =   GetAllImpact(self.res[use],pat) 
            grouped = selected.stack().groupby(level=[0])
            fig, axis = plt.subplots(nrows=len(grouped),ncols=1,figsize=(10,5*len(grouped)),constrained_layout=False)
            width = 0.5  # the width of the barsser
            ntitle = f'Decomposition, {use}' if title == '' else title
            laxis = axis if isinstance(axis,numpy.ndarray) else [axis]
            for j,((name,dfatt),ax) in enumerate(zip(grouped,laxis)):
                dfatt.index = [i[1] for i in dfatt.index]
                if resample=='':
                    tempdf=cutout(dfatt.T,threshold).T
                else:
                    tempdf=cutout(dfatt.T,threshold).T.resample(resample).mean()
#                pdb.set_trace()
                selfstack = (kind == 'line' or kind == 'area') and stacked 
                tempdf = tempdf.rename(columns=self.desdic)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if selfstack:
                        df_neg, df_pos =tempdf.clip(upper=0), tempdf.clip(lower=0)    
                        df_pos.plot(ax=ax,kind=kind,stacked=stacked,title=self.desdic.get(name,name))
                        ax.set_prop_cycle(None)
                        df_neg.plot(ax=ax,legend=False,kind=kind,stacked=stacked,title=self.desdic.get(name,name))
                        ax.set_ylim([df_neg.sum(axis=1).min(), df_pos.sum(axis=1).max()])
                    else:
                        tempdf.plot(ax=ax,kind=kind,stacked=stacked,title=self.desdic.get(name,name))
                        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                ax.set_ylabel(name,fontsize='x-large')
#                ax.set_xticklabels(tempdf.index.tolist(), rotation = 45,fontsize='x-large')
##                ax.xaxis.set_minor_locator(plt.NullLocator())
##                ax.tick_params(axis='x', labelleft=True)
#                ax.xaxis.set_major_locator(years)
#                ax.xaxis_date()
#                ax.xaxis.set_major_formatter(years_fmt)
#                ax.xaxis.set_minor_locator(months)
#                ax.tick_params(axis='x', labelrotation=45,right = True)
            if type(axvline) != type(None): axis.axvline(axvline)    
            fig.suptitle(ntitle,fontsize=20)
            if 1:
                fig.set_constrained_layout(True)

                # plt.tight_layout()
                # fig.subplots_adjust(top=top)
                ...
            return fig
        
#
    def totexplain(self,pat='*',vtype='all',stacked=True,kind='bar',per='',top=0.9,title=''
                   ,use='level',threshold=0.0,ysize=10,**kwargs):
        '''
        Wrapper for different explanations
         - :any:`explain_last` 
         - :any:`explain_per` 
         - :any:`explain_sum` 
         - :any:`explain_all` 
            

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to '*'.
            vtype (per|all|last|sum, optional): what data to attribute. Defaults to 'all'.
            stacked (TYPE, optional): DESCRIPTION. Defaults to True.
            kind (TYPE, optional): DESCRIPTION. Defaults to 'bar'.
            per (TYPE, optional): DESCRIPTION. Defaults to ''.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.
            ysize (TYPE, optional): DESCRIPTION. Defaults to 10.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            fig (TYPE): DESCRIPTION.

        '''       
        if vtype.upper() == 'PER' : 
            fig = self.explain_per(pat=pat,per=per,top=top,use=use,title=title,threshold=threshold,ysize=ysize)
            
        elif vtype.upper() == 'LAST' : 
            fig = self.explain_last(pat=pat,top=top,use=use,title=title,threshold=threshold)
            
        elif vtype.upper() == 'SUM' : 
            fig = self.explain_sum(pat=pat,top=top,use=use,title=title,threshold=threshold)
            
        else:    
            fig = self.explain_all(pat=pat,stacked=stacked,kind=kind,top=top,use=use,title=title,threshold=threshold)
        return fig 

#    def get_att_gui(self,var='FY',spat = '*',desdic={},use='level'):
#        '''Creates a jupyter ipywidget to display model level 
#        attributions ''' 
#        def show_all2(Variable,Periode,Save,Use):
#             global fig1,fig2
#             fig1 = self.totexplain(pat=Variable,top=0.87,use=Use)
#             fig2 = self.totexplain(pat=Variable,vtype='per',per = Periode,top=0.85,use=Use) 
#             if Save:
#                fig1.savefig(f'Attribution-{Variable}-{use}.pdf')
#                fig2.savefig(f'Attribution-{Variable}-{Periode}-{use}.pdf')
#                print(f'Attribution-{Variable}-{use}.pdf and Attribution-{Variable}-{Periode}-{use}.pdf aare saved' )
#    
#        show = ip.interactive(show_all2,
#                  Variable = ip.Dropdown(options = sorted(self.model.endogene),value=var),
#                  Periode  = self.model.current_per,
#                  Use = ip.RadioButtons(options= ['level', 'growth'],description='Use'),
#                  Save = False,
#                  )
#        return show 


if __name__ == '__main__' :
    #%%
    # running withe the mtotal model 
    df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
    df3 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,40.,60.,10.] ,'YD':[10.,49.,36.,40.]},index=[2017,2018,2019,2020])
    ftest = ''' 
    FRMl <>  ii = TY(-1)+c(-1)+Z*c(-1) $
    frml <>  c=0.8*yd+log(1) $
    frml <>  d = c +2*ii(-1) $
   frml <>  c2=0.8*yd+log(1) $
    frml <>  d2 = c + 42*ii $
   frml <>  c3=0.8*yd+log(1) $
    frml <>  d3 = c +ii $
 '''
     
    m2=mc.model(ftest,straight=True,modelname='m2 testmodel')
    df2=mc.insertModelVar(df2,m2)
    df3=mc.insertModelVar(df3,m2)
    z1 = m2(df2)
    z2 = m2(df3)
    ccc = m2.totexplain(pat='D2',per=2019,vtype='all',top=0.8)
    ccc = m2.totexplain('D2',vtype='last',top=0.8)
    ccc = m2.totexplain('D2',vtype='per',top=0.8)
#%%        
    ddd = totdif(m2)
    eee = totdif(m2)
    ddd.totexplain('D2',vtype='all',top=0.8,use='growth');
    eee.totexplain('D2',vtype='all',top=0.8);

    if  False and ( not 'mtotal' in locals() ) :
        # get the model  
        with open(r"models\mtotal.fru", "r") as text_file:
            ftotal = text_file.read()
       
        #get the data 
        base0   = pd.read_pickle(r'data\base0.pc')    
        base    = pd.read_pickle(r'data\base.pc')    
        adve0   = pd.read_pickle(r'data\adve0.pc')      
    #%%    
        mtotal  = mc.model(ftotal)
        
#        prune(mtotal,base)
        #%%    
        baseny        = mtotal(base0   ,'2016q1','2018q4',samedata=False)
        adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
    #%%
        diff  = mtotal.exodif()   # exogeneous variables which are different between baseny and adverseny 
    #%%
        assert 1==2  # just for stopping in test situations 
        #%%  
        adverseny            = mtotal(adve0   ,'2016q1','2018q4',samedata=True)   # to makew sure we have the right adverse.
        countries            = {c.split('__')[2]  for c in diff.columns}  # list of countries 
        countryexperiments   = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in countries } # dic of experiments 
        assert len(diff.columns) == sum([len(c) for c in countryexperiments.values()]) , 'Not all exogeneous chocks variables are accountet for'
        countryimpact        = attribution(mtotal,countryexperiments,save='countryimpactxx',maxexp=30000,showtime = 1)    
        #%% 
        adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
        vartypes             = {c.split('__') [1]  for c in diff.columns}
        vartypeexperiments   = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in vartypes }
        assert len(diff.columns) == sum([len(c) for c in vartypeexperiments.values()]) , 'Not all exogeneous chocks variables are accountet for'
        vartypeimpact        = attribution(mtotal,vartypeexperiments,save='vartypeimpactxx',maxexp=3000,showtime=1)
        ##%%
        #adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
        #allexo                  = {c[7:14] for c in diff.columns} 
        #allexoexperiments       = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in allexo }
        #allexoimpact            = attribution(mtotal,allexoexperiments,base,adverseny,save='allexoimpact',maxexp=2000)
    #%% test of upddf    
        if 0:
            baseny        = mtotal(base0   ,'2016q1','2018q4',samedata=False)
            adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
    #%%
            e = 'EE'     
            var = countryexperiments['EE']
            vardiff = diff[var]
            temp = adverseny[var].copy()
            adverseny[var] = baseny[var]
            temp2 = adverseny[var].copy()
            _ = mc.upddf(adverseny,temp) 
            adverseny[var] = temp
            temp3 = adverseny[var].copy()
            
