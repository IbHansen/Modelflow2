# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:52:40 2020

@author: IBH

Test af target and instruments with delay. 

This feature is for use in calibrating iput variable, which takes effect after some periode


"""
import os
import pandas as pd

from modelclass import model
from modelinvert import targets_instruments_delayed

#%% load and run
mcorona,df  = model.modelload('coronatest.json')  # Get the model and data, saved in "Corona specify model and make eksperiments.ipynb"
df.index = df.index-100
df.index.name = 'Day'
res = mcorona(df,keep='Basis',silent=1)

#%% find jacobi   
ti = targets_instruments_delayed(databank=res,model=mcorona,targets=['DEAD_GROWTH','HOSPITAL_GROWTH','INFECTIOUS_GROWTH'],instruments= [('PROBABILITY_TRANSMISION',0.01)])
ti.instruments
ti.jacobi(0,delay=20)
os.environ['PYTHONBREAKPOINT'] = ''
over_delay = pd.concat([ti.jacobi(0,delay=d).rename(columns=lambda x:f'{d}') for d in range(50)],axis=1).T
over_delay.plot()
#%%
def ti_calibrate(mmodel,instrument,target,value,time,delay=0,silent=1,show=1):
    mmodel.keep_solutions,old_keep_solutions= {},mmodel.keep_solutions
    mmodel.keep_solutions['Before calibration'] = mmodel.lastdf.copy(deep=True)
    targetdf=pd.DataFrame(value,index=[time],columns=[target])
    ti = targets_instruments_delayed(databank=mmodel.lastdf,model=mmodel,targets=targetdf,instruments= [instrument])
    res = ti(mmodel.lastdf,delay=delay,silent=1,debug=1)
    res2 = mmodel(res,silent=silent,keep='After calibration')
    if show:
        figs = mcorona.keep_plot(f'{instrument[0]} {target}',diff=0,legend=0,dec='2',showtype='level',scale='linear');
        for keep,fig in figs.items():
          ymax = fig.axes[0].get_ylim()[1]
          fig.axes[0].annotate('  Calibration time',xy=(time,ymax),fontsize = 13,va='top')
          fig.axes[0].axvline(0,linewidth=4, color='r',ls='dashed')
          
    mmodel.keep_solutions,model_calibrate_keep_solutions =  old_keep_solutions,mmodel.keep_solutions
    return res2
time = 0
delay = 5 
res = mcorona(df,keep='Basis',silent=1)
res2 = ti_calibrate(mcorona,instrument = ('PROBABILITY_TRANSMISION',0.01),
                    target = 'INFECTIOUS_GROWTH',value=3,time=time,delay=delay)
#%%
with mcorona.set_smpl(-30,300):
      figs = mcorona.keep_plot('dead hospital_icu infectious',diff=0,legend=1,dec='0',
                                        showtype='level',scale='linear');
      for keep,fig in figs.items():
          ymax = fig.axes[0].get_ylim()[1]
          fig.axes[0].annotate('Calibration time',xy=(time,ymax),fontsize = 15)
          fig.axes[0].axvline(0,linewidth=4, color='r',ls='dashed')
# with mcorona.set_smpl(-30,300):
#       fig2 = mcorona.keep_plot('PROBABILITY_TRANSMISION',diff=0,legend=1,dec='3',
#                                         showtype='level',scale='linear');

#%% try target a growth 
if 0:
#%%    
    targetdf=pd.DataFrame(2,index=[0],columns=['INFECTIOUS_GROWTH'])
    
    ti_growth = targets_instruments_delayed(databank=res,model=mcorona,targets=targetdf,instruments= [('PROBABILITY_TRANSMISION',0.01)])
    # ti_growth.jacobi(120,delay=10)
    res = ti_growth(res,delay=10,silent=1,debug=1)
    res2 = mcorona(res,silent=0,keep='After calibration')
    #x %% plot
    with mcorona.set_smpl(-30,300):
        mcorona.keep_plot('dead hospital_icu infectious',diff=0,legend=1,dec='2',
                                        showtype='level',scale='linear');
