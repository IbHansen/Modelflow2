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
mcorona,df  = model.modelload('coronatest.json')  # Get the model and data
res = mcorona(df,keep='Basis')

#%% plot
with mcorona.set_smpl(60,200):
    mcorona.keep_plot('dead hospital_icu infectious',diff=0,legend=1,dec='2',
                                        showtype='level',scale='linear');
#%% find jacobi   
ti = targets_instruments_delayed(databank=res,model=mcorona,targets=['DEAD_GROWTH','HOSPITAL_GROWTH'],instruments= [('PROBABILITY_TRANSMISION',0.01)])
ti.instruments
ti.jacobi(120,delay=20)
os.environ['PYTHONBREAKPOINT'] = ''
over_delay = pd.concat([ti.jacobi(120,delay=d).rename(columns=lambda x:f'{d}') for d in range(50)],axis=1).T
over_delay.plot()
