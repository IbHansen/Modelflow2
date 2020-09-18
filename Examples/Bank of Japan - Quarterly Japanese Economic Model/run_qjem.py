# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:09:33 2019

@author: ibh
"""

import pandas as pd 
from modelsandbox import newmodel
from modelclass import ttimer
from modelinvert import targets_instruments

with open('qjem/model/fqjem.frm','rt') as f: 
    fqjem =f.read()
baseline = pd.read_pickle('qjem/data/dfqjem.pk')    

mqjem = newmodel(fqjem)

mqjem.maxlead
mqjem.use_preorder = 1
turbo = 1
#%%
with ttimer('Solve Qjem'):
    res = mqjem(baseline,antal=50,first_test = 1,ljit=turbo,chunk=49)
    
altdf = baseline.copy()
instruments = [ 'V_NUSGAP','V_USGAP']
target      = altdf.loc['2005q1':,['USGDP','NUSGDP']].mfcalc('''\
USGDP  = USGDP*1.01
NUSGDP = NUSGDP*1.01
''',silent=1)

with ttimer('Solve the model as a targets-instruments instrumnets problem'):
    resalt = mqjem.control(baseline,target,instruments,silent=0,ljit=turbo,nonlin=3)

#%%
_ = mqjem['USGDP NUSGDP'].dif.plot(colrow=1)
_ = mqjem['V_NUSGAP V_USGAP'].dif.plot(colrow=1)
_ = mqjem['GDP U PGDP'].dif.plot(sharey=0,colrow=1)
#mqjem.USGDP.draw(up=2,down=7,HR=0)
# mqjem['ZPI_V*'].frml
