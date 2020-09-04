# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:20:00 2019

@author: hanseni
"""

import pandas as pd
import numpy as np
import scipy as sp 

from modelsandbox import newmodel 
from modelmanipulation import explode
import modeldiff as md
import modelclass as mc

fsolow = explode('''\
Y = a * k**alfa * l **(1-alfa) 
C = (1-SAVING_RATIO)  * Y 
I = Y - C 
diff(K) = -depreciates_rate * K(-1) + I 
diff(l) = labor_growth * L(-1) 
K_intense = K/L 
''')

msolow = newmodel(fsolow)
msolow.drawmodel()
msolow.drawendo()
#%%
N = 300
df = pd.DataFrame({'L':[100]*N,'K':[100]*N})
df.loc[:,'ALFA'] = 0.3
df.loc[:,'A'] = 1.
df.loc[:,'DEPRECIATES_RATE'] = 0.02
df.loc[:,'LABOR_GROWTH'] = 0.01
df.loc[:,'SAVING_RATIO'] = 0.05

res1 = msolow(df,antal=100,first_test=10,silent=1)
#df.loc[:,'LABOR_GROWTH'] = 0.02
#
#res2 = msolow(df,antal=100,first_test=10,silent=1)
#_ = msolow[['Y','c','i']].plot_alt()
#msolow.C
#msolow.dekomp('C')

#%%'
def get_dif_mat(mmodel):
    dmodel = md.diffmodel(mmodel,onlyendocur=False   )
    difres = dmodel.res(mmodel.lastdf,mmodel.current_per[0],mmodel.current_per[0]).loc[1,dmodel.endogene]
    lags = {i.rsplit('__',2)[2] for i in dmodel.endogene if i.rsplit('__',2)[1] == 'LAG'}
    maxlag= max(lags)
    leads = {i.rsplit('__',2)[2] for i in dmodel.endogene if i.rsplit('__',2)[1] == 'LEAD'}
    maxleads= 0 if len(leads) == 0 else max(leads)
    clag0b=[[*i.rsplit('__',2)[0].split('__P__'),difres.loc[i]] for i in dmodel.endogene if i.endswith('LAG__0')]
    mlag0 = pd.DataFrame(clag0b,columns=['lhs','rhs','value']).pivot(index='lhs',columns='rhs',values='value')
    all_endo = sorted(list(mmodel.endogene))
    m = mlag0.reindex(columns=all_endo,index=all_endo).fillna(0.0)
    np.fill_diagonal(m.values, -1.0)
    mv = m.values
    return m
startdf =mc.insertModelVar(df,msolow)
m = get_dif_mat(msolow)
endo = m.columns
mv = m.values
mvinv = np.linalg.inv(mv)
minv = pd.DataFrame(mvinv,index=m.index,columns=m.columns)
first = msolow.res(startdf,1,1)
now   = first.loc[1,endo]
before = startdf.loc[1,endo]
distance = now-before
new = before+minv
