# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:04:29 2019

@author: hanseni
"""

import pandas as pd
from numba import jit
import fnmatch 
import numpy as np
import os


from modelclass import ttimer, insertModelVar
from modelsandbox import newmodel
import modelnet as mn
import modelmanipulation as mp
turbo = 0
os.environ['PYTHONBREAKPOINT'] = '1'

@jit("f8(b1,f8,f8)")
def recode(condition,yes,no):
    '''Function which recratetes the functionality of @recode from eviews ''' 
    return yes if condition else no
if 1:
    basedf = pd.read_pickle('data/longmcebase.pc').pipe(
    lambda df: df[[c for c in df.columns if not c.startswith('DA')]].astype(np.float64))
    ffrbusmce_xx  =   open('model/ffrbusmce.txt','rt').read()
    ffrbusmce = mp.un_normalize_simpel(ffrbusmce_xx)
    mfrbusmce = newmodel(ffrbusmce,funks=[recode],straight=True,modelname='FRB/US Forward expectations')
#mfrbusmce.use_preorder = True
assert 1==1
#%%
#basedf = insertModelVar(basedf,mfrbusmce) 

with ttimer('baseline newton all periods '):
    baseres = mfrbusmce.newtonstack(basedf,'2020q1','2040q4',stats=1,antal=70,silent=False,nchunk=50,reset=1,
                                  ljit=turbo,nljit=turbo,timeit=0,nonlin=20,newtonalfa = 1.0, newtonnodamp=20)
#%%

altdf = baseres.copy()
altdf=altdf.mfcalc('''\
rffintay_aerr = rffintay_aerr -0.3
dmpex    = 0 
dmprr    = 0
dmptay   = 0
dmptlr   = 0 
dmpintay = 1

dmpalt   = 0
dmpgen   = 0
'''.upper(),silent=1)
    
    
with ttimer('newton all periods '):
    altres = mfrbusmce.newtonstack(altdf,stats=0,antal=70,silent=False,reset=0,
                                  ljit=turbo,timeit=0,nonlin=10,newtonalfa = 1., newtonnodamp=2,nljit=turbo,)
mfrbusmce.basedf= baseres 
mfrbusmce.lastdf= altres 
mfrbusmce['rff RFFINTAY'].dif.plot(colrow=1)
mfrbusmce['rff RFFINTAY'].dif.print
#%%
difeq = mfrbusmce.newton_diff.diff_model.equations
pymodel = mfrbusmce.newton_diff.diff_model.make_res_text2d
#diffendo = mfrbusmce.newton_diff.modeldiff()
difres = mfrbusmce.newton_diff.difres
mfrbusmce.newton_diff.diff_model.allvar['ZECO__P__QEC__LEAD__1']['frml']


xx=basedf.RFF - baseres.RFF
mfrbusmce.rffintay

melted = mfrbusmce.newton_diff.get_diff_mat_tot()
stacked = mfrbusmce.newton_diff.stacked
113447/stacked.shape[0]**2
  