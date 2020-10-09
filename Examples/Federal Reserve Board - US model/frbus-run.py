# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:04:29 2019

@author: hanseni 
"""

import pandas as pd
from numba import jit
import fnmatch 
import scipy as sp 
import gc


from modelclass import model 
import modelclass as mc
import modelmanipulation as mp
turbo = 0
silent = 1
chunk = 30

basedf = pd.read_pickle('data/baseline.pc')
ffrbus  =   open('model/ffrbusvar.txt','rt').read()
mfrbus = model(ffrbus,modelname = 'FRB_US - var')
''
fnfrbus = mp.un_normalize_model(mp.explode(ffrbus))
mnfrbus = model(fnfrbus,modelname='FRB_US_ - var unnormalized')
mnfrbus.normalized =False

#%%
if 0:
#%%
    figbig = mn.draw_adjacency_matrix(mfrbus.endograph,mfrbus.precoreepiorder,mfrbus._superstrongblock,mfrbus._superstrongtype,size=(80,80))
    figsmall   = mn.draw_adjacency_matrix(mfrbus.endograph,mfrbus.precoreepiorder,mfrbus._superstrongblock,mfrbus._superstrongtype,size=(10,10))
    figsmall.savefig('graph/FRBUS adjancy matrix small.pdf')
    figbig.savefig('graph/FRBUS adjancy matrix big.pdf')
#%%
    mfrbus.drawendo(lag=0,title='FRB/US endogeneous interpendencies',
                    saveas='FRBUS_Endogeneous_structure2',
                    cluster={'Prolog': mfrbus.prevar,
                             'Epilog': mfrbus.epivar,
                             'Core'  : mfrbus.coreorder})
#%%
with model.timer('Gauss periode by period baseline'):
    resgs = mfrbus(basedf,'2020q1','2025q4',max_iterations= 500,relconv=0.000000001,
                 conv='XGDPN',first_test=1,ldumpvar=False,ljit=turbo,silent=silent,debug=0)
#%%
altdf = basedf.copy()
altdf=altdf.eval('''\
rffintay_aerr = rffintay_aerr + 0.5 
dmpex    = 0 
dmprr    = 0
dmptay   = 0
dmptlr   = 0 
dmpintay = 1
dmpalt   = 0
dmpgen   = 0
'''.upper())
silent = 1
first_test = 5
max_iterations = 1000
timeon=0
relconv = 0.00000001
turbo=0
newton_reset = 0
transpile_reset=0
stringjit=1
chunk = 25

altdf = mc.insertModelVar(altdf,[mfrbus,mnfrbus])
mfrbus.keep_solutions={}
for solver in ['sim','sim1d','newton','newtonstack']:   #,'newtonstack']:
# for solver in ['newton']:   #,'newtonstack']:
    with model.timer(f'Solve with:{solver:26}',short=True)  as t:

        _ = mfrbus(altdf,'2020q1','2025q4',
            solver=solver,max_iterations= max_iterations,
            relconv = relconv,alfa=0.5,chunk=chunk,
            conv='XGDPN',dumpvar ='rff*',transpile_reset=transpile_reset,newton_reset = newton_reset,
            first_test=first_test,ldumpvar=False,
            ljit=turbo,silent=silent,debug=1,
            timeon=timeon,stats=0,forcenum=1,stringjit=stringjit,keep=solver)
# mfrbus.keep_plot('rff',diff=0,dec='10')

mnfrbus.keep_solutions=mfrbus.keep_solutions
for solver in ['newton_un_normalized','newtonstack_un_normalized']:   #,'newtonstack']:
    with model.timer(f'Solve with:{solver:26}',short=True)  as t:

        _ = mnfrbus(altdf,'2020q1','2025q4',
            solver=solver,max_iterations= max_iterations,
            relconv = relconv,alfa=0.5,chunk=chunk,
            conv='XGDPN',dumpvar ='rff*',transpile_reset=transpile_reset,newton_reset=newton_reset,
            first_test=first_test,ldumpvar=False,
            ljit=turbo,silent=silent,debug=1,
            timeon=timeon,stats=0,forcenum=1,stringjit=stringjit,keep=solver)
if 0:
    mnfrbus.keep_plot('rff',diff=1,dec='10',legend=0)
