# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:12:34 2019

@author: hanseni
"""
import pandas as pd

from modelclass import model 

#%%
if not 'madam' in locals(): 
    bank  = pd.read_pickle('adam/lang100_2017.pc') 
    fadam = open('adam/jul17x.txt','rt').read()
    madam = model(fadam)    
    madam.use_preorder = True
#%%
    with open('adam/varlist.txt','rt') as f:
        des = f.read()
    
    var_description  = {var.split('\n')[1].upper(): var.split('\n')[2] for i,var in enumerate(des.split('----------')) if 3 <= len(var.split('\n'))}
    madam.set_var_description(var_description)
    #%%   
assert 1==1
grund = madam(bank,2018,2030,conv='YR UL',max_iterations=100,alfa=0.4,stats=0,ljit=0,chunk=30,
              sim2= 0,debug=1)

madam.modeldump('adam/jul17x.pcim')
mulstart =grund.copy()
mulstart.TG = grund.TG+0.02
mulstart.TFN_O = grund.TFN_O - 10
mul = madam(mulstart,2018,2030,conv='YR UL',max_iterations=100,alfa=0.5,stats=0,ljit=0,chunk=30, relconv = 0.000000001)
                 
# print(madam[['ENL','ul','fy','tg']].dif.df)
# _ = madam.totexplain(pat='fy',top=0.86)
assert 1==1
#%%
solverlist = ['base_sim', 'sim', 'sim1d','newton']   #,'newtonstack']
modeldic  = {solver: {'model':model(fadam)} for solver in solverlist}
#%%
turbo = 1
for solver in solverlist:
# for solver in ['newton']:
# for solver in ['sim']:
     with model.timer(f'Solve with:{solver:26} ',short=True)  as t:
         _ = modeldic[solver]['model'](mulstart,2018,2025,solver=solver,max_iterations=600,
                   ljit=turbo,chunk=30,
                   ldumpvar=0,dumpvar='*',
                   conv='cp cv e m pct' if 1 else '*',first_test=4,alfa=.2,
                   relconv=0.0000000001,
                   nonlin=40,newtonalfa = 1.,newtondamp=10,newton_absconv=0.0001,reset=0,forcenum=1,
                   silent=1,timeit=False,stats=0,init=0,
                   keep_solution=solver,keep_variables='enl fy'
                        )
#%%
madam.keep_solutions={} 
for solver in solverlist:
    thismodel = modeldic[solver]['model']
    madam.current_per = thismodel.current_per
    thismodel.show_iterations('ul',per=2025,last=-10000,change=1)
    madam.keep_solutions = {**madam.keep_solutions,**thismodel.keep_solutions}
     # msmec.show_iterations('ul fy tion',per=2020,last=-20,change=0) 
madam.keep_print('*',diff=1)
