# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:45:27 2019

@author: ibh
"""

import pandas as pd
import re
import yaml 
import pickle
import os 

from modelclass import model 
from modelhelp import ttimer, insertModelVar
import modelmanipulation as mp 
from modelpattern import find_statements, split_frml,kw_frml_name,find_frml
import modelnet as mn
import gc

def frml_code_translate(in_equations):
    ''' takes a model and makes a new model by translating PCIM frml_codes to 
   ModelFlow frml_names 
    
    :exo: the value can be fixed in to a value valuename_x by setting valuename_d=1

    :jled: a additiv adjustment element is added to the frml

    :jrled: a multiplicativ adjustment element is added to the frml 
    '''    
    
    nymodel = []
    equations = in_equations[:]  # we want do change the e
    for comment, command, value in find_statements(equations.upper()):
        # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        elif command == 'FRML':
            a, fr, n, udtryk = split_frml((command + ' ' + value).upper())
            #print(fr,n, kw_frml_name(n.upper(),'EXO'))
            # we want a relatov adjustment
            newn = [n.strip()]
            nt=n+'             ' # an quick way to ensure length
            if n.startswith('_'):
                if 'J_' == nt[2:4]:
                    newn.append('J')
                elif 'JR' == nt[2:4]:
                    newn.append('JR')
                elif 'JD' == nt[2:4]:
                    newn.append('JD')    
                if 'D' == nt[4:5]:
                    newn.append('EXO')    
                if 'Z' == nt[5:6] or  'S' == nt[1]:
                    newn.append('DAMP') 
                outn = '<'+','.join(newn)+ '>' 
            else:
                outn = n.strip() 
            nymodel.append(command + ' ' + outn + ' ' + udtryk)
        else:
            nymodel.append(command + ' ' + value)

    equations = '\n'.join(nymodel)
    return equations

def adam_exounroll(in_equations):
    ''' takes a model and makes a new model by enhancing frml's with <exo=,j=,jr=> 
    in their frml name. 
    
    :exo: the value can be fixed in to a value valuename_x by setting valuename_d=1

    :jled: a additiv adjustment element is added to the frml

    :jrled: a multiplicativ adjustment element is added to the frml 
    '''    
    
    nymodel = []
    equations = in_equations[:]  # we want do change the e
    for comment, command, value in find_statements(equations.upper()):
        # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        elif command == 'FRML':
            a, fr, n, udtryk = split_frml((command + ' ' + value).upper())
            #print(fr,n, kw_frml_name(n.upper(),'EXO'))
            # we want a relatov adjustment
            if kw_frml_name(n, 'JRLED') or kw_frml_name(n, 'JR'):
                lhs, rhs = udtryk.split('=', 1)
                udtryk = lhs + \
                    '= (' + rhs[:-1] + ')*(1+JR' + lhs.strip() + ' )' + '$'
            if  kw_frml_name(n, 'JD'):
                lhs, rhs = udtryk.split('=', 1)
                udtryk = lhs + \
                    '= (' + rhs[:-1] + ')+(JD' + lhs.strip() + ' )' + '$'
            if kw_frml_name(n, 'JLED') or kw_frml_name(n, 'J'):
                lhs, rhs = udtryk.split('=', 1)
                # we want a absolute adjustment
                udtryk = lhs + '=' + rhs[:-1] + '+ J' + lhs.strip() +  '$'
            if kw_frml_name(n, 'EXO'):
                lhs, rhs = udtryk.split('=', 1)
                endogen = lhs.strip()
                dummy = 'D'+endogen 
                exogen = 'Z'+endogen 
                udtryk = lhs + \
                    '=(' + rhs[:-1] + ')*(1-' + dummy + ')+' + exogen + '*' + dummy + '$'

            nymodel.append(command + ' ' + n + ' ' + udtryk)
        else:
            nymodel.append(command + ' ' + value)

    equations = '\n'.join(nymodel)
    return equations


#%% get the model 
with open('model/smec.frm','rt') as f:
    rsmec0 = f.read()
with open('model/varlist.dat','rt') as f:
    vsmec = f.read()
    
# breakpoint()    
rsmec,asmec = rsmec0.split('AFTER $')
esmec00 = re.sub('//.*$','',rsmec,flags=re.MULTILINE).replace('Dif(','diff(')
esmec0 = re.sub('^[" "]*$\n','',esmec00,flags=re.MULTILINE).replace(';','$').replace('[','(').replace(']',')')
esmec1 = frml_code_translate(esmec0)
esmec2 = mp.normalize(esmec1)
esmec3 = adam_exounroll(esmec2)
fsmec = mp.explode(esmec3)
msmec = model(fsmec)

msmec.normalized =True


fnsmec = mp.un_normalize_model(fsmec)
mnsmec = model(fnsmec)
mnsmec.normalized =False

#%% get the data 
e19df = pd.read_excel('data/SMEC_lang.xlsx',index_col=0).T.pipe(lambda df:df.rename(columns = {v:v.upper() for v in df.columns}))
# e19df.index = dates 
dates = pd.period_range(start=e19df.index[0],end=e19df.index[-1],freq='Y')
baseline = insertModelVar(e19df,[msmec,mnsmec]).fillna(0.0) 

# make a dict of descriptions 
vtrans  = {var.split('\n')[1]: {'des' :var.split('\n')[2]} for i,var in enumerate(vsmec.split('----------')) if 2 <= len(var.split('\n'))}
vsmec = {v : i['des'] for v,i in vtrans.items()}



if 0:
    fig2 = mn.draw_adjacency_matrix(msmec.endograph,msmec.precoreepiorder,msmec._superstrongblock,msmec._superstrongtype,
                                        size=(100,100))


#%%
#%%
per1 = 2020 
per2 = 2050   
athreshold = 0.001
pthreshold = 0.000001
maxerr = 10
if 0:
    res1 = msmec.sim2d(baseline,per1,per2,antal=300,first_test=10,conv='FY',relconv=0.0001,
                       ldumpvar=True)
    
    msmec.basedf = e19df
    msmec.lastdf = res1
    errors = 0
    for var in msmec.solveorder:
        err = False
        try:
            pdif = 100. * (res1.loc[per2,var]/e19df.loc[per2,var]-1)
            adif = res1.loc[per2,var]-e19df.loc[per2,var]
            if abs(pdif )>= pthreshold:
                err = True
        except:
            breakpoint()
            adif = res1.loc[per2,var]-e19df.loc[per2,var]
            pdif = -99999
            if abs(adif) >= athreshold:
                err = True
        if err:
            errors += 1
            print(f'Endogeneous: {var} \nFormular: {msmec.allvar[var]["frml"]}')
     
            print(f'\nValues : \n{msmec.get_values(var)}\n' )
            #msmec.varvis(var=var).show
        if errors >= maxerr:break 
    
        msmec['fy ul enl'].dif.plot(sharey=False,colrow=1,title='Forskel mellem DØRS grundforløb og Modelflow')
        msmec['fy ul enl'].difpct.plot(sharey=True,colrow=1,title='Forskel mellem DØRS grundforløb og Modelflow, i pct')
        res1.to_pickle(r'data\basis201909.pc') 
        with open('model/smecfeb19.txt','wt') as f:
            f.write(fsmec)
        with open('model/smec.des.yaml','wt') as f:
            f.write(yaml.dump(vtrans))
# #%%
#         os.environ['PYTHONBREAKPOINT'] = ''
#         muldf = baseline.copy() 
#         with mc.ttimer('simulation  baseline ')  as t: 
#             grund = mnsmec(baseline,per1,per2,max_iterations=1100,first_test=3,conv='FY',relconv=0.0001,solver='newtonstack',reset=1,
                            # ldumpvar=False,alfa = 0.1,silent=0,ljit=turbo,chunk=30,stats=1,forcenum=0)
#%%
turbo = 0
muldf = baseline.copy() 

muldf.TG = muldf.TG + 0.05 
assert 1==1
 
#%% Try different solvers 
if 0:
    os.environ['PYTHONBREAKPOINT'] = '99'

    turbo =  0
    msmec.keep_solutions = {}
    for init in [False]:
        for solver in ['sim','sim1d','newton']:   #,'newtonstack']:
        # for solver in ['newton']:
        # for solver in ['sim']:
            with model.timer(f'Solve with:{solver:26} init:{init}',short=True)  as t:
                _ = msmec(muldf,2020,2150,solver=solver,max_iterations=400,
                          ljit=turbo,chunk=30,
                          ldumpvar=True,dumpvar='*',
                          conv='fy*',first_test=20,alfa=.4,
                          relconv=0.0000000000001,
                          nonlin=19,newtonalfa = 1.,newtondamp=10,newton_absconv=0.0001,reset=1,forcenum=0,
                          silent=1 if solver != 'newton' else 1,timeit=False,stats=0,init=init,
                          keep_solution=solver,keep_variables='*'
                               )
            # msmec.show_iterations('ul fy tion',per=2020,last=-20,change=0) 
    # msmec.keep_print('tg',diff=0)
# #%%
# msmec.show_iterations('ul enl fy',per=2020,last=-400,change=1) 
# #%%
assert 1==1
if 0:
    mnsmec.keep_solutions = {}
    for solver in ['newton_un_normalized','newtonstack_un_normalized']:
    # for solver in ['newton_un_normalized']:
        with model.timer(f'Solve with:{solver:26} ',short=True)  as t:
            res2 = mnsmec(muldf,2020,2025,**{**msmec.oldkwargs,**{'solver':solver,'keep_solution':solver}})
    mnsmec.keep_solutions = {**msmec.keep_solutions,**mnsmec.keep_solutions}
    # mnsmec.keep_print('*',diff=1,start=2045,slut=2050)
    # mnsmec.keep_plot('*',diff=1,start=2045,slut=2050)
    xx = mnsmec.keep_get_dict('*',diff=1)

            # mnsmec.show_iterations('UL enl',last=-300,change=True) 
#%% Try different values  
if 1:
    os.environ['PYTHONBREAKPOINT'] = ''

    turbo =  1
    msmec.keep_solutions = {}
    for tg_plus in [-0.06,  0.  ,0.06 ]:
        muldf = baseline.copy() 

        muldf.TG = muldf.TG + tg_plus 
        print(f'\nExperiment tg + {tg_plus} ',flush=True)
        for solver in ['sim','sim1d','newton','newtonstack']:   #,'newtonstack']:
        # for solver in ['newton']:
        # for solver in ['sim']:
            with model.timer(f'Solve with:{solver:26}',short=True)  as t:
                _ = msmec(muldf,2020,2040,solver=solver,max_iterations=400,
                          ljit=turbo,chunk=30,transpile_reset=True,stringjit=0,
                          ldumpvar=True,dumpvar='*',
                          conv='fy*',first_test=20,alfa=.4,
                          relconv=0.0000000000001,
                          nonlin=15,newtonalfa = 1.,newtondamp=10,newton_absconv=0.0001,reset=1,forcenum=0,
                          silent=1 if solver != 'newton' else 1,timeit=False,stats=0,init=False,
                          keep=f'Experiment "TG + {tg_plus}, solver:{solver}":',keep_variables='*'
                             )

# msmec['fy ul enl'].dif.rename(vsmec).plot(sharey=False,colrow=1,title='Difference of solutions')
#%% try plot
         
os.environ['PYTHONBREAKPOINT'] = ''

with msmec.set_smpl(2017,2200):        
    msmec.keep_plot('ul enl fy')

