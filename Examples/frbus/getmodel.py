# -*- coding: utf-8 -*-
"""
This i a script which handels the FRB/US model 
the model folder has to be downloaded from the FRB home page

Phases:
    
    1. Load the model from the xml file 
    2. Transform the models eviews equations to ModelFlow Business logic (the tricky part) 
    3  Make a ModelFlow model instance
    4. Save the data (in eview) to a csv file 
    5. Load the Data 
    6. solve the model 
    
"""
import xml
from sympy import solve, sympify
import re
from numba import jit




from modelmanipulation import find_arg, lagone,udtryk_parse
import modelmanipulation as mp
import modelclass as mc
import modelclass2 as mc2
import modelnet as mn
from  modelpattern import namepat,lagpat
import modelsandbox as ms

import pandas as pd

from getmodelmep import mcedic

assert 1==1


#%%
@jit("f8(b1,f8,f8)")
def recode(condition,yes,no):
    '''Function which recratetes the functionality of @recode from eviews ''' 
    return yes if condition else no

def getinf(y): 
    ''' Extrct informations on a variables and equations 
    in the FRB/US model'''
    name = y.find('name').text
    definition = y.find('definition').text
    type = y.find('equation_type').text
    
#    print('name:',name)
    if y.find('standard_equation'):
        neq= y.find('standard_equation/eviews_equation').text.replace('\n','').replace(' ','')
        cdic = {coeff.find('cf_name').text : coeff.find('cf_value').text for coeff in y.findall('standard_equation/coeff')}
        if 'y_dmptlur(1)' in cdic: cdic['y_dmptlur(1)']='-25'
        for c,v in cdic.items():
            neq=neq.replace(c,v)
#        print(neq)
    else:
        neq=''
        cdic={}
        
    if y.find('mce_equation'):
        meq = y.find('mce_equation/eviews_equation').text
        cmdic = {coeff.find('cf_name').text : coeff.find('cf_value').text for coeff in y.findall('mce_equation/coeff')}
        for c,v in cmdic.items():
            meq=meq.replace(c,v)
    else:
        meq=''
        cmdic={}
#        print(meq)
    return name,{'definition':definition, 'type':type , 'neq':neq, 'cdic':cdic, 'meq':meq}
#%%
    
def norm(var,eq):  
    '''normalize an equation with respect to var using sympy'''
    lhs, rhs = eq.split('=', 1)
    print(f'Normalize {var}')
        # we take the the $ out
    kat = sympify('Eq(' + lhs + ',' + rhs + ')')
    var_sym = sympify(var)
    try:
        out = str(solve(kat, var_sym,simplify=False,manual=True))[1:-1] # simplify takes for ever
        ret =f'{var} = {out}'
    except:
        out = 'Error '+var+' '+eq
        ret=out
        print(f'\nNormalize {var}\n {eq}')
        print('result\n',ret)
    return ret
norm('v','log(v)/33*x= 44+ x/2')    
#%%
def de_log(eq,funks=[]):
    ''' Explode dlog and diff in an expression'''
    udtryk=eq[:]
#    print(f'udtryk : {udtryk}')
    while  'DLOG(' in udtryk.upper():
        fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
#        udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
        udtryk=fordlog+'log(('+dlogudtryk+')/('+lagone(dlogudtryk+'',funks=funks)+'))'+efterdlog           
    while  'DIFF(' in udtryk.upper():
        fordif,difudtryk,efterdif=find_arg('diff',udtryk)
        udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif           
    return udtryk.lower()
de_log('dlog(a)')
de_log('dlog(a+b(-1))')
de_log('diff(a+b(-1))')
#%%
def de_movav(eq,funks=[]):
    ''' Explode the eviews function @movav , 3 periods as used in frb/us is assumed''' 
    udtryk=eq[:]
#    print(f'udtryk : {udtryk}')
    if  'MOVAV(' in udtryk.upper():
        formov,movudtryk,eftermov=find_arg('movav',udtryk)
        assert movudtryk[-2:]==',3',',3 is the only argument allowed to movav'
        movudtryk= movudtryk[:-2]  # easy to have other lengths but not nessecary 
        now = movudtryk
        one = lagone(now,funks=funks)
        two = lagone(one,funks=funks)
        udtryk=f'{formov}( (({now}) + ({one}) + ({two}))/3. ) {eftermov}'           
    return udtryk.lower()

de_movav('movav(a,3)')
de_movav('movav(a+b(-1),3)')
de_movav('+1.290723676327916*fxgap(-1)+-0.4680091148746248*fxgap(-2)+-0.05*(movav(frs10(-1)-(fpi10(-1)+fpi10(-2)+fpi10(-3)+fpi10(-4))/4,3)-frstar)+0.03734559019022718*xgap2(-1)+fxgap_aerr')
#%%
def normshift(udtryk,funks=[]):
    ''' Normalize (simplifying to one untransformed variable on the lhs) 
    an equation by shifting, 
    useful for simple lhs and much faster than sympy ''' 
    lhs,rhs=udtryk.split('=',1)
    
    if 'DLOG(' in lhs.upper():
        fordlog,dlogudtryk,efterdlog=find_arg('dlog',lhs)
        rhs='EXP(LOG('+lagone(dlogudtryk,funks=funks)+')+'+rhs+')'
        lhs=fordlog+dlogudtryk+efterdlog
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+'='+rhs # now the equation is normalized 
    elif 'LOG(' in lhs.upper(): 
        forlog,logudtryk,efterlog=find_arg('log',lhs)
        rhs='EXP('+rhs+')'
        lhs=forlog+logudtryk+efterlog
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+'='+rhs# now the equation is normalized 
    elif  'DIFF(' in lhs.upper(): 
        fordiff,diffudtryk,efterdiff=find_arg('diff',lhs)
        rhs=lagone(diffudtryk,funks=funks)+'+('+rhs+')'
        lhs=fordiff+diffudtryk+efterdiff
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+'='+rhs # now the equation is normalized 
#                    else:
#                        print('*** ERROR operand not allowed left of =',lhs)
#                        print(lhsterms)
    print(f'Norm hift {lhs}')
    return udtryk
normshift('dlog(b) = 0.1*f/6 ')
normshift('log(b) = 0.1*f/6 ')
normshift('diff(b) = -0.1*f/6 ')

if not 'ffrbusvar'   in locals(): 
    #%%    
    with open(r'..\frbus_package\frbus_package\mods\model.xml','rt') as t:
        tfrbus = t.read()
        
    rfrbus = xml.etree.ElementTree.fromstring(tfrbus)
    
    frbusdic = {name:inf for name,inf in (getinf(y) for y in rfrbus.iterfind('variable'))}
    firstm = [d['neq'] for v,d in frbusdic.items() if d['meq']]
    first = [d['neq'] for v,d in frbusdic.items() if d['neq']]
    vars =  [v for v,d in frbusdic.items() if d['neq']]
    [f for f in first if '@' in f]
    #%% look at d' on the lhs
    dcheck0 = [f.
        replace('d(log(','(dlog(').replace('@','') 
        for f in first]
    # liik at isolated d( -> diff( 
    dcheck00 = [re.sub(r'(^|[+\-*/\(\)])d\(',r'\g<1>diff(', f) 
                      for f in dcheck0] 
    dcheck = [f.replace(',0,1)',')') for f in dcheck00]    
    #%% move the error trm to the rhs
    
    lhs0 = [l.split('=',1)[0].replace('-'+var+'_aerr','') for var,l in zip(vars,dcheck) ]
    rhs0 = [l.split('=',1)[1]+'+'+var+'_aerr' for var,l in zip(vars,dcheck)  ] 
    rhs0 = [l.split('=',1)[1]+'+'+var+'_aerr' for var,l in zip(vars,dcheck)  ] 
    
    # de_log the tight hand side 
    rhs = [de_log(l) for l in rhs0  ] 
    rhs = [de_movav(l) for l in rhs]
    #%% expand the dlogs and diff's on the left hand
    lhscom = [de_log(l) if ('log'in l  or 'diff'in l) else l for l in lhs0 ]
    
    adjustedcom = [l+'='+r  for l,r in zip(lhscom,rhs)]  # A introduced because of a sympy bug
    
    def normalize(v,a,funks=[]):
        ''' Normalize an equation for a variable, sympy cnat handle recode, s0
        it is handled by the shifting ''' 
        if len(udtryk_parse(a.split('=',1)[0],funks=funks))< 2:
            out=a
            print(f'No normalization {v}')
        else:
            if 'recode'in a:
                out=normshift(a)
            else:
                out = norm(v,a)
                # 1.0  -> 1 if a lag 
                out = re.sub(namepat+r'(?:(\()([+-][0-9])+(.0)(\)))?',r'\g<1>\g<2>\g<3>\g<5>',out)
    #            out = re.sub(namepat+r'(?:(\()([+-][0-9])+(.0)(\)))?',r'\g<1>\g<2>\g<3>\g<5>',out)
    #            print(f'From:{a}\nTo  :{out}')
        return out
    
    rescom = [normalize(v,a)  for a,v in zip(adjustedcom,vars)]
    #also make truely additive adjustments 
    rescom2 = [f'{a} ' if '_aerr' in a else  f'{a} + {v+"_aerr"} ' for a,v in zip(rescom,vars)]
    
    ffrbusvar = '\n'.join(rescom2)
    
if 1:    
    mfrbus = ms.newmodel(ffrbusvar,funks=[recode])
    
    eqvar = [(e.split('=')[0].strip(),e) for e in rescom2]
    lffrbusmce = [f'{v} = {mcedic[v]}' if v in mcedic.keys() else e for v,e in eqvar]
    
    ffrbusmce = '\n'.join(lffrbusmce)

    #%%  get the data 
    
    frbusdhist = pd.read_csv(r'..\data_only_package\data_only_package\HISTDATA.TXT')
    dates = pd.period_range(start='1975q1',end='2030q4',freq='Q')
    frbusdf = pd.read_csv(r'..\data_only_package\example1solved.csv').pipe(lambda 
                         df:df.rename(columns={c:c.upper() for c in df.columns})).iloc[:-1,:]
    frbusdf.index = dates
    frbusdf = mc.insertModelVar(frbusdf,mfrbus)
    frbusdferr= frbusdf.loc[:,[c for c in frbusdf.columns if c.endswith('_AERR')]]
    
    avars    = [v.upper()+'_A' for v in vars]
    avarsdf =  frbusdf.loc[:,avars].copy(deep=True).pipe(lambda df: df.rename(columns= {c:c+'ERR' for c in df.columns})) 
    aerrvars = [v.upper()+'_AERR' for v in vars] 
    frbusdf.loc[:,aerrvars] = avarsdf 
    #%% get the mce data     
    longbase = pd.read_csv(r'..\data_only_package\data_only_package\eviews_database\longbase.csv').pipe(lambda 
                     df:df.rename(columns={c:c.upper() for c in df.columns})).iloc[:-1,:]
    dates = pd.period_range(start='1975q1',end='2125q4',freq='Q')
    longbase.index = dates  
#%%
if 0: # the model for forward looking 
    xx = mp.tofrml(ffrbusmce)
    mfrbusmce =  ms.newmodel(ffrbusmce,funks=[recode])
    #res = mfrbusmce.sim2d(longbase,'2020q1','2115q4',solveorder=mfrbusmce.strongorder,antal= 50,fair=10,silent=1)
    fairopt = {'fairconv':'XGDPN','fairantal':30}
    gausopt = {'conv':'XGDPN','antal':10,'alfa':0.4}
    res = mfrbusmce.sim2d(longbase,'2020q1','2047q4',solveorder=mfrbusmce.strongorder,
                          reverse=False,antal=40, fairopt=fairopt,silent=1,ldumpvar=1,conv='XGDPN')
    mfrbusmce.lastdf = res 
    mfrbusmce.basedf = longbase 
    check = mfrbusmce.dumpdf
    #%%
    mn.draw_adjacency_matrix(mfrbusmce.endograph,mfrbusmce.strongorder,mfrbusmce.strongblock,mfrbusmce.strongtype,size=(40,40))

assert 1==2    
#%%
#%% test
m2frbus =mc2.simmodel(ffrbusvar,funks=[recode])
m2frbus.findpos()
tt = m2frbus.outsolve3(chunk=80,ljit=1)
with open('solve.py','wt') as out:
    out.write(tt)
exec(tt,globals())
solvefunk=make(funks=[recode])
mfrbus.solve_jit = solvefunk


#%%
frbusdfsol = frbusdf.loc[:,sorted([v for v in mfrbus.allvar])]
mfrbus.superblock()
with mc.ttimer('Solve the model'):
    res = mfrbus.sim1d(frbusdfsol,'2020q1','2025q4',sim=True,antal= 500,
                 conv='XGDPN',first_test=10,ldumpvar=False,ljit=True,silent=False,stats=1,debug=0)
#%%
print(mfrbus.XGDPN.show)
print(frbusdf.XGDPN)
#mfrbus.dekomp('DPADJ')
#%%
if 0:
    def show(v):
        print(f'\n : {mfrbus.allvar[v]["frml"]}')
        print(mfrbus.get_eq_dif(v,showvar='True',filter=False))
    
    ''' for debugging each equation in solving order''' 
    for i,v in enumerate(mfrbus.solveorder):
        if i >70: break
        print(f'{v}')
        mtest = mc.model(rescom2[i],funks=[recode])
        mtestdf= mtest.res(frbusdf,'2020q1','2020q1')
        mtest.lastdf = mtestdf
        mtest.basedf = frbusdf
        d = mtest.get_eq_dif(v,filter=False,showvar=True)
        if len(d):
            print(f'{i}: {mtest.allvar[v]["frml"]}')
            print(f'{d} \n')
        #%%
if 0:
    #%%
    errlist = [v for v in mfrbus.exogene if v.endswith('_AERR')]
    mfrbus.graph_remove(errlist)
    _ = mfrbus.drawmodel()
    mn.draw_adjacency_matrix(mfrbus.endograph,mfrbus.strongorder,mfrbus.strongblock,mfrbus.strongtype,size=(40,40))
    mn.draw_adjacency_matrix(mfrbus.endograph,mfrbus.precoreepiorder,mfrbus._superstrongblock,mfrbus._superstrongtype,size=(8,8))
    mfrbus.FRSTAR.draw(endo=1,up=1,down=2)
    mfrbus.DMPTLUR.draw(up=1,down=2)
    mfrbus.DMPTR.draw(up=1,down=2)
    mfrbus.drawendo(cluster={'Preamble':mfrbus.prevar,'Epilog':mfrbus.epivar})
    
