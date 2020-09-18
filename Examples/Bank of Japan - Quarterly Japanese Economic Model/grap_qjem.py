# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:46:18 2019

@author: ibh

This script processes the J-QEM model. 
the model is normalized, the coifficents are imterpolated, the data is loaded
"""

import pandas as pd 
import re
from sympy import sympify, solve
import glob


from modelclass import model 
import modelclass as mc
import modelmf 
import modelmanipulation as mp 
from modelmanipulation import find_arg, lagone,udtryk_parse,lag_n
import modelpattern as pt
from modelpattern import udtryk_parse, namepat
import modelnet as mn

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
    while  '@PC(' in udtryk.upper():
        forpc,pcudtryk,efterpc=find_arg('@PC',udtryk)
        udtryk=forpc+f'((({pcudtryk})/({lagone(pcudtryk,funks=funks)})-1.)*100.)'+efterpc           

    return udtryk.upper()
de_log('dlog(a)')
de_log('dlog(a+b(-1))')
de_log('diff(a+b(-1))')

#%%
def de_movav(eq,funks=[]):
    ''' Explode the eviews function @movav , 4 periods as used in wb/saudi is assumed''' 
    udtryk=eq[:].upper()
#    print(f'udtryk : {udtryk}')
    while  '@MOVAV(' in udtryk:
        formov,movudtryk,eftermov=find_arg('@MOVAV',udtryk)
        avrnr = int(movudtryk.rsplit(',',1)[-1])
        movudtryk= movudtryk.rsplit(',',1)[0]  # easy to have other lengths but not nessecary 
        mov = '+'.join([lag_n(movudtryk,n) for n in range(avrnr)])
        udtryk=f'{formov}( ({mov})/ {str(avrnr)}) {eftermov}'           
    return udtryk

de_movav('SSWAGE  = @MOVAV(SAUGGEXPCOEPCN+a(-4)  , 4)') # a test
de_movav('SSWAGE  = @MOVAV(SAUGGEXPCOEPCN+22  , 2)') # a test
de_movav('SSWAGE  = @MOVAV(SAUGGEXPCOEPCN+22  , 8)') # a test
def de_movsum(eq,funks=[]):
    ''' Explode the eviews function @movsum''' 
    udtryk=eq[:]
#    print(f'udtryk : {udtryk}')
    while  '@MOVSUM(' in udtryk:
        formov,movudtryk,eftermov=find_arg('@MOVSUM',udtryk)
        avrnr = int(movudtryk.rsplit(',',1)[-1])
        movudtryk= movudtryk.rsplit(',',1)[0]  # easy to have other lengths but not nessecary 
        mov = '+'.join([lag_n(movudtryk,n) for n in range(avrnr)])
        udtryk=f'{formov}({mov}){eftermov}'           
    return udtryk


de_movsum('SSWAGE  = @MOVSUM(SAUGGEXPCOEPCN+22  , 4)') # a test
de_movsum('SSWAGE  = @MOVSUM(a+1,2) +@MOVSUM(SAUGGEXPCOEPCN+22,4) ') # a test
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
        udtryk=lhs+'='+rhs # now the equation is normalized 
    elif 'LOG(' in lhs.upper(): 
        forlog,logudtryk,efterlog=find_arg('log',lhs)
        rhs='EXP('+rhs+')'
        lhs=forlog+logudtryk+efterlog
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+' = '+rhs# now the equation is normalized 
    elif  'DIFF(' in lhs.upper(): 
        fordiff,diffudtryk,efterdiff=find_arg('diff',lhs)
        rhs=lagone(diffudtryk,funks=funks)+'+('+rhs+')'
        lhs=fordiff+diffudtryk+efterdiff
        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+' = '+rhs # now the equation is normalized 
    elif  '@PC(' in lhs.upper(): 
        forpc,pcudtryk,efterpc=find_arg('@PC',lhs)
        # var  = (rhs / 100 +1) * var(-1) 
        rhs=f'(({rhs})/100. +1.) *( {lagone(pcudtryk,funks=funks)}) '
        lhs=forpc+pcudtryk+efterpc
#        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
#        lhs=lhs[1:-1] if lhs.startswith('(') and lhs.endswith(')') else lhs
        udtryk=lhs+' = '+rhs # now the equation is normalized 

#                        print(lhsterms)
#    print(f'Norm shift {lhs}')
    return udtryk
# some examples 
def answer(s):
    return 42

# some tests 
normshift('dlog(b) = 0.1*f/6 ')
normshift('log(b) = 0.1*f/6 ')
normshift('diff(b) = -0.1*f/6 ')
normshift('@pc(b) = -0.1*f/6 ')
normshift('@pc(b) = @pc(c) ')
normshift('frml z b = @pc(-0.1*f/6 +answer(3+ff(-3))) $',funks=[answer])
normshift('frml z b = xxpc(-0.1*f/6 +answer(3+ff(-3))) $',funks=[answer])

#%%
def norm(var,eq):  
    '''normalize an equation with respect to var using sympy'''
    lhs, rhs = eq.split('=', 1)
#    print(f'Normalize {var}')
        # we take the the $ out
    kat = sympify('Eq(' + lhs + ',' + rhs + ')')
    var_sym = sympify(var)
    try:
        out = str(solve(kat, var_sym,simplify=False,manual=False))[1:-1] # simplify takes for ever
        out2 = re.sub(namepat+r'(?:(\()([+-][0-9])+(.0)(\)))?',r'\g<1>\g<2>\g<3>\g<5>',out)
        ret =f'{var} = {out2}'
    except:
        out = 'Error '+var+' '+eq
        ret=out
        print(f'\nNormalize {var}\n {eq}')
        print('result\n',ret)
    return ret

norm('v','log(v)-log(v(-1)) = a+d(-1)')
norm('y','log(y) = log(y(-1)) + alpha +b1*log(x)+b2*log(x2)')
norm('y','log(y)=log(y(-1)) + alpha +b1*log(x)')
#%%
def normalize(a,funks=[]):
    ''' Normalize an equation for a variable, sympy cnat handle recode, s0
    it is handled by the shifting ''' 
    b = de_log(a,funks=funks)
    lhs,rhs = a.split('=',1)
    terms = udtryk_parse(lhs)
    if len(terms)< 2:
        v = terms[0]
        out=a
#        print(f'No normalization {lhs}')
    elif  (len(terms) == 4 and ('DLOG' in lhs.upper() or 'DIFF' in lhs.upper()  or 'LOG' in lhs.upper())) \
       or (len(terms) == 5 and  '@PC' in lhs.upper()) :
#        print(f'Shift normalization {lhs}')
        out = normshift(a)
    else:

        # Get the first variable name
        v = [ t.var for t in terms if t.var][0]     
        print(f'\nSymbolic normalization {v} for \n{lhs} = {rhs}')
        print(f'\nSymbolic normalization {v} for \n{de_log(lhs)} = {rhs}')
        b2=de_log(b)
        out = norm(v.lower(),b2.lower())
        print(f'\nResult : {out}\n')
            # 1.0  -> 1 if a lag 
        out = re.sub(namepat+r'(?:(\()([+-][0-9])+(.0)(\)))?',r'\g<1>\g<2>\g<3>\g<5>',out)
#            out = re.sub(namepat+r'(?:(\()([+-][0-9])+(.0)(\)))?',r'\g<1>\g<2>\g<3>\g<5>',out)
#            print(f'From:{a}\nTo  :{out}')

    return out
normalize('dlog(v)  = a+d(-1)')
normalize('diff(v)  = a+d(-1)')
normalize('@pc(v)  = a+d(-1)')
 #%%   



filepath = 'qjem_replication/input/'

# Read dateserise, has been exported from eviews with file/saveas/ csv file 
    
dfqjem0 = pd.read_csv(filepath+'basecase.csv')
#pidx = pd.PeriodIndex(start ='2000-1-1 08:45 ', end ='2009-10-01 8:55', freq ='Q')
dates = dates = pd.period_range(start='2000q1',end='2009q4',freq='Q')
dfqjem0.index = dates

dfqjem = dfqjem0.drop('dateid01',axis=1).pipe(lambda df: df.rename(columns={c:c.upper() for c in df.columns}))
#%%
tqjem0 =  open(filepath+'qjem_plain.txt').read().upper().replace('D(LOG(','DLOG((').replace('@ABS(','ABS(').replace('^','**')
tqjem = re.sub(r'([^A-Z]|^)D\(', r'\1DIFF(',tqjem0)
# get dictionary of endogen : (lefthand side, righthand side)
dqjem = {l.split(':')[0] : l.split(':')[-1].split('=')  for l in tqjem.split('\n') if l.split(':')[0] !=''}

eviews_functions = {c for c in re.findall(r'@[a-zA-Z]*',tqjem)}
diff_eq = dspecial = {k:v for k,v in dqjem.items() if 'DIFF' in v[0] or 'DIFF' in v[1] or 'DIFF' in k}
dunnormalized = {k:v for k,v in dqjem.items() if k!=v[0]}
dspecial = {k:v for k,v in dqjem.items() if '@' in v[0] or '@' in v[1] or '@' in k}
#%%
dqjem1 = {k:(v[0],de_movav(de_movsum(v[1]))) for k,v in dqjem.items()}
dspecial1 = {k:v for k,v in dqjem1.items() if '@' in v[0] or '@' in v[1] or '@' in k}

dqjem2 = {k:(v[0],de_log(v[1])) for k,v in dqjem1.items()}
dunnormalized2 = {k:v for k,v in dqjem2.items() if k!=v[0]}

dlines = {k : f'{v[0]} = {v[1]}' for k,v in dqjem2.items()} 

dtest = {k : normalize(v) for k,v in dlines.items() }
fqjem0 = '\n'.join([l for k,l in dtest.items()])


#%% all parameters has been written to files in folder coeff/
paramfilelist = [(f.rsplit('\\')[1], open(f,"rt").read().strip().split("\n")) for f in glob.glob(filepath+r"coeff/c*")]
dparam = {c : {f'{c}({i+1})' : v  for i,v in enumerate(vlist)}  for c,vlist in paramfilelist}
lparam = [(var,value) for k,v in dparam.items() for var,value in v.items()]
#%%
fqjem = fqjem0[:].upper().replace('(1.0)','(1)')
for i,(var,value) in enumerate(lparam):
    if i > 2000:break 
    if not i%40:
        print(f'Now {i} of {len(lparam)} parameters replaced ')
    fqjem = fqjem.replace(var,value)
print('Paremeters all replaced')    
mqjem = model(fqjem)
#%%
if 0:
    with open('model/fqjem.frm','wt') as f: 
        f.write(fqjem)
    dfqjem.to_pickle('data/dfqjem.pk')    

#%%
with model.timer('Solve model'):
    res = mqjem(dfqjem,max_iterations=50,first_test = 40,ljit=0) 
    res2 = mqjem.sim2d(res,max_iterations=50,first_test = 40,ljit=1) 
    mqjem.basedf=res
    mqjem.lastdf = res2
mqjem.modeldump('qjem.pcim')
#%%
if 0:
    fig1 = mn.draw_adjacency_matrix(mqjem.endograph,mqjem.strongorder,mqjem.strongblock,mqjem.strongtype,size=(40,40))
    fig2 = mn.draw_adjacency_matrix(mqjem.endograph,mqjem.precoreepiorder,mqjem._superstrongblock,mqjem._superstrongtype,size=(10,10))
    fig2.savefig('qjem2.pdf')       

newqjem,newbase = model.modelload('qjem.pcim',run=1)
