# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:47:37 2013


Developement Module - only for the adventeous 

 
This module handels symbolic differentiation of models \n
calculates the values of all the partial differentialkoifficients 
and creates matrices for each lag 

@author: Ib Hansen


"""
#from makrostart import smecstart,adamstart,monastart
import matplotlib.pyplot  as plt 
import re
from matplotlib import colors
from matplotlib import cm,patches
import itertools


import pandas as pd
from collections import defaultdict
from math import exp, log
import numpy as np
from sympy import sympify,Symbol
from sympy.abc import _clash1,_clash
from sympy import pprint
from IPython.display import display, Math ,Latex 
from sympy import latex
import re
import networkx as nx
import numpy as np 
import numpy.linalg as linalg
import sys

from modelmanipulation import split_frml,udtryk_parse,find_arg,stripstring,pastestring 
from modelclass import model 
from modelmanipulation import normalize
import modelclass as mc
from modelpattern import nterm,udtrykre


# in order to use theese reserved words from sympy as variables 
_clash['im']=Symbol('im')
_clash['li']=Symbol('li')
_clash['if']=Symbol('if')
_clash['IF']=Symbol('IF')
_clash['pi']=Symbol('pi')
_clash['PI']=Symbol('PI')
_clash['PCT']=Symbol('PCT')
_clash['pct']=Symbol('pct')
_clash['QP']=Symbol('QP')
_clash['qp']=Symbol('qp')
_clash['QQ']=Symbol('QQ')
_clash['qq']=Symbol('qq')
_clash['E']=Symbol('E')
_clash['e']=Symbol('e')
_clash['i']=Symbol('i')
_clash['I']=Symbol('I')
_clash['lt']=Symbol('lt')
_clash['LT']=Symbol('LT')
_clash['EX']=Symbol('EX')


#print(_clash)

def findallvar(model,v):
    '''Finds all endogenous variables which is on the right side of = in the expresion for variable v
    lagged variables are included '''
    terms= model.allvar[v]['terms'][model.allvar[v]['assigpos']:-1]
    rhsvar={(nt.var+('('+nt.lag+')' if nt.lag != '' else '')) for nt in terms if nt.var and nt.var in model.endogene}
    var2=sorted(list(rhsvar))
    return var2

def findendocur(model,v):
    '''Finds all endegenoujs variables which is on the right side of = in the expresion for variable v
    lagged variables are **not** included '''
    terms= model.allvar[v]['terms'][model.allvar[v]['assigpos']:-1]
    rhsvar={nt.var for nt in terms if nt.var and nt.var in model.endogene and nt.lag =='' and nt.var != v}
    var2=sorted(list(rhsvar))
    return var2
        
def modeldiff(model,silent=False,onlyendocur=False,endovar= None, maxdif=9999999999999999,forcenum=False):
    ''' Differentiate all relations with respect to all variable 
    The result is placed in a dictory in the model instanse: model.diffendocur
    '''
    with model.timer('Find espressions for partial derivatives',not silent):
        model.diffendocur={} #defaultdict(defaultdict) #here we wanmt to store the derivativs
        i=0
        for nvar,v in enumerate(model.endogene):
            if nvar >= maxdif:
                break 
            if not silent: 
                print(f'Now differentiating {v} {nvar}')
                
            if onlyendocur:
                endocur=findendocur(model,v)
            else: 
                endocur=findallvar(model,v)
            model.diffendocur[v]={}
            t=model.allvar[v]['frml'].upper()
            a,fr,n,udtryk=split_frml(t)
            udtryk=udtryk
            udtryk=re.sub(r'LOG\(','log(',udtryk) # sympy uses lover case for log and exp 
            udtryk=re.sub(r'EXP\(','exp(',udtryk)
            lhs,rhs=udtryk.split('=',1)
            try:
                kat=sympify(rhs[0:-1], _clash) # we take the the $ out _clash1 makes I is not taken as imiganary 
            except:
                print('* Problem sympify ',lhs,'=',rhs[0:-1])
            for rhv in endocur:
                try:
                    if not forcenum:
                        ud=kat.diff(sympify(rhv,_clash))
                    if forcenum or 'Derivative(' in str(ud) :
                        ud = numdif(model,v,rhv,silent=silent)
                        if not silent: print('numdif of {rhv}')
                    model.diffendocur[v.upper()][rhv.upper()]=str(ud)
    
                except:
                    print('we have a serous problem deriving:',lhs,'|',rhv,'\n',lhs,'=',rhs)
                i+=1
    if not silent:        
        print('Model                           :',model.name)
        print('Number of variables             :',len(model.allvar))
        print('Number of endogeneus variables  :',len(model.diffendocur))
        print('Number of derivatives           :',i) 
        
def diffout(model):
    l=model.maxnavlen
    llhs = max( len(lhsvar) for lhsvar in model.diffendocur)
    lrhs = max( len(rhsvar) for lhsvar in model.diffendocur 
               for rhsvar in model.diffendocur[lhsvar])
    out = [f'{lhsvar:>{llhs}} {rhsvar:>{lrhs}} {model.diffendocur[lhsvar][rhsvar]} '
            for lhsvar in sorted(model.diffendocur)
              for rhsvar in sorted(model.diffendocur[lhsvar])
            ] 
    return out 
 
       
def diffprint(model,udfil=''):
    f=sys.stdout
    if udfil:f=open(udfil,'w+')
    i=0
    l=model.maxnavlen
    for v in sorted(model.diffendocur):
        for e in sorted(model.diffendocur):
            print(v.ljust(l),e.ljust(l+5),model.diffendocur[v][e],file=f)
            i+=1
    print('Number of variables             :',len(model.allvar),file=f)
    print('Number of endogeneus variables  :',len(model.diffendocur),file=f)
    print('Number of derivatives           :',i,file=f) 
def vardiff(model,var='*'):
    ''' Displays espressions for differential koifficients for a variable
    if var ends with * all matchning variables are displayes'''
    l=model.maxnavlen
    if var.endswith('*'):
        for v in sorted(model.diffendocur):
            if v.startswith(var[:-1]):
                print(model.allvar[v]['frml'])
                for e in sorted(model.diffendocur[v]):
                    print(v.ljust(l),e.ljust(l+5),model.diffendocur[v][e])
                print(' ')
    else:
        for v in sorted(model.diffendocur):
            if v == var:
                print(model.allvar[v]['frml'])                
                for e in sorted(model.diffendocur[v]):
                    print(v.ljust(l),e.ljust(l+5),model.diffendocur[v][e])
def invdiff(model,var):
    ''' Displays espressions for differential koifficients for a variable
    if var ends with * all matchning variables are displayes'''
    l=model.maxnavlen
    for v in sorted(model.diffendocur):       
        for e in sorted(model.diffendocur[v]):
            if e == var:
                print(model.allvar[v]['frml'])
                print(v.ljust(l),e.ljust(l),model.diffendocur[v][e])
def rettet(ind):
    print(ind)
    print(ind.group(0))
    print(ind.group(1))

def fouteval(model,databank):
    ''' takes a dict of derivatives for a model and makes a function which returns a function which evaluates the 
    derivatives in a period.
    The derivatives is both returned from the function and places in \n
    :model.difvalue'''
    columnsnr=model.get_columnsnr(databank)
    fib=[]
    fib.append('def make_diffcalculate():\n')
    fib.append('  def diffcalculate(model,databank,periode):\n')
    fib.append('    '+'from math import exp, log \n')
    fib.append('    '+'import sys\n')
    fib.append('    '+'row=databank.index.get_loc(periode)\n')
    fib.append('    '+'values=databank.values\n')
    fib.append('    '+'diffvalue={} \n')   
    fib.append('    '+'try :\n')
    for i,v in enumerate(sorted(model.diffendocur)):
#        print(v)
        if i>= 20000000: # to make small tests calculations 
            break
        fib.append('        '+'diffvalue["'+v+'"]={} \n')        
        for e in sorted(model.diffendocur[v]):
#            print('   ',e)
            if e == 'D'+v or e == 'Z'+v :continue                
            if e == v+'_D' :continue     # dont want dummyes 
#            print(v.ljust(l),e.ljust(l),model.diffendocur[v][e],file=f)
            fib.append('        '+'diffvalue["'+v+'"]["'+e+'"]=')
            terms=udtryk_parse(settozero(str(model.diffendocur[v][e])))
            for t in terms:       
                if t.op:
                    ud=t.op.lower()
                elif t.number:
                    ud=t.number
                elif t.var:
                    if t.lag== '':
                        ud= 'values[row,'+str(columnsnr[t.var])+']'
                    else :  
                        ud= 'values[row'+t.lag+','+str(columnsnr[t.var])+']' 
                fib.append(ud)
            fib.append('\n')
    fib.append('    '+'except :\n')
    fib.append('    '+'   '+'print("Error in",sys.exc_info())\n')
    fib.append('    '+'   '+'raise\n')
    fib.append('    '+'model.diffvalue=diffvalue\n')
    fib.append('    '+'return diffvalue\n')
    fib.append('  return diffcalculate\n')
#    print('Function to evaluate derivatives of',model.name,'created')
#        print (fib)
    return ''.join(fib)
    
def calculate_diffvalue(model,bank,per):
    ''' calculates the numeric value of derivatives. 
    the values are returnes and also places in model.diffvalue'''
    funk=fouteval(model,bank)
    exec(funk,globals())
    xx=make_diffcalculate()
    res=xx(model,bank,per)
    return res
def calculate_delta(databank):
    ''' calculates the standard deviation of the change in all variable in a databank
    returns a panda series'''
    delta=databank.data['2000':'2013'].diff()
    return delta.std()     
def calculate_impact(model,bank):
    '''Calculate the impact of every variable in equation on the result based on 
    the standard deviation and differential coefficient'''
    impact={}
    temp=calculate_delta(bank)
    for v in sorted(model.diffvalue):
        impact[v]={}
        for e in sorted(model.diffvalue[v]):
            evar=e.split('(')[0] # evar= variable name without lag - to find the standard error 
            impact[v][e]=abs(model.diffvalue[v][e]*temp[evar])
    model.impact=impact
def calculate_diffvalue_d3d(model):
    """ creates a 3D dictonary derivatives for each lag"""
    tree = lambda: defaultdict(tree) # crates a nested dict 
    res=tree()
    for v in model.diffvalue:
        for e in model.diffvalue[v]:
            ee=e.split('(')
            evar=ee[0] # evar= variable name without lag - to find the standard error 
            lag=0 if 1==len(ee) else int(ee[1][:-1]) # extract the lag 
            d=model.diffvalue[v][e]
            if np.isinf(d) or np.isnan(d):
                print('Differentialkoifficienten mellen',v,'og',e,'kan ikke beregnes')
            res[lag][v][evar]=d
    model.diffvalue_d3d=res
    return res

def calculate_mat(model,lag=0):
    ''' calcultae matrix of derivative values.
    very slow should be reworked 
    
    '''
    rowname=sorted(model.endogene)
    colname=sorted(model.endogene) + sorted(model.exogene)
    ud=pd.DataFrame(0,index=rowname,columns=colname)
    
    for pcol in colname:
        for prow in rowname:
            try:
                ud.loc[prow,pcol] =  model.diffvalue_d3d[lag][prow][pcol]
            except:
                pass
    return ud

def calculate_endocurmat(model,df,per):
    ''' for Newton solution find jacobi'''
    with model.timer('Finding derivatives') as t: 
        _ = modeldiff(model,silent=True,onlyendocur =True)
    with model.timer('Calculating derivatives') as t: 
        res = calculate_diffvalue(model,df,per)
    df = pd.DataFrame.from_dict(madam.diffvalue)
    return df
    

def calculate_allmat(model,df,per,show=False):
    ''' Calculate and return a dictionary with a matrix of derivative values for each lag''' 
    with model.timer('Finding derivatives',show) as t: 
        _ = modeldiff(model,silent=True)
    with model.timer('Calculating derivatives',show) as t: 
        res = calculate_diffvalue(model,df,per)
    with model.timer('creating dataframe',show) as t: 
        res2 = calculate_diffvalue_d3d(model)
    return {l:calculate_mat(model,l) for l in range(0,model.maxlag-1,-1)}

def calculate_matold(model,lag=0,endo=True):
    ''' calcultae matrix of derivative values.
    endo deteriins if it is with respect to endogeneous og exogeneous variables
    '''
    rowname=sorted(model.endogene)
    colname=rowname if endo else sorted(model.exogene)
    col=len(colname)
    row=len(rowname)
    ud=np.zeros((row,col))
    placdirrow={v:i for i,v in enumerate(rowname)}
    placdircol={v:i for i,v in enumerate(colname)}
    for vx in [m for m in model.diffvalue_d3d[lag].keys() if m in colname]:
                for vy in [m for m in model.diffvalue_d3d[lag][vx].keys() if m in rowname]:
                    ud[placdirrow[vx],placdircol[vy]]=model.diffvalue_d3d[lag][vx][vy]
    return pd.DataFrame(ud,index=rowname,columns=colname) 
def modelnet_dict(d,model,lag):
    ''' creates a network where weight is determined by a 3d dict of impacts
    d: 3 d dictinorary '''
    g=nx.DiGraph()
    for lag in d:
        for v in d[lag]:
            for e in d[lag][v]:
                if v in model.endogene and e in model.endogene:
                    g.add_edge(e,v,weight=d[lag][v][e])
    return g
def pagerank(g):
    ''' ranks the equations in a model according to the pagerank algoritme
    returns order in pagerank'''
    xx=nx.pagerank(g)
    sorteret=sorted(zip(xx.values(),xx.keys()),reverse=True)
    rankorder=[yy for xxx,yy in sorteret]
    return rankorder 
def display_diff(model,bank,var=''):
    temp=calculate_delta(bank)
    l=model.maxnavlen
    for v in sorted(model.diffvalue):
        if var and not v==var:continue 
        f=re.sub(r' |\n','',split_frml(model.allvar[v]['frml'])[3])
        print(f)
        for e,w in sorted(model.impact[v].items(), key=lambda x: x[1]):
            evar=e.split('(')[0]
            diffvalue = '{:12.3f}'.format(model.diffvalue[v][e])
            delta     = '{:12.3f}'.format(temp[evar])
            timpact   = '{:12.1f}% '.format(model.impact[v][e]/temp[v]*100.)
            print(v.ljust(l),e.ljust(l),diffvalue,delta,timpact,model.diffendocur[v][e])  
def display_ip_old(model,ivar):
    var=ivar.upper()
    a,fr,n,udtryk= split_frml(model.allvar[var]['frml'])
    udtryk2=re.sub(r'_',r'\_',udtryk[:-1])  # get rid of _
    udtryk3=re.sub(r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>-[0-9]+)\)','\g<var>_{\g<lag>}',udtryk2) 
    udtryk3 = re.sub(r'delta_',r'\Delta ',udtryk3)
    totud=[udtryk3]
    for i in sorted(model.diffendocur[var].keys()):
        ud=r'\frac{\partial '+var+'}{\partial '+i+'}  = '+str(model.diffendocur[var][i])
        ud = re.sub(r'delta_',r'\Delta ',ud)

        ud=re.sub(r'_',r'\_',ud)
        ud2=re.sub(r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>-[0-9]+)\)','\g<var>_{\g<lag>}',ud)
        totud.append(r''+ud2+r'')
       # print(' ')
    ud=r'\\'.join(totud)
    ud2=re.sub(r'=',r' & = &',ud)
    ud3=r'$\begin{eqnarray}'+r' \\'+ud2+r'\end{eqnarray}$'
    return ud3

def display_all(model,df,per):
    res = calculate_diffvalue(model,df,per)
    res2 = calculate_diffvalue_d3d(model)
    for v in sorted(model.endogene):
        display_ip(model,v)

def display_ip(model,ivar):
    var=ivar.upper()
    a,fr,n,udtryk= split_frml(model.allvar[var]['frml'])
    udtryk2=re.sub(r'_',r'\_',udtryk[:-1])  # get rid of _
    udtryk3=re.sub(r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>-[0-9]+)\)','\g<var>_{\g<lag>}',udtryk2) 
    udtryk4=re.sub(r'=',r' & = &',udtryk3,1)
    totud=['Formula:'+udtryk4+r'\\']
    
    for i in sorted(model.diffendocur[var].keys()):
        ud=r'\frac{\partial '+var+'}{\partial '+i+'}  = '+str(model.diffendocur[var][i])
        ud=re.sub(r'_',r'\_',ud)
        ud2=re.sub(r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>-[0-9]+)\)','\g<var>_{\g<lag>}',ud)
        ud2=re.sub(r'=',r' & = &',ud2,1)

        totud.append(r''+ud2+r'')
        try:
            totud.append(f'& = & {model.diffvalue[var][i]}')
        except:
            pass
       # print(' ')
    ud=r'\\'.join(totud)
 #   display(Latex(r'\begin{align}'+r' \\'+ud+r'\end{align}')) 
    display(Latex(r'\begin{eqnarray*}'+ud+r'\end{eqnarray*} '))
  #  print(       (r'\begin{eqnarray}'+ud+r'\end{eqnarray} '))

    
def settozero(instring):
    ''' sets subst() or Derivative() to zero in string 
    the purpose is to get rid of derivatives of logical espressions 
    '''
    value=instring[:].upper()
    try:
        while 'SUBS(' in value:
            forsum, sumudtryk, eftersum = find_arg('SUBS', value)
            value = forsum + '0' + eftersum
        while 'DERIVATIVE(' in value:
            forsum, sumudtryk, eftersum = find_arg('DERIVATIVE', value)
            value = forsum + '0' + eftersum
    except:
        print('*** Error i settozero:',value)
    return value 

def get_A(model,df,per):
    jacobi = calculate_allmat(model,df,per)  # calculate derivative matrix for all lags
    names = sorted(model.endogene)
    A = {l:j.loc[names,names] for l,j in jacobi.items()}
    return A


def get_AINV(A):
    I=np.eye(A[0].shape[0])                                        # a idendity matrix
    AINV=linalg.inv(I-A[0])  
    out = pd.DataFrame(AINV,index=A[0].index,columns=A[0].columns)
    return out


def get_compagnion(model,df,per,show=False) :
    
    A = get_A(model,df,per)
    AINV=get_AINV(A)  
    names = AINV.columns#    print(A[0])                                          # inverse(A) 
#    print(AINV)                                          # inverse(A) 
    C={lag:np.dot(AINV,A[lag]) for lag in A.keys() if lag < 0}         # calculate A**-1*A(lag)
#    print(C) 
    dim=AINV.shape[0]
    lags=[lag for lag in sorted(C.keys(),reverse=True)]
#    print('Lags ',lags)
    number= len(lags) 
    top=np.eye((number-1)*dim,number*dim,dim)
    bottom=np.bmat([C[tlag] for tlag in sorted(lags) ])
#    print('Top shape   ',top.shape)
#    print('Bottom shape',bottom.shape)
    ib2=np.bmat([[top],[bottom]])
    newnames = [c+'('+str(l)+')' for l in sorted(lags,reverse=False) for c in names]       
    
    out =  pd.DataFrame(ib2,index=newnames,columns=newnames)
    # breakpoint()
    return out

def get_eigen(model,df,per) :
    comp = get_compagnion(model,df,per)
    compeig=np.linalg.eig(comp)       # find the eigenvalues
    return compeig
    
def eigplot0(w):
    fig = plt.figure(figsize=(8,9)) 
    ax = fig.add_subplot(2, 2,1 , projection='polar')
    ax.set_title('Eigenvalues',loc='left')
    for x in w:
        ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o',)
    ax.set_rticks([0.5, 1, 1.5])    
    langde =[np.abs(x) for x in w if np.abs(x)> -0.00000000000000000000000000000000000000001]
    return fig
#    ax = fig.add_subplot(2,2,  2   )# , projection='polar')
#    ax.hist(langde)
#    ax.set_title('Numbers')
def eigplot(w,size=(3,3)):
    fig, ax = plt.subplots(figsize=size,subplot_kw={'projection': 'polar'})  #A4 
    ax.set_title(' Eigenvalues',loc='right')
    for x in w:
        ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
    ax.set_rticks([0.5, 1, 1.5])    
    langde =[np.abs(x) for x in w if np.abs(x)> -0.00000000000000000000000000000000000000001]
    return fig
#    ax = fig.add_subplot(2,2,  2   )# , projection='polar')
#    ax.hist(langde)
#    ax.set_title('Numbers')
           
def stabilitet(model,minnumber=8,maxnumber=8):
    A={lag:modeldiff.calculate_mat(model,lag,endo=True) for lag in val3d.keys()} # calculate derivative matrix for all lags
    I=np.eye(np.shape(A[0])[0])                                        # a idendity matrix
    AINV=linalg.inv(I-A[0])                                            # inverse(A) 
    C={lag:np.dot(AINV,A[lag]) for lag in A.keys() if lag < 0}         # calculate A**-1*A(lag)
    dim=A[0].shape[0]
    lags=[lag for lag in sorted(C.keys(),reverse=True)]
#    print('Lags ',lags)
    eigenval={}
    eigenvec={}
    for number,lag in enumerate(lags):
        if minnumber <= number <= maxnumber:
            print(number,lag)
            top=np.eye(number*dim,(number+1)*dim,dim)
            bottom=np.bmat([C[tlag] for tlag in sorted(lags) if tlag >= lag])
            #print('Top shape   ',top.shape)
            #print('Bottom shape',bottom.shape)
            ib2=np.bmat([[top],[bottom]])
            zzz=np.linalg.eig(ib2)
            w,v=zzz  # w=egenværdien , v= egenvektorer
            eigenval[lag]=w
            eigenvec[lag]=v
    return eigenval,eigenvec

def tout(t):
    if t.lag:
        return f'{t.var}({t.lag})'
    return t.op+t.number+t.var

def numdif(model,v,rhv,delta = 0.005,silent=True) :
#        print('**',model.allvar[v]['terms']['frml'])
        def tout(t):
            if t.lag:
                return f'{t.var}({t.lag})'
            return t.op+t.number+t.var

               
        nt = model.allvar[v]['terms']
        assignpos = nt.index(model.aequalterm)                       # find the position of = 
        rhsterms = nt[assignpos+1:-1]
        vterm = udtryk_parse(rhv)[0]
        plusterm =  udtryk_parse(f'({rhv}+{delta/2})',funks=model.funks)
        minusterm = udtryk_parse(f'({rhv}-{delta/2})',funks=model.funks)
        plus  = itertools.chain.from_iterable([plusterm if t == vterm else [t] for t in rhsterms])
        minus = itertools.chain.from_iterable([minusterm if t == vterm else [t] for t in rhsterms])
        eplus  = f'({"".join(tout(t) for t in plus)})'
        eminus = f'({"".join(tout(t) for t in minus)})'
        expression = f'({eplus}-{eminus})/{delta}'
        if not silent:
            print(expression)
        return expression 
                
if __name__ == '__main__':
#%%
    def recode(condition,yes,no):
        '''Function which recratetes the functionality of @recode from eviews ''' 
        return yes if condition else no

    ftest = '''\
    a= b*2*c**3
    b = a(-1)
    d = recode(a>3,a(-1))
    c= (a>3)
     '''
    mtest = model(ftest,funks=[recode])   
    dmodel = diffmodel(mtest,forcenum=1)
    zz = dmodel.equations
    print(zz)
        
