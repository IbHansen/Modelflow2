# -*- coding: utf-8 -*-
"""
Created on Mon May 26 21:11:18 2014

@author: Ib Hansen

A good explanation of quadradic programming in cvxopt is in 
http://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

This exampel calculates the efficient forntier in a small example 
the example is based on a mean variance model for Indonesian Rupia running in Excel 

"""

import numpy as np
import pandas as pd
from cvxopt         import matrix,  spdiag
from cvxopt.solvers import qp    ,  options 

"""

"""
def MV_test(lprint=True):
    P= matrix([
    [0.01573635,	0.01436816,	0.01045556],
    [0.01436816,	0.02289016,	0.01172995],
    [0.01045556,	0.01172995,	0.01748076]])   # the covariance matrix

    q       =  matrix([0.048,0.040,0.035])      # return vector 
    bsum=1.0
    
    wsum1=20.   # weighted sum should be less than: 
    weights1=matrix([2.5 , 1 , 1 ] ,(3,1))     
    
    wsum2= 1    # weightet sum should be greater than: 
    weights2=matrix([0.2 , 2.4 , 1 ] ,(3,1))     
    
    
    hmin    = -matrix([0. , 0 , 0])
    
    hmax    =  matrix([1. ,1. , 1.])
    
    
    options['show_progress'] = False
    
    riskaversions = [r/100. for r in range(101)]             # compute 100 points on the efficient frontier 
    portefolios   = [mv_opt(P,q,riskaversion,bsum,[[weights1],[-weights2]],[wsum1,-wsum2],hmin,hmax) for riskaversion in riskaversions] # minimize risk and maximize return 
    p_return      = [100  * x.T * q           for x in portefolios]
    risk          = [100 *( x.T * P *x)**0.5  for x in portefolios]
    res           = [list(r)+list(p)+list(x)  for r,p,x in zip(risk,p_return,portefolios) ]  # a row in the Dataframe 

    columns=['risk','return']+['Asset'+str(i)  for i,temp in enumerate(q)]   # to handle a number of assets 
    results=pd.DataFrame(res,columns=columns)                 # create an empty pandas.Dataframe 
    return results     
    
def mv_opt(PP,qq,riskaversion,bsum,weights,weigthtedsum,boundsmin,boundsmax,lprint=False,solget=None):
    ''' Performs mean variance optimization by calling a quadratic optimization function from the cvxopt \n
    library 
    
    '''
    q_size   =  len(qq)
    P        =  matrix(2.0*(1.0-riskaversion)*PP)                 # to 
    q        =  matrix(-1.0*riskaversion*qq)
    Gmin     = -matrix(np.eye(q_size))
    hmin     = -matrix(boundsmin)
    Gmax     =  matrix(np.eye(q_size))
    hmax     =  matrix(boundsmax)
    Gweights =  matrix(weights)
    hweights =  matrix(weigthtedsum)
    G        =  matrix([Gmin,Gmax,Gweights.T])                       # creates the combined inequalities 
    h        =  matrix([hmin,hmax,hweights])
    A        =  matrix(1.,(1,q_size)) if bsum else None                                # sum of shares equal to bsum 
    b        =  matrix([bsum])        if bsum else None   
    options['show_progress'] = False
    options['refinement']=10
    sol      = qp(P,q,G,h,A,b) 
    if solget:
        return sol
    else:                                     # minimize risk and maximize return 
        x        = sol['x'] 
        res      = x
        return res                                                       # get the solution 

def mv_opt_bs(msigma,vreturn,riskaversion,budget,risk_weights,capital,lcr_weights,lcr,leverage_weights,equity,boundsmin,boundsmax,lprint=False,solget=None):
    '''performs balance sheet optimization
    '''
    res = mv_opt(msigma,vreturn,riskaversion,budget,[risk_weights,-lcr_weights,leverage_weights],[capital,-lcr,equity],boundsmin,boundsmax,lprint=False,solget=None)
    return res
    

def mv_opt_prop(PP,qq,riskaversion,bsum,weights,weigthtedsum,boundsmin,boundsmax,probability=None,lprint=False):
    ''' select a numner of assets/liabilities which. when the selection is feasible an Mean variance optimazation is performed\n
    the selection is based on probabilities '''
    q_size           =  len(qq)
    selectsize       = 2 
    newboundsmax     = matrix(0.,(q_size,1))
    prop             = list(probability/sum(probability)) if probability else [1./q_size for i in range(q_size)]
# Find a feasible set of banks  
    while selectsize < q_size : 
        selected               = [ int(i) for i in np.random.choice(q_size,selectsize,replace=False,p=prop)] # select banks 
        selectvector           =  matrix(0.,(q_size,1))
        selectvector[selected] = 1.0                                                      # the selected banks is marked by 1.1
        newboundsmax           =  matrix([s*bm for s,bm in zip(selectvector,boundsmax)])  # elementwise multiplication, so max=0 if the bank is not selected 
#        print(sum(newboundsmax))
        if  sum(newboundsmax) >= bsum and [weigthtedsum] > list(newboundsmax.T*weights):
            break
        selectsize=selectsize+1
    else: 
        print('*** Constraints to do not allow a solution')
        raise
# now optimize 
    try:
        sol= mv_opt(PP,qq,riskaversion,bsum,weights,weigthtedsum,boundsmin,newboundsmax,lprint=False,solget=True)
        shares= sol['x']
    except:
        print('** The Mean variance problem can not be solved')
        raise
    return shares         
 
if __name__ == '__main__': 
    ib=MV_test()
    import matplotlib.pyplot as plt
#    pd.DataFrame.plot(ib,x='risk',y='return')
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))
    ib.plot(x='risk',y=['return'],kind='area'                  ,ax=axes[0])
    ib.plot(x='risk',y=['Asset0','Asset1','Asset2'],ax=axes[1],kind='line')
    ib.plot(x='risk',y=['Asset0','Asset1','Asset2'],ax=axes[2],kind='area')

    
        