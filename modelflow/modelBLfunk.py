# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:01:49 2018

@author: hanseni

Functions placed here are included in the Pyfs business language 

"""
from math import exp, log, sqrt
from numpy import transpose , array
from scipy.stats import norm,lognorm
from scipy.stats import gamma
import inspect 
from numba import jit
classfunk = []
try:
    from cvxopt import matrix
    from model_cvx import mv_opt, mv_opt_prop
    classfunk = ['TRANS']    # names a classfunk which can be called 
except:
    print('ModelFlow info: CVXopt not installed. Only matters if you are incorporating optimization')
    pass

from numpy import array

from model_financial_stability import lifetime_credit_loss

def sum_excel(*arg):
    ''' a functions which sums the arguments used in models franslated from excel 
    '''
    return sum(arg)
    
def logit(number):
    ''' A function which returns the logit of a number 
    '''
    return(-log(1.0/number-1.0)) 

@jit("f8(f8)",fastmath=True,cache=True)
def logit_inverse(number):
    ''' A function which returns the logit of a number 
    
    takes care of extreme values 
    '''
    if number > 100:
        return 1.0
    elif number < -100:
        return 0.0
    else: 
        return 1/(1+exp(-number))

def normcdf(input,mu=0.0,sigma=1.0):
    return norm.cdf(input,mu,sigma)    


def qgamma(q,a,loc):
    res = gamma.ppf(q,a,loc,scale=1)
    return res



def clognorm(input,mu=0.0,sigma=1.0):
    res = lognorm.cdf(input,mu,sigma)
    return res    

if __name__ == '__main__' and 1:
    xx = logit_inverse(3)
    print(xx)


