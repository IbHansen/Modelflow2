# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:01:49 2018

@author: hanseni

Functions placed here are included in the Pyfs business language 

"""
from math import exp, log, sqrt, tanh, erf
from numpy import transpose , array
from scipy.stats import norm,lognorm
from scipy.stats import gamma
import inspect 

try:
    # raise ImportError("Simulating ImportError for numba.")

    from numba import jit
except ImportError:
    print("Numba is not available. No worry")

    def jit(*args, **kwargs):
        def wrapper(func):
            # print('jit called')
            return func
        return wrapper
    
@jit("f8(f8)",nopython=True)
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

# @jit("f8(f8)")
# def logit_inverse(number):
#     ''' A function which returns the logit of a number 
    
#     takes care of extreme values 
#     '''
#     if number > 100:
#         return 1.0
#     elif number < -100:
#         return 0.0
#     else: 
#         return 1/(1+exp(-number))

def normcdf(input,mu=0.0,sigma=1.0):
    return norm.cdf(input,mu,sigma)    


def qgamma(q,a,loc):
    res = gamma.ppf(q,a,loc,scale=1)
    return res


def cdf_lognorm_econ(x, mu, sigma, eps=1e-12):
    """
    Econ / GAMS-style log-normal cumulative distribution function (Newton-safe).

    Computes the CDF of a log-normal distribution using the mean-parameterized
    (econ / GAMS) formulation, where `mu` is the mean of the level variable X,
    not the mean of log(X).

    Mathematically:
        F(x) = Φ((ln(x / mu) + 0.5 * sigma**2) / sigma)

    where Φ(·) is the standard normal CDF.

    This implementation is numerically safe for use in Newton solvers and
    numerical differentiation by applying a soft lower bound to x and using
    a stable normal CDF.

    Parameters
    ----------
    x : float
        Evaluation point. May be zero or negative during numerical
        differentiation.
    mu : float
        Mean of the level variable X (E[X] = mu).
    sigma : float
        Standard deviation of ln(X).
    eps : float, optional
        Small positive floor used to ensure log(x) is well-defined.
        Default is 1e-12.

    Returns
    -------
    float
        Value of the log-normal CDF in the interval [0, 1].

    Notes
    -----
    - The median of the distribution is mu * exp(-0.5 * sigma**2),
      not mu.
    - As x → 0, the function smoothly approaches 0 with a vanishing
      derivative, which is critical for Newton stability.
    - Equivalent to the standard log-mean parameterization with:
          mu_log = ln(mu) - 0.5 * sigma**2
    """
    from scipy.special import ndtr

    x_safe = x if x > eps else eps
    z = (log(x_safe / mu) + 0.5 * sigma**2) / sigma
    return float(ndtr(z))

def part_exp_lognorm(k, mu, std, eps=1e-12):
    """
    Safe scalar equivalent of the GAMS PartExpLogNorm function.

    Parameters
    ----------
    k : float
        Marginal input
    mu : float
        Average input (scale)
    std : float
        Dispersion / smoothing parameter
    eps : float, optional
        Small positive number to avoid log/division errors

    Returns
    -------
    float
        Smoothed, bounded transformation
    """
    # protect against division by zero
    mu_safe = mu if abs(mu) > eps else eps

    # protect against log(0)
    ratio = abs(k / mu_safe)
    ratio_safe = ratio if ratio > eps else eps

    # protect against std = 0
    std_safe = std if abs(std) > eps else eps

    return mu_safe * erf((log(ratio_safe) - 0.5 * std_safe**2) / std_safe)



def clognorm(input,mu=0.0,sigma=1.0):
    res = lognorm.cdf(input,mu,sigma)
    return res    

if __name__ == '__main__' and 1:
    xx = logit_inverse(-3*10
                      )
    print(xx)


