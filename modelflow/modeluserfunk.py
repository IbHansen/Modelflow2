# -*- coding: utf-8 -*-
"""
To make userdefined function avaiable to Business logic define them here
Function names have to be all lower case !!!

Created on Fri Mar  2 14:50:18 2018

@author: hanseni
"""

try:
    from numba import jit
    
    @jit("f8(b1,f8,f8)",nopython=True)
    def recode(condition,yes,no):
        '''Function which recreates the functionality of @recode from eviews ''' 
        return yes if condition else no
except Exception as e:
    print(f' Import of numba failed {e} ')
    def recode(condition,yes,no):
        '''Function which recreates the functionality of @recode from eviews ''' 
        return yes if condition else no
    

try:
    from stem import ste 
except:
    pass 

def __pd():
    ''' Returns functions translating pd to REA veights. 
    
    The reason for making a closure is to avoid namespace clutter with the imported functions'''
    
    
    from math import isclose,sqrt,erf 
    from math import exp, log
    from scipy.special import erfinv , ndtri

    
    def phi(x):
        ''' Cumulative normal distribution function '''
        return (1.0 + erf(x / sqrt(2.0))) / 2.0

    def phiinv(x):
        ''' inverse Cumulative normal distribution function '''
        return ndtri(x) 

    
    
    def pd_to_w(PD=0.01,LGD=0.5,cat='mrtg'):
        ''' based on pd,lgd and sector this function calculates the risk weigts based on Basel 3
        this function is based on Marco Gross's matlab function and chekked against the results 
        the function is Risk_weights.m and a copy is located at 'Python poc' directory 
        This function only distinguis betwen 3 types,
        Alternative is to implement the parameters from BST.steet(CR_MAP) ''' 
        NORM99 = 3.0902323061678132
        # from phiinv(0.999)
        
        PD_ = 1e-10 if isclose(PD,0.0,abs_tol=1e-9)  else PD
        if PD < -1e-9:
            PD_ =  1e-10
    
        if cat == 'corp':
             R = 0.12*(1-exp(-50*PD_   ))/(1-exp(-50)) + 0.24* (1-(1-exp(-50*PD_   ))/(1-exp(-50)))
             b = (0.11852 - 0.05478*log(PD_))**2.
             M = 2.5;
             normal_dist_comp = phi(((1-R)**-0.5) *phiinv(PD_)   + NORM99 * ((R /(1-R))**0.5))
             K = LGD    *(normal_dist_comp-PD_   )  *(1+(b*(M-2.5))) /(1- b*1.5)
        elif cat == 'mrtg':
            R = 0.15;
            normal_dist_comp = phi(((1-R)**-0.5)*phiinv(PD_) + NORM99 *((R/(1-R))**0.5))
            K = LGD*(normal_dist_comp-PD)
        elif cat == 'retail':
            R = 0.03*(1-exp(-35*PD_))/(1-exp(-35)) + 0.16*(1-(1-exp(-35*PD_))/(1-exp(-35)));
            normal_dist_comp = phi(((1-R)**-0.5)*phiinv(PD_) + NORM99 * ((R/(1-R))**0.5))
            K = LGD*(normal_dist_comp-PD)
        else: 
            print('Major mistake. No Basel categori :',cat)
         
        return K * 12.5
    
    
    def pd_to_w_corp(PD=0.01,LGD=0.5):
        return pd_to_w(PD,LGD,cat='corp' )
    
    
    def pd_to_w_mrtg(PD=0.01,LGD=0.5):
        return pd_to_w(PD,LGD,cat='mrtg' )
    
    
    def pd_to_w_retail(PD=0.01,LGD=0.5):
        return pd_to_w(PD,LGD,cat='retail' )
    
    return pd_to_w,pd_to_w_corp,pd_to_w_mrtg,pd_to_w_retail,phi,phiinv

# pd_to_w,pd_to_w_corp,pd_to_w_mrtg,pd_to_w_retail,phi,phiinv = __pd()
