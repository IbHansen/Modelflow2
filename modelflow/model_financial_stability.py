# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:59:39 2021

@author: bruger
"""

import pandas as pd
import numpy as np 

def lifetime_credit_loss(maturity,discount_rate,lgd,PDefault,debug=False):
    '''
    

    Parameters
    ----------
    maturity : integer or float
        maturity over which the exposure is amortised - by equal instalments.
    discount_rate : float
        discount rate 
    lgd : array of float 
        list of loss given default
    PDefault : array of float
        propability of defaults 
    debug : bool, optional
        calculate a intermidiately dataframes . The default is False.

    Returns
    -------
    float the long term credit loss in percent 
        

    '''
    startstock = 100
    _mat = float(int(maturity))
    time = list(range(int(_mat)+1))
    cashflow = pd.Series(float(startstock)/_mat,index=time,name='cashflow')
    accumulated_cashflow = cashflow.cumsum().shift(fill_value=0.0)
    accumulated_cashflow.name='accumulated_cashflow'
    
    stock = startstock-accumulated_cashflow
    stock.name = 'Outstanding'
    
    stock_lag = stock.shift().fillna(0.0)
    stock_lag.name = 'outstanding_lagged'
    
    stock_average = stock.rolling(2).mean().fillna(0)
    stock_average.name = 'Outstanding  average'
    
    assert cashflow.sum() != startstock,'Amortisation does not match outstanding '
    discount_series = pd.Series([1./(discount_rate+1)**t for t in time],index=time,name='discount')

    lgd_series = pd.Series(lgd[:len(cashflow)] if type(lgd) == np.ndarray else lgd,index = time,name='lgd')
    pd_series  = pd.Series(PDefault[:len(cashflow)]  if type(PDefault)  == np.ndarray else PDefault, index=time,name='pd')
    pd_series[0] = 0.0
    # breakpoint()
    
    pd_survival = 1-pd_series
    pd_survival.name = 'pd_survival'
    
    pd_incremental = pd_series*pd_survival.shift().cumprod() 
    pd_incremental.name = 'pd_incremental'
    
    lgd_decay = stock_average*lgd_series/100.
    lgd_decay.name = 'lgd_decay '
    
    lt_loss = pd_incremental* lgd_decay*stock_lag
    lt_loss.name='Loss'
    
    lt_loss_discounted = lt_loss*discount_series
    lt_loss_discounted.name = 'Discounted Loss'
    
    lt_loss_discounted_sum = lt_loss_discounted.sum()
     # breakpoint()
    if debug:
         debugout = pd.concat([stock,stock_average,pd_series,pd_survival,pd_incremental,
                          lgd_series,lgd_decay,lt_loss,discount_series,lt_loss_discounted],axis=1)
         print(f'Values:\n{debugout}')
    return     lt_loss_discounted_sum/startstock

if __name__ == '__main__':
    pd_fut = np.append([0],np.full(20,0.006831))
    lgd_fut = np.full(20,0.2)
    
    pd_ser = pd.Series(pd_fut)
    xx = lifetime_credit_loss(maturity=5,discount_rate=0.015,lgd=lgd_fut,PDefault=pd_fut,debug=1)
    yy = lifetime_credit_loss(maturity=5,discount_rate=0.015,lgd=0.2,    PDefault=0.006831,debug=1)
    zz = lifetime_credit_loss(maturity=5,discount_rate=0.015,lgd=0.2,    PDefault=pd_ser,debug=1)
