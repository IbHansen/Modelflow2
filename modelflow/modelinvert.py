# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:41:10 2017

@author: hanseni

    Class to handle general target/instrument problems. 
    Number of targets should be equal to number of instruments 
    an instrument can comprice of severeral variables 
    instruments are inputtet as a list of instruments

"""

import pandas as pd
import numpy as np 

from modelhelp import update_var


class targets_instruments():
    ''' Class to handle general target/instrument problems. 
    Number of targets should be equal to number of instruments 
    an instrument can comprice of severeral variables 
    instruments are inputtet as a list of instruments
    To calculate the jacobian each instrument variable has a impuls, which is used as delta when evaluating the jacobi matrix. 
    [ 'QO_J','TG']   Simple list each variable are shocked by the default impulse 
    [ ('QO_J',0.5), 'TG']  Here QO_J is getting its own impuls (0.5)
    [ [('QO_J',0.5),('ORLOV',1.)] , ('TG',0.01)] here an impuls is given for each variable, and the first instrument consiste of two variables 
    
    Targets are list of variables 
    convergence is achieved when all targets are within convergens distance from the target value
    convergencedistance can be set individual for a target variable by setting a value in <modelinstance>.targetconv 
    
    Targets and target values are provided by a dataframe. 
    
    '''
    
    def __init__(self,databank,targets,instruments,model,DefaultImpuls=0.01,defaultconv=0.01, 
                 nonlin=False,silent = True, maxiter=30,solveopt={}):

        self.model = model
        self.df = model.lastdf
        self.targetvars = targets if isinstance(targets,list) else targets.columns
        self.targetconv  = {t: defaultconv for t in self.targetvars} # make sure there is a convergence criteria ]
        self.targets    = targets
        self.instruments = {}
        self.solveopt = solveopt 
        self.silent = silent 
        self.debug=False
        self.maxiter = maxiter
        self.nonlin=nonlin
        self.databank = databank.copy()
        self.savesolvearg = model.oldkwargs if hasattr(model,'oldkwargs') else {}
        for inumber,i in enumerate(instruments): 
            vars =  i if isinstance(i,list) else [i] # make it a list even if one variable
            xx =   [v if isinstance(v,tuple) else (v,DefaultImpuls) for v in vars] # make sure there is a impuls 
            name = ','.join(n for n,i in xx)
            name = f'Instrument_{inumber}' if len(name)>500 else name 
            self.instruments[inumber] = {'name':name , 'vars': xx,  }
        self.DefaultImpuls = DefaultImpuls
        
    def jacobi(self,per):
        ''' Calculates a jecobi matrix of derivatives based on the instruments and targets 
        
        returns a dataframe '''
        
        self.jac = jac=pd.DataFrame(0,index=self.targetvars, columns=[v['name'] for v in self.instruments.values()])
        mul = self.model(self.df,per ,per ,setlast=False,silent=self.silent, **self.solveopt)           # start point for this quarter 
        basis = mul.copy(deep=True)  # make a reference point for the calculation the derivatives. 
        if not self.silent: print(f'Update jacobi: {per}')
        for instrument in self.instruments.values():
     #       print('instrument: ',instrument['name'])
# set the instrument            
            for var,impuls in instrument['vars']:
                mul.loc[per,var]    =     mul.loc[per,var] + impuls                      # increase loan growth
# calculate the effect 
            opt = {'antal':600,'first_test':20,'ljit':0}
            res = self.model(mul,per ,per ,setlast=False,silent=self.silent, **self.solveopt) #antal=600,first_test=20,ljit=1)        # solve model 
            
            jac.loc[self.targetvars,instrument['name']] = res.loc[per,self.targetvars]-basis.loc[per,self.targetvars] # store difference in original bank 
# reset the instrument 
            for var,impuls in instrument['vars']:
                mul.loc[per,var]=basis.loc[per,var]
        self.model.oldkwargs = self.savesolvearg 

        return jac
    
    def invjacobi(self,per,diag=False):
        ''' Calculates the inverted jacobi matrix
        
        returns a dataframe '''
        
        x = self.jacobi(per)
#        print(x)
        if diag:
            out = pd.DataFrame(np.diag(1.0/x.values.diagonal()),x.columns,x.index)
        else:    
            out= pd.DataFrame(np.linalg.inv(x),x.columns,x.index)
        return out
    
    def targetseek(self,databank=None,shortfall=False,**kwargs):
        ''' Calculates the instruments as a function of targets '''
        silent =  kwargs.get('silent',self.silent)
        self.maxiter = kwargs.get('maxiter',self.maxiter)
        self.nonlin = kwargs.get('nonlin',self.nonlin)
        tindex = self.model.current_per.copy()
        res    = self.databank 
#        self.inv  = inv  = self.invjacobi(self.targets.index[0])
        self.inv  = inv  = self.invjacobi(self.targets.index[0],diag=shortfall) 

        self.conv = pd.Series([self.targetconv[v] for v in self.targetvars],self.targetvars)
#        print(inv)
        for per in self.targets.index:
            if not silent: print('Period:',per)
            res = self.model(res,per ,per ,setlast=False,silent=self.silent, **self.solveopt)
            orgdistance = self.targets.loc[per,self.targetvars] - res.loc[per,self.targetvars]
            shortfallvar = (orgdistance >= 0)
            for iterations in range(self.maxiter):
                if not silent: print('Period:',per,' Target instrument iteration:',iterations)
                startdistance = self.targets.loc[per,self.targetvars] - res.loc[per,self.targetvars]
                self.distance = distance = startdistance*shortfallvar if shortfall else startdistance
                if self.debug: print(f'Distance    :{startdistance}\nOrgDistance :{distance}')
                if (distance.abs()>=self.conv).any():
                    if self.nonlin:
                         self.inv  = inv  = self.invjacobi(per,diag=shortfall)
                    update = inv.dot(distance)
                    if self.debug : print(update)
                    for instrument in self.instruments.values():
                        for var,impuls in instrument['vars']:
                            res.loc[per,var]    =   res.loc[per,var] + update[instrument['name']] * impuls  # increase loan growth
                    res = self.model(res,per ,per ,setlast=False,silent=self.silent, **self.solveopt)
                else:
                    break
        self.model.lastdf = res
        self.model.oldkwargs = self.savesolvearg 
        self.model.current_per = tindex

        return res
    
   
    def __call__(self, *args, **kwargs ):
        return self.targetseek( *args, **kwargs)
        
            
#%%
if __name__ == '__main__':
    import modelclass as mc
    try:
        # If not already there, get the model and make it into an modelinstance 
        if 'mmonas' not in locals(): 
            with open(r"J:\Udvikling - feedback\Systemfiler\super.fru", "r") as text_file:
                fmonas = text_file.read()
            mmonas  = mc.model(fmonas)   
        # get the baseline  
        grund   = pd.read_pickle(r'..\Udvikling - feedback\Systemfiler\supergrund.pc')           
        
        #%% Make a copy of the data 
        mul = grund.copy(deep=True) 
        
        #%% solve the model. Setbase make sure this is the reference run  when comparing 
        yy=mmonas(mul,start='2015q1',slut='2017q4',
            first_test=20,conv='FY',silent=True,antal=100,ljit=True,setbase=True)
        # Define instruments and targets 
        instruments = [ [('QO_J',0.5),('ORLOV',1.)] , ('TG',0.01)]
        target      = mmonas.lastdf.loc['2015q3':,['UL','ENL']]+[-2,10]
        
        tt          = targets_instruments(target,instruments,mmonas)
        tt.debug=1
        result      = tt(silent=False,shortfall=1)
        
        _ = mmonas.vis(['UL','ENL']).dif.plot()
        _ = mmonas.vis(['QO_j','ORLOV']).dif.plot()
        _ = mmonas.vis(['TG']).dif.plot()
#%%    
    except:
        print('Could not find model')