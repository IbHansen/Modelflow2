# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:38:28 2017

@author: hanseni

utilities for Stampe models 

"""


import networkx as nx
import pandas as pd
import numpy as np
import time
from contextlib import contextmanager
import sys
import itertools
import operator as op




def update_var(databank,xvar,operator='=',inputval=0,start='',slut='',create=1, lprint=False,scale=1.0):
    
        """Updates a variable in the databank. Possible update choices are: 
        \n \= : val = inputval 
        \n \+ : val = val + inputval 
        \n \- : val = val - inputval 
        \n \* : val = val * inputval 
        \n \^  : val = val(t-1)+inputval +
        \n \% : val = val(1+inputval/100)
        \n 
        \n scale scales the input variables default =1.0 
        
        """ 
        import sys 
        import numpy as np 
        var = xvar.upper()
        if var not in databank: 
            if not create:
                print('** Error, variable not found:',var)
                print('** Update =',var,'Data=',inputval)
                print('Create=True if you want to create the variable in the databank')
                sys.exit()
            else:
                if 0:
                    print('Variable not in databank, created ',var)
                databank[var]=0.0

        current_per = databank.index[databank.index.get_loc(start):databank.index.get_loc(slut)]
        orgdata=pd.Series(databank.loc[current_per,var]).copy(deep=True)
        antalper=len(current_per)
        # breakpoint()
        if isinstance(inputval,float) or isinstance(inputval,int) :
            inputliste=[float(inputval)]
        elif isinstance(inputval,str):
            inputliste=[float(i) for i in inputval.split()]
        elif isinstance(inputval,list):
            inputliste= [float(i) for i in inputval]
        elif isinstance(inputval, pd.Series):
#            inputliste= inputval.base
            inputliste= list(inputval)   #Ib for at håndtere mulitindex serier
        else:
            print('Fejl i inputdata',type(inputval))
        inputdata=inputliste*antalper if len(inputliste) == 1 else inputliste 

        if len(inputdata) != antalper :
            print('** Error, There should be',antalper,'values. There is:',len(inputdata))
            print('** Update =',var,'Data=',inputdata,start,slut)
            sys.exit()
        else:     
            inputserie=pd.Series(inputdata,current_per)*scale            
#            print(' Variabel------>',var)
#            print( databank[var])
            if operator=='=': #changes value to input value
                outputserie=inputserie
            elif operator == '+':                
                outputserie=orgdata+inputserie
            elif operator == '*':
                outputserie=orgdata*inputserie
            elif operator == '%':
                outputserie=orgdata*(1.0+inputserie/100.0)
            elif operator == '^': # data=data(-1)+inputdata 
                ilocrow =databank.index.get_loc(start)-1
                iloccol = databank.columns.get_loc(var)
                temp=databank.iloc[ilocrow,iloccol]
                opdater=[temp+sum(inputdata[:i+1]) for i in range(len(inputdata))]
                outputserie=pd.Series(opdater,current_per) 
            elif operator.upper() == '=GROWTH': # data=data(-1)+inputdata 
                ilocrow =databank.index.get_loc(start)-1
                iloccol = databank.columns.get_loc(var)
                temp=databank.iloc[ilocrow,iloccol]
                factor = list(itertools.accumulate([(1+i/100) for i in inputdata],op.mul))
                opdater=[temp * it for it in factor]
                outputserie=pd.Series(opdater,current_per) 
            else:
                print('Illegal operator in update:',operator,'Variable:',var)
                outputserie=pd.Series(np.NaN,current_per) 
            outputserie.name=var
            databank.loc[current_per,var]=outputserie
            if lprint:
                print('Update',operator,inputdata,start,slut)
                forspalte=str(max(6,len(var)))
                print(('{:<'+forspalte+'} {:>20} {:>20} {:>20}').format(var,'Before', 'After', 'Diff'))
                newdata=databank.loc[current_per,var]
                diff=newdata-orgdata
                for i in current_per:
                    print(('{:<'+forspalte+'} {:>20.4f} {:>20.4f} {:>20.4f}').format(str(i),orgdata[i],newdata[i],diff[i]))                


def tovarlag(var,lag):
    ''' creates a stringof var(lag) if lag else just lag '''
    if type(lag)==int:
        return f'{var}({lag:+})' if lag else var
    else:
        return f'{var}({lag})' if lag else var

def cutout(input,threshold=0.0):
    '''get rid of rows below treshold and returns the dataframe or serie '''
    if type(input)==pd.DataFrame:
        org_sum = input.sum(axis=0)   
        new = input.iloc[(abs(input) >= threshold).any(axis=1).values,:]
        if len(new) < len(input):
            new_sum = new.sum(axis=0)
            small = org_sum - new_sum
            small.name = 'Small'
            output = new.append(small)
        else:
            output = input 
        return output
    if type(input)==pd.Series:
        org_sum = input.sum()   
        new = input.iloc[(abs(input) >= threshold).values]
        if len(new) < len(input):
            new_sum = new.sum()
            small = pd.Series(org_sum - new_sum)
            small.index = ['Small']
            output = new.append(small)
        else:
            output=input
        return output
 
@contextmanager
def ttimer(input='test',show=True,short=False):
    '''
    A timer context manager, implemented using a
    generator function. This one will report time even if an exception occurs"""    

    Parameters
    ----------
    input : string, optional
        a name. The default is 'test'.
    show : bool, optional
        show the results. The default is True.
    short : bool, optional
        . The default is False.

    Returns
    -------
    None.

    '''
    
    start = time.time()
    if show and not short: print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ')
    try:
        yield
    finally:
        if show:  
            end = time.time()
            seconds = (end - start)
            minutes = seconds/60. 
            if minutes < 2.:
                afterdec='1' if seconds >= 10 else ('3' if seconds >= 1 else '10')
                print(f'{input} took       : {seconds:>{15},.{afterdec}f} Seconds')
            else:
                afterdec='1' if minutes >= 10 else '4'
                print(f'{input} took       : {minutes:>{15},.{afterdec}f} Minutes')

def finddec(df):
    ''' find a suitable number of decimal places from the magnitudes of a dataframe '''
    try:
        thismax = df.abs().max().max()
    except:
        thismax = df.abs().max()
        
    if thismax > 1000:
        outdec = 0
    elif thismax >  10: 
        outdec = 3
    else: 
        outdec = 6

    return outdec


def insertModelVar(dataframe, model=None):
    """Inserts all variables from model, not already in the dataframe.
    Model can be a list of models """ 
    if isinstance(model,list):
        imodel=model
    else:
        imodel = [model]

    myList=[]
    for item in imodel: 
        myList.extend(item.allvar.keys())
    manglervars = list(set(myList)-set(dataframe.columns))
    if len(manglervars):
        extradf = pd.DataFrame(0.0,index=dataframe.index,columns=manglervars).astype('float64')
        data = pd.concat([dataframe,extradf],axis=1)        
        return data
    else:
        return dataframe
  
def df_extend(df,add=5):
    '''Extends a Dataframe, assumes that the indes is of period_range type'''
    newindex = pd.period_range(df.index[0], periods=len(df)+add, freq=df.index.freq)
    return df.reindex(newindex,method='ffill')    
  
if __name__ == '__main__':
    #%% Test
    if not  'baseline' in locals() or 1 :    
        from modelclass import model 
        madam,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0,ljit=1,stringjit=0 )
        # make a simpel experimet VAT
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = madam(scenarie)
    
    update_var(baseline,'tg','=GROWTH',1,2021,2024,lprint=1)
