# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:12:34 2019

@author: hanseni
"""
import pandas as pd

from modelclass import model 

#%%
if not 'madam' in locals(): 
    bank  = pd.read_pickle('adam/lang100_2017.pc') 
    fadam = open('adam/jul17x.txt','rt').read()
    madam = model(fadam)    
    madam.use_preorder = True
    #%%   

grund = madam(bank,2018,2030,conv='YR UL',max_iterations=100,alfa=0.4,stats=0,ljit=0,chunk=30,
              sim2= 0,debug=1)

#%% dump the results 
madam.modeldump('test.pcim')


madam2,bank = model.modelload('test.pcim',run=1)
