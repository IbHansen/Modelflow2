# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:50:07 2020

@author: bruger

Handeling of models with equiblibium condition 
for instance supply = demand 

"""
import pandas as pd
import numpy as np

from modelclass import model
import modelmanipulation as mp 

# Specify Model 
rdm = '''\
             demand = demand_ofset + demand_slope * price + demand_second * price**2
             supply = supply_ofset + supply_slope * price + supply_second * price**2
<endo=price> supply = demand  
'''
# rewrite the model as an un normalized model 0 = G(y,x)
fdm  = mp.un_normalize_simpel(rdm.upper())

# create a model instance 
mdm = model(fdm)

#%% Now we need some data, so we make a datataframe. 
# it has three rows with different parameters 

grunddf = pd.DataFrame(index=[1,2,3],columns=['DEMAND_OFSET', 'DEMAND_SLOPE',
 'DEMAND_SECOND', 'SUPPLY_OFSET', 'SUPPLY_SLOPE', 'SUPPLY_SECOND', 'PRICE'])

# Stuff some values into the Dataframe

demandparam = 1,-0.5,0.0
supplyparam = 0.0,0.5,0.

grunddf.loc[:,:] = demandparam+supplyparam+(4.,)
grunddf.loc[:,'DEMAND_OFSET']=[1.0,0.9,0.8]
grunddf = grunddf.astype('float')
print(grunddf)

# fill all the missing values 

# Run the model one experimet at a time and all experiments 

result = mdm(grunddf,max_iterations=30,silent=0,nonlin=2)
