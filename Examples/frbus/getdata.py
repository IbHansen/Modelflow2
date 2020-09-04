# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:33:54 2019

@author: hanseni
"""
import pandas as pd 
longbase = pd.read_csv(r'..\data_only_package\data_only_package\eviews_database\longbase.csv').pipe(lambda 
                     df:df.rename(columns={c:c.upper() for c in df.columns})).iloc[:-1,:]
dates = pd.period_range(start='1975q1',end='2125q4',freq='Q')
longbase.index = dates    
