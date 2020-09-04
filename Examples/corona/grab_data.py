# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:30:42 2020

@author: bruger


his module grab the age distribution and saves the dataframe  
"""
import pandas as pd 
import requests
from pathlib import Path


try:
    0/0  # uncomment to force read from github
    agedistribution_df = pd.read_excel('data/agedistribution.xlsx',index_col=0)
    print('Agedistribution read from file')
except: 
    ageurl = 'https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/ageDistribution.json'
    agelist =  requests.get(ageurl).json()['all']
    agedic  = {d['name'] : {'POP__'+str(age['ageGroup']): float(age['population']) for age in d['data']} 
                for d in agelist }
    agedistribution_df = pd.DataFrame(agedic).T.pipe(
        lambda df_ : df_.rename(columns = {c: c.replace('-','_').replace('+','_120') 
                                                    for c in df_.columns} ))
    agedistribution_df.loc[:,'N'] = agedistribution_df.sum(axis=1)
    Path("data").mkdir(parents=True, exist_ok=True)
    agedistribution_df.to_excel('data/agedistribution.xlsx')
    print('Agedistribution read from https://raw.githubusercontent.com/neherlab')
agedistribution_df.head()

sevurl = 'https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/severityDistributions.json'
sevreq = requests.get(sevurl)
sevlist = sevreq.json()['all']
sevdic  = {d['name']  : {i+'__'+inf['ageGroup'] : value for inf in d['data'] for i,value in inf.items()}   for d in sevlist }
sevdf = (pd.DataFrame(sevdic).T.
         pipe(lambda df_: df_.drop([c for c in df_.columns if c.startswith('ageGroup')],axis=1)).
         pipe(lambda df_ : df_.rename(columns = {c: c.replace('-','_').replace('+','_120').upper() for c in df_.columns} )).
         astype('float')).T