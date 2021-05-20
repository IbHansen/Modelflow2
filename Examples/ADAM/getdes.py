# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:25:09 2021

@author: bruger
"""
from pathlib import Path
from modelclass import model 


with open(Path('adam/varlist.txt'),'rt') as f:
    vadam = f.read()
    
varinf = [var.split('\n')[0:2]  for var in vadam.split('----------\n') if len(var)] 
adam_var_descriptions = {var.upper() : des for var,des  in varinf}

madam,df = model.modelload('baseline.pcim',run=1)
madam.set_var_description(adam_var_descriptions)
madam.modeldump('baselinenew.pcim')
