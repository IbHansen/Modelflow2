# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:04:29 2019

@author: hanseni
"""

import pandas as pd
from numba import jit
import fnmatch 
import numpy as np
import os


from modelclass import model
import modelmf

turbo = 0

mfrbusmce,basedf = model.modelload('mfrbusmce.json')


with model.timer('baseline newton all periods '):
    baseres = mfrbusmce(basedf,'2020q1','2041q4',stats=0,max_iterations=70,silent=1,nchunk=50,newton_reset=0,
                                  ljit=turbo,timeit=0,nonlin=10,newtonalfa = 1.0, newtonnodamp=20,forcenum=0)
#%%
mfrbusmce.keep_solutions = {}
for shock in [0.0,-0.001]:
    altdf = baseres.copy()
    altdf=altdf.mfcalc(f'''\
    rffintay_aerr = rffintay_aerr + {shock}
    dmpex    = 0 
    dmprr    = 0
    dmptay   = 0
    dmptlr   = 0 
    dmpintay = 1
    
    dmpalt   = 0
    dmpgen   = 0
    ''')
        
        
    with model.timer(f'newton all periods,shock {shock:9} '):
        altres = mfrbusmce(altdf,keep=f'shock {shock:9}')
mfrbusmce.keep_plot('rff',diff=1);
