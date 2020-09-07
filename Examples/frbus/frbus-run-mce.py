# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:04:29 2019

@author: hanseni
"""

import pandas as pd

from modelclass import model
import modelmf

turbo = 0

mfrbusmce,basedf = model.modelload('mfrbusmce.json',run=1)

with model.timer('baseline newton all periods '):
    baseres = mfrbusmce(basedf,stats=0)
#%%
mfrbusmce.keep_solutions = {}
for shock in [0.0, 0.01]:
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
        altres = mfrbusmce(altdf,keep=f'shock {shock}')
mfrbusmce.keep_plot('rff',diff=1);
