# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:36:40 2018

@author: hanseni
"""

import re
import pandas as pd
import sys 
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np
from collections import namedtuple
import os 

import modelmanipulation as mp
import modelclass        as mc
import model_Excel       as me 
#import databank          as pb
from itertools import groupby,chain


path= 'imf-lcr\\'
if __name__ == '__main__'  :
    # read from the P: drive and save to local else read from local drive (just to work if not on the network) 
    tlcr = open(path+'lcr.txt').read()  # read the template model 
    t2lcr = doable(tlcr)
#%% 
                       
            
    print(t2lcr)
    flcr =     mp.explode(mp.tofrml(t2lcr))
    mlcr = mc.model(flcr)
    mlcr.drawall()
    
    exp='frml ddd ddd %0_ib %1__ha' 
    l  = ['{abekat}','{hest}']    
    ud = dosubst(l,exp)
