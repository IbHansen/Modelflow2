# -*- coding: utf-8 -*-
"""
This script runs a model with numba  


@author: hanseni 
"""

import sys
from modelclass import model 

try:
    mmodel,   basedf    = model.modelload(sys.argv[1],run=1,ljit=1,stringjit=False)
except:
    ...