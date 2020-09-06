# -*- coding: utf-8 -*-
"""

This is a module for testing new features of the model class, but in a smaler file. 

Created on Sat Sep 29 06:03:35 2018

@author: hanseni


"""


import sys  
import time


import matplotlib.pyplot  as plt 
import matplotlib as mpl

import pandas as pd
from sympy import sympify,Symbol
from collections import defaultdict, namedtuple
import numpy as np 
import scipy as sp
import networkx as nx
import os
from subprocess import run 
import webbrowser as wb
import seaborn as sns 
import ipywidgets as ip
import inspect 
from itertools import chain, zip_longest
import fnmatch 
from IPython.display import SVG, display, Image, Math ,Latex, Markdown

try:
    from numba import jit
except:
    print('Numba not avaiable')
import itertools
from collections import namedtuple
from dataclasses import dataclass, field, asdict

import sys  
import time
import re

# print(f'name:{__name__} and package={__package__}!-' )
__package__ = 'ModelFlow'

import modelpattern as pt

import modelmanipulation as mp
import modeldiff as md 
from modelmanipulation import split_frml,udtryk_parse,find_statements,un_normalize_model,explode
from modelclass import model, insertModelVar
from modelinvert import targets_instruments
import modeljupyter as mj

import modelvis as mv
import modelmf
from modelhelp import tovarlag   

 
    
class newmodel(model):
    pass
    

 

class newvis(mv.vis):
    
    pass 

